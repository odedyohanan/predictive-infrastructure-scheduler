import copy

import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.candidates import _cpus, is_valid, policy_key, random_valid_policies  # noqa: E402
from pis.forecaster import load_world_model  # noqa: E402
from pis.probe import (DRIVERS, INTERVENTIONS, PersistencePredictor, StateRef,  # noqa: E402
                       named_interventions, noise_floor, probe, sample_states,
                       simulate_policies, summarise, true_outcomes)

REF = StateRef(seed=1100, window=9, driver="base_greedy")
REFS = [REF, StateRef(seed=1101, window=12, driver="adaptive")]


@pytest.fixture(scope="module")
def state():
    return REF.rebuild()


def _policies(state, n=3):
    rng = np.random.default_rng(0)
    return [state.current, *random_valid_policies(state.spec, state.current, n - 1, 1.0, rng)]


class OraclePredictor:
    """Stub 'model' that returns the true simulated states: a perfect predictor."""

    def __init__(self, n_seeds):
        self.n_seeds = n_seeds

    def predict(self, state, policies, horizon):
        return simulate_policies(state, policies, self.n_seeds, horizon)


def test_true_outcomes_reproducible(state):
    pols = _policies(state)
    a = true_outcomes(REF, pols, n_seeds=2)
    b = true_outcomes(REF.rebuild(), pols, n_seeds=2)
    assert a.shape == (3, 2)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_noise_floor_positive(state):
    f = noise_floor(state, _policies(state), n_seeds=3)
    assert np.isfinite(f) and f > 0


def test_spearman_perfect_model_is_one():
    res = probe(OraclePredictor(2), REFS, k=6, n_seeds=2, horizon=1)
    rhos = [r for r, const in zip(res["h1"]["spearman_per_state"], res["h1"]["truth_constant"]) if not const]
    assert rhos and all(r == pytest.approx(1.0) for r in rhos)
    assert res["h1"]["pairwise_sign_acc"] in (1.0, None)
    for name in INTERVENTIONS:
        assert res["h1"]["interventions"][name]["sign_acc"] in (1.0, None)


def test_action_blind_control_scores_near_zero_on_stub_that_ignores_actions():
    res = probe(PersistencePredictor(), REFS, k=6, n_seeds=2, horizon=1)
    assert all(r == 0.0 for r in res["h1"]["spearman_per_state"])
    assert res["h1"]["pairwise_sign_acc"] in (0.0, None)


def test_action_blind_gru_cannot_tell_candidates_apart():
    res = probe(load_world_model(), REFS[:1], k=6, n_seeds=2, action_blind=True, horizon=1)
    assert res["h1"]["spearman_per_state"] == [0.0]
    preds = np.array(res["records"][0]["pred"]["model"]["1"])
    assert np.ptp(preds) == 0.0


def _layouts():
    spec = adapters.make_spec(1100)
    rng = np.random.default_rng(1)
    init = adapters.initial_policy()
    open_all = copy.deepcopy(init)
    for w in "ABC":
        open_all["workloads"][w]["preferred_cpus"] = []
    b_min = copy.deepcopy(init)
    b_min["workloads"]["B"].update(preferred_cpus=[7], cpu_share=10)
    b_min["workloads"]["A"]["cpu_share"] = 80
    b_min["workloads"]["C"]["cpu_share"] = 10
    b_min["workloads"]["C"]["preferred_cpus"] = [4, 5, 6]
    no_gpu_b = copy.deepcopy(init)
    no_gpu_b["workloads"]["B"]["gpus"] = []
    no_gpu_b["workloads"]["C"]["gpus"] = [0, 1]
    out = [init, open_all, b_min, no_gpu_b]
    for start in (init, open_all, b_min, no_gpu_b):
        for s in (0.0, 0.5, 1.0):
            out += random_valid_policies(spec, start, 15, s, rng)
    assert all(is_valid(p) for p in out)
    return out


def test_named_interventions_defined_for_all_workload_layouts():
    for cur in _layouts():
        iv = named_interventions(cur)
        assert set(iv) == set(INTERVENTIONS)
        for name, (before, after) in iv.items():
            assert is_valid(before) and is_valid(after), name
            assert policy_key(before) != policy_key(after), name
            assert policy_key(cur) in (policy_key(before), policy_key(after)), name
            wb, wa = before["workloads"], after["workloads"]
            if name == "take_from_B":
                assert (wa["B"]["cpu_share"] < wb["B"]["cpu_share"]
                        or set(_cpus(wa["B"])) < set(_cpus(wb["B"])))
            elif name == "gpu_to_B":
                assert len(wa["B"]["gpus"]) == len(wb["B"]["gpus"]) + 1
            else:
                a_after = set(_cpus(wa["A"]))
                assert not a_after & (set(_cpus(wa["B"])) | set(_cpus(wa["C"])))
                a_before = set(_cpus(wb["A"]))
                assert a_before & (set(_cpus(wb["B"])) | set(_cpus(wb["C"])))


def test_sample_states_diverse_and_in_range():
    seeds = list(range(1100, 1110))
    refs = sample_states(seeds, 40, np.random.default_rng(0))
    assert len(refs) == 40 and len(set(refs)) == 40
    assert {r.driver for r in refs} == set(DRIVERS)
    assert {r.seed for r in refs} <= set(seeds) and len({r.seed for r in refs}) == len(seeds)
    assert all(8 <= r.window <= 115 for r in refs)
    again = sample_states(seeds, 40, np.random.default_rng(0))
    assert refs == again


def test_pairs_below_noise_floor_are_excluded():
    # One synthetic state: candidates 0 and 1 differ by less than the floor, 2 is far away.
    per_seed = np.array([[1.0, 1.02], [1.01, 1.03], [3.0, 3.02]])
    rec = {"ref": {"seed": 1, "window": 9, "driver": "adaptive"},
           "true": {"1": per_seed.mean(1).tolist()},
           "true_per_seed": {"1": per_seed.tolist()},
           "floor": {"1": 0.5},
           "pred": {"m": {"1": [2.0, 1.0, 3.0]}},      # wrong order on the unresolvable pair only
           "interventions": {}}
    out = summarise([rec], ["m"], horizons=(1,))["m"]["h1"]
    assert out["n_resolvable_pairs"] == 2
    assert out["pairwise_sign_acc"] == 1.0
