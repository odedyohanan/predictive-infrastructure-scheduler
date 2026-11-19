import numpy as np
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters  # noqa: E402
from pis.forecaster import have_full_history, load_world_model, rollout_batch  # noqa: E402

SEED = 1100


@pytest.fixture(scope="module")
def model():
    return load_world_model()


def _policies(n: int) -> list[dict]:
    from sched.controllers import isolated_policy
    from sched.dataset import perturb_policy, policy_dict
    spec = adapters.make_spec(SEED)
    rng = np.random.default_rng(3)
    p = isolated_policy()
    out = [policy_dict(p)]
    for _ in range(n - 1):
        p = perturb_policy(p, rng, spec)
        out.append(policy_dict(p))
    return out


def _history(n: int) -> list[tuple[dict, dict]]:
    """n real windows as (tel_t, policy applied in the window after t); the newest policy is a
    placeholder (rollout_batch replaces it by each candidate)."""
    from sim.policy import policy_from_dict
    sim = adapters.make_sim(SEED)
    pols = _policies(n + 1)
    tels = [sim.run_window(policy_from_dict(pols[0]))]
    for pd in pols[1:n]:
        tels.append(sim.run_window(policy_from_dict(pd)))
    return list(zip(tels, pols[1:]))


@pytest.mark.parametrize("n_hist", [12, 1, 2])
def test_batch_equals_single(model, n_hist):
    hist, known = _history(n_hist), adapters.known_of(adapters.make_spec(SEED))
    cands = _policies(4)
    batch = rollout_batch(model, hist, cands, known, 5)
    for i, c in enumerate(cands):
        single = rollout_batch(model, hist, [c], known, 5)
        assert np.allclose(batch[i], single[0], atol=1e-5)


def test_shape_and_bounds(model):
    hist, known = _history(12), adapters.known_of(adapters.make_spec(SEED))
    out = rollout_batch(model, hist, _policies(3), known, 10)
    assert out.shape == (3, 10, 39)
    lo, hi = model.bounds
    assert np.isfinite(out).all()
    assert (out >= lo - 1e-6).all() and (out <= hi + 1e-6).all()


def test_one_step_matches_model_predict_next(model):
    from worldmodel.features import clip_state, encode_input, encode_state
    hist, known = _history(12), adapters.known_of(adapters.make_spec(SEED))
    cand = _policies(5)[4]
    k = model.history
    rows = [encode_input(encode_state(t), p, known) for t, p in hist[-k:-1]]
    s = encode_state(hist[-1][0])
    rows.append(encode_input(s, cand, known))
    want = clip_state(model.predict_next(model.prepare(np.stack(rows)[None])[0], s)[0])
    want = np.clip(want, *model.bounds)
    got = rollout_batch(model, hist, [cand], known, 1)[0, 0]
    assert np.allclose(got, want, atol=1e-5)


def test_short_history_reports_not_full(model):
    assert not have_full_history(model, _history(model.history - 1))
    assert have_full_history(model, _history(model.history))


def test_short_history_still_rolls_out(model):
    known = adapters.known_of(adapters.make_spec(SEED))
    out = rollout_batch(model, _history(2), _policies(2), known, 3)
    assert out.shape == (2, 3, 39) and np.isfinite(out).all()
