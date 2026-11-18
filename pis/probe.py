"""Step 0 counterfactual probe: does the world model rank the consequences of actions like reality does?

Protocol: docs/protocols/step0-gate.md (frozen before any result). For each sampled real state, K candidate
policies are simulated from a deep copy of the real simulator (several reseeded branches each) and predicted
by the world model; per-state Spearman, pairwise sign accuracy and named-intervention signs are compared with
controls that cannot see the action.

Window convention (as in `pis.harness.run_scenario`): window index w = the w-th telemetry of the run. A state
at window w is the simulator right after window w ran, the policy that ran during it (`current`) and the
history up to it; a candidate is the policy for window w + 1.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from pis.adapters import (BaseScheduler, clone_sim, initial_policy, known_of, make_sim,
                          make_spec)
from pis.candidates import (N_CPUS, N_GPUS, _cpus, is_valid, policy_key,
                            random_valid_policies)
from pis.config import setup_imports
from pis.forecaster import rollout_batch
from pis.reward import reward_from_state

setup_imports()

from sim.controllers import AdaptiveController  # noqa: E402
from sim.dataset import policy_dict  # noqa: E402
from sim.policy import policy_from_dict  # noqa: E402
from worldmodel.features import (POLICY_NAMES, STATE_NAMES, decode_state,  # noqa: E402
                                 encode_state)

BASE_CKPT = "models/rl_r6/final.pt"
DRIVERS = ("base_greedy", "base_sampled", "adaptive", "adaptive_perturbed")
INTERVENTIONS = ("take_from_B", "gpu_to_B", "isolate_A")
QUANTITIES = ("a.p99_ms", "b.rate", "c.work_core_ms")
LAM = 10.0                 # SLO weight the RL scheduler was trained with
MIN_WINDOW = 8             # full 8-window GRU history
H2 = 4                     # secondary horizon (windows the candidate is held)
TRUTH_RESEED = 7_000_000   # branch j of every candidate uses reseed TRUTH_RESEED + j
PERTURB_P, PERTURB_STRENGTH = 0.3, 0.5
MIN_K = 8
GATE = {"min_resolvable_states": 200, "min_intervention_cases": 100, "median_spearman": 0.50,
        "margin_lower_bound": 0.25, "pairwise_sign_acc": 0.70, "intervention_sign_acc": 0.70}
_DRIVER_RNG, _CAND_RNG = 51_000, 52_000


# ------------------------------------------------------------------ states
@dataclass(frozen=True, order=True)
class StateRef:
    """A real state: scenario seed, window index and the controller that drove the run up to it."""
    seed: int
    window: int
    driver: str

    def rebuild(self, duration_s: float = 60.0) -> "ProbeState":
        return next(iter_states([self], duration_s))

    def as_dict(self) -> dict:
        return {"seed": self.seed, "window": self.window, "driver": self.driver}


@dataclass
class ProbeState:
    ref: StateRef
    spec: object
    sim: object                 # simulator right after window `ref.window` ran
    tel: dict                   # telemetry of that window
    current: dict               # policy that ran during it
    history: list = field(default_factory=list)   # forecaster convention, newest = (tel, current)
    known: dict = field(default_factory=dict)
    base_rates: list = field(default_factory=list)  # RL scheduler's rate history before this window

    @property
    def window_s(self) -> float:
        return float(self.tel["window_s"])


def n_windows(seed: int, duration_s: float = 60.0) -> int:
    return int(round(duration_s / make_spec(seed).policy_period_s))


def _driver(name: str, seed: int, spec):
    """Returns decide(tel, current) -> next policy dict. Called exactly once per window."""
    if name in ("base_greedy", "base_sampled"):
        base = BaseScheduler(BASE_CKPT, spec)
        rng = np.random.default_rng([_DRIVER_RNG, seed])

        def decide(tel, current):
            greedy, samples = base.propose(tel, current, 1 if name == "base_sampled" else 0, rng)
            if name == "base_sampled" and is_valid(samples[0]):
                return samples[0]
            return greedy
        return decide
    if name in ("adaptive", "adaptive_perturbed"):
        ctrl = AdaptiveController(spec)
        prng = np.random.default_rng([_DRIVER_RNG + 1, seed])

        def decide(tel, current):
            p = policy_dict(ctrl.decide(tel))
            if name == "adaptive_perturbed" and prng.random() < PERTURB_P:
                p = random_valid_policies(spec, p, 1, PERTURB_STRENGTH, prng)[0]
            return p
        return decide
    raise ValueError(f"unknown driver {name!r}")


def drive(seed: int, driver: str, stop_windows, duration_s: float = 60.0):
    """Run one trajectory; yield a ProbeState (deep copies) at every window in `stop_windows`.

    If `stop_windows` is None, yield every window. The loop mirrors `pis.harness.run_scenario`.
    """
    spec = make_spec(seed)
    sim = make_sim(seed, duration_s)
    decide = _driver(driver, seed, spec)
    known = known_of(spec)
    stops = None if stop_windows is None else set(stop_windows)
    last = None if stops is None else max(stops)
    current = initial_policy()
    history: list[tuple[dict, dict]] = []
    rates: list[float] = []
    w = 0
    tel = sim.run_window(policy_from_dict(current))
    while True:
        if stops is None or w in stops:
            yield ProbeState(StateRef(seed, w, driver), spec, clone_sim(sim), copy.deepcopy(tel),
                             copy.deepcopy(current), copy.deepcopy(history + [(tel, current)]),
                             known, list(rates))
        if (last is not None and w >= last) or sim.done():
            return
        policy = decide(tel, current)
        rates.append(tel["a"]["rate"])
        history.append((tel, policy))
        current = policy
        tel = sim.run_window(policy_from_dict(current))
        w += 1


def iter_states(refs: list[StateRef], duration_s: float = 60.0):
    """ProbeStates for `refs`, one trajectory run per (seed, driver), in sorted order."""
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for r in refs:
        groups[(r.seed, r.driver)].append(r.window)
    for (seed, drv), windows in sorted(groups.items()):
        yield from drive(seed, drv, sorted(set(windows)), duration_s)


def sample_states(seeds: list[int], n: int, rng: np.random.Generator, duration_s: float = 60.0,
                  horizon: int = H2) -> list[StateRef]:
    """`n` distinct states spread evenly over (seed, driver) trajectories, windows uniform.

    Windows run from MIN_WINDOW to the last one that leaves `horizon` real windows after it.
    """
    pairs = [(s, d) for s in seeds for d in DRIVERS]
    order = [pairs[i] for i in rng.permutation(len(pairs))]
    out: list[StateRef] = []
    seen: set[StateRef] = set()
    i = tries = 0
    while len(out) < n:
        seed, drv = order[i % len(order)]
        hi = n_windows(seed, duration_s) - 1 - horizon
        ref = StateRef(seed, int(rng.integers(MIN_WINDOW, hi + 1)), drv)
        i += 1
        tries += 1
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
        if tries > 100 * n:
            raise ValueError("not enough distinct states")
    return out


# ------------------------------------------------------------------ candidates and interventions
def make_candidates(state: ProbeState, k: int) -> list[dict]:
    """Protocol section 3: keep-current, greedy, 2 samples, then random perturbations; distinct, valid."""
    r = state.ref
    rng = np.random.default_rng([_CAND_RNG, r.seed, r.window, DRIVERS.index(r.driver)])
    base = BaseScheduler(BASE_CKPT, state.spec)
    base.history = list(state.base_rates)
    greedy, samples = base.propose(state.tel, state.current, 2, rng)
    pool = [copy.deepcopy(state.current), greedy, *samples]
    for s in (0.0, 0.5, 1.0):
        pool += random_valid_policies(state.spec, state.current, 2, s, rng)
    out, seen = [], set()

    def add(p):
        key = policy_key(p)
        if key not in seen and is_valid(p) and len(out) < k:
            seen.add(key)
            out.append(p)
    for p in pool:
        add(p)
    tries = 0
    while len(out) < k and tries < 50:
        add(random_valid_policies(state.spec, state.current, 1, 0.5, rng)[0])
        tries += 1
    return out


def _set_cpus(wp: dict, cpus) -> None:
    cpus = sorted(set(cpus))
    wp["preferred_cpus"] = [] if len(cpus) == N_CPUS else cpus


def _move_share(p: dict, src: str, dst: str, d: int) -> None:
    d = min(d, p["workloads"][src]["cpu_share"] - 10)
    p["workloads"][src]["cpu_share"] -= d
    p["workloads"][dst]["cpu_share"] += d


def _take_from_b(cur: dict) -> dict | None:
    p = copy.deepcopy(cur)
    wl = p["workloads"]
    b = _cpus(wl["B"])
    _move_share(p, "B", "A", 20)
    if len(b) > 1:
        keep, drop = b[len(b) // 2:], b[:len(b) // 2]
        _set_cpus(wl["B"], keep)
        _set_cpus(wl["A"], set(_cpus(wl["A"])) | set(drop))
    return p if policy_key(p) != policy_key(cur) else None


def _give_to_b(cur: dict) -> dict:
    """Reverse of take_from_B, for a B already at its minimum: +20 share and one more CPU."""
    p = copy.deepcopy(cur)
    wl = p["workloads"]
    donor = max(("A", "C"), key=lambda w: wl[w]["cpu_share"])
    _move_share(p, donor, "B", 20)
    b = set(_cpus(wl["B"]))
    extra = [c for c in range(N_CPUS) if c not in b]
    if extra:
        _set_cpus(wl["B"], b | {extra[0]})
    return p


def _gpu_to_b(cur: dict) -> tuple[dict, dict]:
    wl = cur["workloads"]
    free = [g for g in range(N_GPUS) if g not in wl["B"]["gpus"]]
    if free:
        p = copy.deepcopy(cur)
        g = free[0]
        for w in ("A", "B", "C"):
            p["workloads"][w]["gpus"] = [x for x in p["workloads"][w]["gpus"] if x != g]
        p["workloads"]["B"]["gpus"] = sorted(p["workloads"]["B"]["gpus"] + [g])
        return copy.deepcopy(cur), p
    before = copy.deepcopy(cur)
    g = N_GPUS - 1
    before["workloads"]["B"]["gpus"] = [x for x in wl["B"]["gpus"] if x != g]
    before["workloads"]["C"]["gpus"] = sorted(before["workloads"]["C"]["gpus"] + [g])
    return before, copy.deepcopy(cur)


def _a_isolated(p: dict) -> bool:
    wl = p["workloads"]
    return not set(_cpus(wl["A"])) & (set(_cpus(wl["B"])) | set(_cpus(wl["C"])))


def _isolate_a(cur: dict) -> dict:
    p = copy.deepcopy(cur)
    wl = p["workloads"]
    a = set(_cpus(wl["A"]))
    if len(a) == N_CPUS:
        a = {0, 1, 2, 3}
    _set_cpus(wl["A"], a)
    rest = set(range(N_CPUS)) - a
    for w in ("B", "C"):
        left = set(_cpus(wl[w])) - a
        _set_cpus(wl[w], left or rest)
    uncovered = rest - set(_cpus(wl["B"])) - set(_cpus(wl["C"]))
    if uncovered:                          # every CPU must stay used: C takes what nobody has
        _set_cpus(wl["C"], set(_cpus(wl["C"])) | uncovered)
    return p


def _share_a(cur: dict) -> dict:
    """Reverse of isolate_A: B and C may also run on A's CPUs."""
    p = copy.deepcopy(cur)
    wl = p["workloads"]
    a = set(_cpus(wl["A"]))
    for w in ("B", "C"):
        _set_cpus(wl[w], set(_cpus(wl[w])) | a)
    return p


def named_interventions(current: dict) -> dict[str, tuple[dict, dict]]:
    """Protocol section 5: name -> (before, after); one of the two is always `current`."""
    taken = _take_from_b(current)
    take = (copy.deepcopy(current), taken) if taken is not None else (_give_to_b(current),
                                                                       copy.deepcopy(current))
    iso = ((_share_a(current), copy.deepcopy(current)) if _a_isolated(current)
           else (copy.deepcopy(current), _isolate_a(current)))
    return {"take_from_B": take, "gpu_to_B": _gpu_to_b(current), "isolate_A": iso}


# ------------------------------------------------------------------ truth
def simulate_policies(state: ProbeState, policies: list[dict], n_seeds: int, horizon: int) -> np.ndarray:
    """Encoded real states [K, n_seeds, horizon, 39]; branch j uses reseed TRUTH_RESEED + j for all."""
    out = np.empty((len(policies), n_seeds, horizon, len(STATE_NAMES)), dtype=np.float32)
    for i, p in enumerate(policies):
        pol = policy_from_dict(p)
        for j in range(n_seeds):
            sim = clone_sim(state.sim, reseed=TRUTH_RESEED + j)
            for h in range(horizon):
                out[i, j, h] = encode_state(sim.run_window(pol))
    return out


def rewards_of(spec, enc: np.ndarray, window_s: float, lam: float = LAM) -> np.ndarray:
    """Reward of every encoded state; shape enc.shape[:-1]."""
    flat = enc.reshape(-1, enc.shape[-1])
    r = np.array([reward_from_state(spec, decode_state(v), window_s, lam) for v in flat])
    return r.reshape(enc.shape[:-1])


def _state(ref_or_state) -> ProbeState:
    return ref_or_state if isinstance(ref_or_state, ProbeState) else ref_or_state.rebuild()


def true_outcomes(ref, policies: list[dict], n_seeds: int) -> np.ndarray:
    """h = 1 rewards from the real simulator, [K, n_seeds]."""
    st = _state(ref)
    return rewards_of(st.spec, simulate_policies(st, policies, n_seeds, 1), st.window_s)[:, :, 0]


def floor_from(per_seed: np.ndarray) -> float:
    """Protocol section 4: 2 * s_pool * sqrt(2 / n_seeds), at least 1e-6."""
    per_seed = np.asarray(per_seed, float)
    n = per_seed.shape[1]
    s_pool = float(np.sqrt(np.mean(per_seed.var(axis=1, ddof=1))))
    return max(2.0 * s_pool * np.sqrt(2.0 / n), 1e-6)


def noise_floor(ref, policies: list[dict], n_seeds: int, outcomes: np.ndarray | None = None) -> float:
    if outcomes is None:
        outcomes = true_outcomes(ref, policies, n_seeds)
    return floor_from(outcomes)


# ------------------------------------------------------------------ predictors
class Predictor(Protocol):
    def predict(self, state: ProbeState, policies: list[dict], horizon: int) -> np.ndarray:
        """Encoded predicted states [K, S, horizon, 39]; S = samples (ensemble members, 1 for one GRU)."""


class ActionBlindModel:
    """A world model whose policy inputs (every history row) are set to their training mean."""

    def __init__(self, model):
        self.model = model
        n_s = len(STATE_NAMES)
        self._cols = slice(n_s, n_s + len(POLICY_NAMES))

    def __getattr__(self, name):
        if name == "model":                # not yet set (unpickling)
            raise AttributeError(name)
        return getattr(self.model, name)

    def prepare(self, H):
        H = np.array(self.model.prepare(H), dtype=np.float32, copy=True)
        H[..., self._cols] = np.asarray(self.model.x_norm.mean, np.float32)[self._cols]
        return H


class GRUPredictor:
    def __init__(self, models, action_blind: bool = False):
        models = list(models) if isinstance(models, (list, tuple)) else [models]
        self.models = [ActionBlindModel(m) if action_blind else m for m in models]

    def predict(self, state, policies, horizon):
        outs = [rollout_batch(m, state.history, policies, state.known, horizon) for m in self.models]
        return np.stack(outs, axis=1)


class PersistencePredictor:
    """Next state = current state, whatever the action."""

    def predict(self, state, policies, horizon):
        s0 = encode_state(state.tel)
        return np.broadcast_to(s0, (len(policies), 1, horizon, s0.shape[0])).copy()


def as_predictor(model_or_ensemble, action_blind: bool = False):
    if hasattr(model_or_ensemble, "predict") and not hasattr(model_or_ensemble, "predict_next"):
        if action_blind:
            raise ValueError("action_blind needs a world model, not a custom predictor")
        return model_or_ensemble
    return GRUPredictor(model_or_ensemble, action_blind)


# ------------------------------------------------------------------ one state
def _scores(rewards: np.ndarray, horizons) -> dict[str, np.ndarray]:
    """rewards [..., H] -> {str(h): mean of the first h windows}."""
    return {str(h): rewards[..., :h].mean(-1) for h in horizons}


def evaluate_state(state: ProbeState, predictors: dict, k: int, n_seeds: int,
                   horizons=(1, H2)) -> dict | None:
    """Truth and predictions for one state, as a JSON-ready record (None if too few candidates)."""
    horizon = max(horizons)
    cands = make_candidates(state, k)
    if len(cands) < min(MIN_K, k):
        return None
    iv = named_interventions(state.current)
    pool, index = [], {}
    for p in cands + [p for pair in iv.values() for p in pair]:
        key = policy_key(p)
        if key not in index:
            index[key] = len(pool)
            pool.append(p)
    ci = [index[policy_key(p)] for p in cands]
    true_enc = simulate_policies(state, pool, n_seeds, horizon)            # [P, n, H, 39]
    true_r = rewards_of(state.spec, true_enc, state.window_s)              # [P, n, H]
    true_s = _scores(true_r, horizons)                                     # h -> [P, n]
    true_dec = [[decode_state(true_enc[i, j, 0]) for j in range(n_seeds)] for i in range(len(pool))]
    rec = {"ref": state.ref.as_dict(), "k": len(cands),
           "true": {h: v[ci].mean(1).tolist() for h, v in true_s.items()},
           "true_per_seed": {h: v[ci].tolist() for h, v in true_s.items()},
           "floor": {h: floor_from(v[ci]) for h, v in true_s.items()},
           "pred": {}, "interventions": {}}
    preds = {}
    for name, pr in predictors.items():
        enc = pr.predict(state, pool, horizon)                             # [P, S, H, 39]
        r = rewards_of(state.spec, enc, state.window_s)
        rec["pred"][name] = {h: v[ci].mean(1).tolist() for h, v in _scores(r, horizons).items()}
        preds[name] = enc
    for iname, (before, after) in iv.items():
        b, a = index[policy_key(before)], index[policy_key(after)]
        entry = {"true": {}, "floor": {}, "pred": {n: {} for n in predictors}}
        for q in QUANTITIES:
            vb = np.array([d[q] for d in true_dec[b]])
            va = np.array([d[q] for d in true_dec[a]])
            entry["true"][q] = float(va.mean() - vb.mean())
            entry["floor"][q] = float(max(2 * np.sqrt((vb.var(ddof=1) + va.var(ddof=1)) / n_seeds), 1e-9))
            for name, enc in preds.items():
                pa = np.mean([decode_state(enc[a, s, 0])[q] for s in range(enc.shape[1])])
                pb = np.mean([decode_state(enc[b, s, 0])[q] for s in range(enc.shape[1])])
                entry["pred"][name][q] = float(pa - pb)
        rec["interventions"][iname] = entry
    return rec


# ------------------------------------------------------------------ metrics
def _ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    _, inv = np.unique(x, return_inverse=True)
    sums = np.bincount(inv, weights=r)
    counts = np.bincount(inv)
    return (sums / counts)[inv]


def spearman(pred, true) -> float:
    """Rank correlation with average ranks for ties; 0 if either side is constant."""
    rp, rt = _ranks(pred), _ranks(true)
    if np.ptp(rp) == 0 or np.ptp(rt) == 0:
        return 0.0
    rp, rt = rp - rp.mean(), rt - rt.mean()
    return float((rp * rt).sum() / np.sqrt((rp ** 2).sum() * (rt ** 2).sum()))


def _pairs(true: np.ndarray, floor: float):
    k = len(true)
    allp = [(i, j) for i in range(k) for j in range(i + 1, k)]
    return allp, [(i, j) for i, j in allp if abs(true[i] - true[j]) >= floor]


def _median(xs) -> float | None:
    return float(np.median(xs)) if len(xs) else None


def bootstrap_lower(a: list[float], b: list[float], reps: int = 2000, seed: int = 0) -> float | None:
    """95% lower bound of median(a) - median(b), resampling states (same resample for both)."""
    if not a:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(a), size=(reps, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float(np.quantile(d, 0.025))


def random_ranking_null(records: list[dict], h: str, draws: int = 1000, seed: int = 0) -> float | None:
    """95th percentile of the median Spearman of random orderings over the resolvable states."""
    trues = [np.array(r["true"][h]) for r in records if _resolvable(r, h)]
    if not trues:
        return None
    rng = np.random.default_rng(seed)
    meds = [np.median([spearman(rng.permutation(len(t)), t) for t in trues]) for _ in range(draws)]
    return float(np.quantile(meds, 0.95))


def _resolvable(rec: dict, h: str) -> bool:
    true = np.array(rec["true"][h])
    allp, res = _pairs(true, rec["floor"][h])
    return bool(allp) and len(res) >= len(allp) / 2


def summarise(records: list[dict], names: list[str], horizons=(1, H2)) -> dict:
    """Per predictor and horizon: Spearman (per state, medians), pairwise and intervention sign accuracy."""
    out = {}
    for name in names:
        out[name] = {}
        for hh in horizons:
            h = str(hh)
            rhos, rhos_res, truth_const = [], [], []
            correct = n_pairs = 0
            mae = []
            for rec in records:
                true, pred = np.array(rec["true"][h]), np.array(rec["pred"][name][h])
                rho = spearman(pred, true)
                rhos.append(rho)
                truth_const.append(bool(np.ptp(true) == 0))
                if _resolvable(rec, h):
                    rhos_res.append(rho)
                _, res = _pairs(true, rec["floor"][h])
                for i, j in res:
                    n_pairs += 1
                    correct += int(np.sign(pred[i] - pred[j]) == np.sign(true[i] - true[j]))
                mae += np.abs(pred - true).tolist()
            m = {"spearman_per_state": rhos, "truth_constant": truth_const,
                 "spearman_resolvable": rhos_res,
                 "median_spearman": _median(rhos_res), "median_spearman_all": _median(rhos),
                 "n_states": len(records), "n_resolvable_states": len(rhos_res),
                 "pairwise_sign_acc": float(correct / n_pairs) if n_pairs else None,
                 "n_resolvable_pairs": n_pairs,
                 "reward_mae": float(np.mean(mae)) if mae else None}
            if h == "1":
                m["interventions"] = _intervention_metrics(records, name)
            out[name][f"h{h}"] = m
    return out


def _intervention_metrics(records: list[dict], name: str) -> dict:
    out = {}
    for iname in INTERVENTIONS:
        ok = n = states = states_ok = 0
        per_q = {q: [0, 0] for q in QUANTITIES}
        for rec in records:
            e = rec["interventions"].get(iname)
            if not e:
                continue
            hits = []
            for q in QUANTITIES:
                t, f, p = e["true"][q], e["floor"][q], e["pred"][name][q]
                if abs(t) >= f:
                    hit = bool(np.sign(p) == np.sign(t) and p != 0)
                    hits.append(hit)
                    per_q[q][0] += hit
                    per_q[q][1] += 1
            ok += sum(hits)
            n += len(hits)
            if hits:
                states += 1
                states_ok += int(sum(hits) > len(hits) / 2)
        out[iname] = {"sign_acc": ok / n if n else None, "n_cases": n,
                      "state_majority_acc": states_ok / states if states else None,
                      "per_quantity": {q: (c / t if t else None, t) for q, (c, t) in per_q.items()}}
    return out


def gate(summary: dict, records: list[dict], model: str = "model",
         controls=("action_blind", "persistence")) -> dict:
    """Protocol section 7, on h = 1."""
    m = summary[model]["h1"]
    checks = {}
    iv = m["interventions"]
    g0 = (m["n_resolvable_states"] >= GATE["min_resolvable_states"]
          and all(iv[i]["n_cases"] >= GATE["min_intervention_cases"] for i in INTERVENTIONS))
    checks["G0 enough evidence"] = (g0, f"{m['n_resolvable_states']} resolvable states; cases "
                                        + ", ".join(f"{i} {iv[i]['n_cases']}" for i in INTERVENTIONS))
    med = m["median_spearman"]
    checks["G1 median Spearman >= 0.50"] = (med is not None and med >= GATE["median_spearman"],
                                            f"{med}")
    lbs = {}
    for c in controls:
        lbs[c] = bootstrap_lower(m["spearman_resolvable"], summary[c]["h1"]["spearman_resolvable"])
    checks["G2 lower bound vs controls >= 0.25"] = (
        all(v is not None and v >= GATE["margin_lower_bound"] for v in lbs.values()),
        ", ".join(f"{c} {v}" for c, v in lbs.items()))
    pw = m["pairwise_sign_acc"]
    checks["G3 pairwise sign accuracy >= 0.70"] = (pw is not None and pw >= GATE["pairwise_sign_acc"],
                                                   f"{pw}")
    checks["G4 interventions sign accuracy >= 0.70"] = (
        all(iv[i]["sign_acc"] is not None and iv[i]["sign_acc"] >= GATE["intervention_sign_acc"]
            for i in INTERVENTIONS),
        ", ".join(f"{i} {iv[i]['sign_acc']}" for i in INTERVENTIONS))
    passed = all(v[0] for v in checks.values())
    verdict = "PASS" if passed else ("INCONCLUSIVE" if not g0 else "FAIL")
    return {"verdict": verdict, "checks": checks, "lower_bounds": lbs,
            "random_ranking_p95": random_ranking_null(records, "1")}


# ------------------------------------------------------------------ drivers of the whole probe
_WORKER: dict = {}


def _init_worker(predictors, k, n_seeds, horizons, duration_s):
    import torch
    torch.set_num_threads(1)
    _WORKER.update(predictors=predictors, k=k, n_seeds=n_seeds, horizons=horizons, duration_s=duration_s)


def _run_group(job):
    seed, drv, windows = job
    w = _WORKER
    out = []
    for st in drive(seed, drv, windows, w["duration_s"]):
        rec = evaluate_state(st, w["predictors"], w["k"], w["n_seeds"], w["horizons"])
        out.append(rec if rec is not None else {"ref": st.ref.as_dict(), "dropped": True})
    return out


def probe_many(predictors: dict, refs: list[StateRef], k: int, n_seeds: int, horizons=(1, H2),
               procs: int = 1, duration_s: float = 60.0, progress=None) -> tuple[list[dict], list[dict]]:
    """Evaluate every predictor on the same states and candidates. Returns (records, dropped)."""
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for r in refs:
        groups[(r.seed, r.driver)].append(r.window)
    jobs = [(s, d, sorted(set(ws))) for (s, d), ws in sorted(groups.items())]
    args = (predictors, k, n_seeds, tuple(horizons), duration_s)
    results = []
    if procs <= 1:
        _init_worker(*args)
        for i, job in enumerate(jobs, 1):
            results += _run_group(job)
            if progress:
                progress(i, len(jobs))
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(procs, initializer=_init_worker, initargs=args) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_group, jobs), 1):
                results += res
                if progress:
                    progress(i, len(jobs))
    results.sort(key=lambda r: (r["ref"]["seed"], r["ref"]["driver"], r["ref"]["window"]))
    return [r for r in results if not r.get("dropped")], [r for r in results if r.get("dropped")]


def probe(model_or_ensemble, refs: list[StateRef], k: int, n_seeds: int, action_blind: bool = False,
          horizon: int = H2) -> dict:
    """Single-predictor probe: metrics at h = 1 (and h = `horizon` if > 1) plus the per-state records."""
    horizons = (1,) if horizon == 1 else (1, horizon)
    pred = as_predictor(model_or_ensemble, action_blind)
    records, dropped = probe_many({"model": pred}, refs, k, n_seeds, horizons)
    out = summarise(records, ["model"], horizons)["model"]
    out.update(records=records, dropped=dropped)
    return out
