import json

import numpy as np
import pytest

from tests._load import load_script

rule = load_script("b_incident_rule")
lead = load_script("b_lead_time")


# ------------------------------------------------------------------ k selection
def test_choose_k_smallest_within_limit():
    # 100 blocks: max run 1 in 80, 2 in 10, 3 in 6, 5 in 4 -> share(>=2)=20 %, (>=3)=10 %, (>=4)=4 %
    blocks = [[1]] * 80 + [[1, 2]] * 10 + [[3]] * 6 + [[5, 1]] * 4
    assert rule.choose_k(blocks, limit=0.05) == 4
    assert rule.choose_k(blocks, limit=0.10) == 3
    assert rule.choose_k(blocks, limit=0.20) == 2


def test_choose_k_at_least_two():
    assert rule.choose_k([[]] * 50, limit=0.05) == 2          # no violations at all
    assert rule.choose_k([[1]] * 50, limit=0.05) == 2


def test_choose_k_refuses_no_blocks():
    with pytest.raises(ValueError):
        rule.choose_k([], limit=0.05)


def test_block_run_lengths_after_warmup_full_blocks_only():
    viol = np.zeros(20 + 120 * 2 + 50, dtype=bool)
    viol[5:15] = True                    # warm-up: ignored
    viol[20 + 10:20 + 13] = True         # block 0: run 3
    viol[20 + 118:20 + 123] = True       # starts in block 0, crosses into block 1: one run of 5 in block 0
    viol[20 + 240:20 + 280] = True       # partial block: dropped
    viol[18:24] = True                   # starts in the warm-up: not counted
    out = rule.block_run_lengths(viol, warmup=20, block=120)
    assert len(out) == 2
    assert sorted(out[0]) == [3, 5]
    assert out[1] == []


def test_rule_config_keys():
    cfg = rule.rule_config(3)
    assert cfg == {"k": 3, "interval": 12, "majority": 0.5, "warmup": 20, "chronic": 0.5}


def test_rule_config_keeps_lead_time(tmp_path):
    p = tmp_path / "incident.json"
    p.write_text(json.dumps({"k": 2, "interval": 8, "min_lead_windows": 6, "H": 20}))
    rule.write_config(p, rule.rule_config(4))
    cfg = json.loads(p.read_text())
    assert cfg["k"] == 4 and cfg["interval"] == 16 and cfg["min_lead_windows"] == 6 and cfg["H"] == 20


# ------------------------------------------------------------------ lead time
def test_effect_window_first_significant_negative():
    diffs = np.tile([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0], (40, 1))   # mean 0: CI holds 0
    diffs[12:] -= 10.0                   # from window index 12 the intervention clearly lowers overdue
    assert lead.effect_window(diffs) == 13       # 1-based: windows after the intervention starts
    assert lead.effect_window(diffs, persist=1) == 13


def test_effect_window_single_noisy_window_counts():
    # one chance significant window is not an effect by default (3 consecutive windows are needed)
    diffs = np.zeros((10, 8))
    diffs[2] = [-1.0, -1.1, -0.9, -1.0, -1.2, -0.8, -1.0, -1.0]
    assert lead.effect_window(diffs) is None
    assert lead.effect_window(diffs, persist=1) == 3


def test_effect_window_persistence():
    diffs = np.zeros((10, 8))
    diffs[2] = -1.0
    diffs[5:] = -1.0
    assert lead.effect_window(diffs) == 6
    diffs[7:] = 0.0                      # significant in 5, 6 only
    assert lead.effect_window(diffs) is None


def test_effect_window_none_when_no_effect():
    rng = np.random.default_rng(1)
    assert lead.effect_window(rng.normal(0.0, 1.0, size=(40, 8)) + 5.0) is None   # worse, never better
    assert lead.effect_window(np.zeros((40, 8))) is None                            # identical arms


def test_effect_window_constant_negative():
    diffs = np.zeros((10, 8))
    diffs[3:] = -2.0                     # no spread: the interval collapses onto the mean
    assert lead.effect_window(diffs) == 4


def test_min_lead_and_horizon():
    assert lead.min_lead([3, 5, 8]) == 5
    assert lead.min_lead([3, 4]) == 4                # median 3.5 rounded up
    assert lead.horizon_for(5) == 20
    assert lead.horizon_for(13) == 26
    with pytest.raises(ValueError):
        lead.min_lead([])


def test_lead_state_needs_three_rising_windows():
    q = np.zeros(100)
    q[50:60] = np.arange(10) * 3.0       # backlog grows from window 50
    t = lead.lead_state(q, warmup=20, last=100 - 40)
    es = lead.early_signal_series(q)
    assert t is not None and np.all(es[t - 2:t + 1] > 0) and not np.all(es[t - 3:t] > 0)
    assert lead.lead_state(np.zeros(100), warmup=20, last=60) is None


def test_lead_state_respects_warmup_and_room():
    q = np.zeros(100)
    q[2:12] = np.arange(10) * 3.0        # only inside the warm-up
    assert lead.lead_state(q, warmup=20, last=60) is None
    q = np.zeros(100)
    q[70:80] = np.arange(10) * 3.0       # too late for a full horizon
    assert lead.lead_state(q, warmup=20, last=60) is None


# ------------------------------------------------------------------ analysis on a synthetic store
def _store(root, episodes):
    """A minimal store: shard s0 with episode rows and window rows (violation and active flags only)."""
    import pandas as pd
    d = root / "s0"
    d.mkdir(parents=True)
    for seed, kind, source, viol, active in episodes:
        n = len(viol)
        pd.DataFrame({"episode": [seed] * n, "window": np.arange(n), "slo_v2_violation": viol,
                      "deg_active": active, "a_queue_overdue": np.zeros(n, dtype=np.int64)}
                     ).to_parquet(d / f"windows-{seed}.parquet", index=False)
        pd.DataFrame([{"seed": seed, "kind": kind, "source": source, "windows": n}]).to_parquet(
            d / f"episodes-{seed}.parquet", index=False)
    (d / "done").write_text("")
    from pis.phase_b.history import Store
    return Store(root)


def test_analyse_synthetic_history(tmp_path):
    eps = []
    for i in range(20):                  # ordinary R6 episodes: 4 blocks each, one block in 20 with a run of 3
        v = np.zeros(600, dtype=bool)
        v[30:32] = True                  # a run of 2 in block 0 of every episode
        if i < 4:
            v[200:203] = True            # a run of 3 in 4 of 80 blocks (5 %)
        eps.append((3100 + i, "none", "r6", v, np.zeros(600, dtype=bool)))
    v = np.zeros(600, dtype=bool)
    v[300:310] = True                    # building-up incident in a degraded episode
    a = np.zeros(600, dtype=bool)
    a[250:] = True
    eps.append((3120, "leak", "random", v, a))
    v = np.zeros(600, dtype=bool)
    v[100:450] = True                    # chronic
    eps.append((3121, "load_ramp", "r3", v, np.ones(600, dtype=bool)))
    r = rule.analyse(_store(tmp_path, eps), list(range(3100, 3400)))
    assert r["k"] == 3 and r["rule"]["interval"] == 12
    assert r["quiet_sources"] == ["r6"] and r["quiet_episodes"] == 20 and r["blocks"] == 80
    assert r["per_kind"]["leak"] == {"building_up": 1}
    assert r["chronic"] == {"none": 0, "leak": 0, "load_ramp": 1}
    assert r["building_up"] == 2 and r["raise_magnitudes"]
    text = rule.note_text(r, 0.05)
    assert "k = 3" in text and "YES" in text


def test_analyse_adds_intervention_episodes_when_few(tmp_path):
    z = np.zeros(600, dtype=bool)
    eps = [(3100 + i, "none", "r6" if i < 5 else "r6_interventions", z, z) for i in range(10)]
    r = rule.analyse(_store(tmp_path, eps), list(range(3100, 3400)))
    assert r["quiet_sources"] == ["r6", "r6_interventions"] and r["quiet_episodes"] == 10


def test_rule_script_guards(tmp_path):
    with pytest.raises(SystemExit):
        rule.main(["--seeds", "3400-3410", "--label", "smoke", "--smoke-dir", "data/phase_b/smoke_t"])
    with pytest.raises(SystemExit):
        rule.main(["--label", "smoke", "--smoke-dir", str(tmp_path)])          # outside data/phase_b/smoke*
    with pytest.raises(SystemExit):
        rule.main(["--smoke-dir", "data/phase_b/smoke_t"])                     # needs the smoke label


def test_lead_script_guards(tmp_path):
    with pytest.raises(SystemExit):
        lead.main(["--seeds", "3450", "--label", "smoke", "--smoke-dir", "data/phase_b/smoke_t"])  # selection
    with pytest.raises(SystemExit):
        lead.main(["--states", "3"])                                           # smoke-only sizes
    with pytest.raises(SystemExit):                                            # real config from a smoke run
        lead.main(["--seeds", "3104", "--label", "smoke", "--smoke-dir", "data/phase_b/smoke_t"])
