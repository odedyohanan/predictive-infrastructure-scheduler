import dataclasses
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from pis.config import setup_imports

setup_imports()

from pis import adapters, shards  # noqa: E402
from pis.candidates import is_valid, policy_key  # noqa: E402
from pis.phase_b import history, incidents, policies, scenarios, seeds  # noqa: E402
from pis.phase_b.episode import run_episode  # noqa: E402
from pis.phase_b.scenarios import Degradation  # noqa: E402
from tests._load import load_script  # noqa: E402
from worldmodel.features import POLICY_NAMES, STATE_NAMES, encode_policy, encode_state  # noqa: E402

gen = load_script("b_generate_history")


def _short(seed, duration=10.0, degradation=None):
    sc = scenarios.scenario_for(seed)
    if degradation is not None:
        sc = dataclasses.replace(sc, degradation=degradation)
    return dataclasses.replace(sc, duration_s=duration)


def _episode(seed, source, duration=10.0, degradation=None):
    sc = _short(seed, duration, degradation)
    ctl = policies.make(source, sc.spec, seed)
    return sc, run_episode(sc, ctl, source)


def _shard(root, name, items):
    d = root / name
    d.mkdir(parents=True)
    for sc, recs, src in items:
        history.write_episode(d, sc, recs, src)
    (d / shards.DONE).write_text("")
    return d


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    deg = Degradation("cpu_loss", 4.0, 0.0, 2.0)
    a = _episode(3100, "r6", degradation=deg)
    b = _episode(3101, "random")
    _shard(root, "s3100", [(a[0], a[1], "r6")])
    _shard(root, "s3101", [(b[0], b[1], "random")])
    # an unfinished shard is never read
    bad = root / "s3102"
    bad.mkdir()
    c = _episode(3102, "r6", duration=2.0)
    history.write_episode(bad, c[0], c[1], "r6")
    return root, {3100: a, 3101: b}


# ------------------------------------------------------------------ sources
def test_source_shares_over_history():
    got = Counter(policies.source_for(s) for s in seeds.HISTORY)
    assert set(got) == set(policies.SOURCES)
    for name, share in policies.SOURCES.items():
        expected = share * len(seeds.HISTORY)
        assert abs(got[name] - expected) <= 0.3 * expected, (name, got[name], expected)


def test_source_deterministic():
    assert [policies.source_for(s) for s in range(3100, 3140)] == \
           [policies.source_for(s) for s in range(3100, 3140)]


def test_calibration_and_selection_are_r6():
    for s in list(seeds.CALIBRATION) + list(seeds.SELECTION):
        assert policies.source_for(s) == "r6"


@pytest.mark.parametrize("seed", [3000, 3099, 1100, 2000, 3500, 999])
def test_source_refuses_other_seeds(seed):
    with pytest.raises(ValueError):
        policies.source_for(seed)


def test_make_refuses_unknown_source():
    with pytest.raises(ValueError):
        policies.make("nope", scenarios.scenario_for(3100).spec, 3100)


@pytest.mark.parametrize("source", sorted(policies.SOURCES))
def test_every_source_runs(source):
    sc, recs = _episode(3103, source, duration=3.0)
    assert len(recs) == 6
    assert all(r["source"] == source for r in recs)
    assert all(is_valid(r["policy"]) for r in recs)


def test_random_changes_every_ten_windows():
    sc = _short(3104, duration=15.0)
    ctl = policies.make("random", sc.spec, 3104)
    recs = run_episode(sc, ctl, "random")
    keys = [policy_key(r["policy"]) for r in recs]
    changes = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    # record i chooses the policy of window i + 1: windows 10, 20, ... start a new policy
    assert changes and all((i + 1) % policies.RANDOM_EVERY == 0 for i in changes)


def test_interventions_held_and_never_overlap():
    sc = _short(3105, duration=20.0)
    ctl = policies.make("r6_interventions", sc.spec, 3105)
    ctl.start_p = 0.5                      # many starts, to exercise the hold logic
    recs = run_episode(sc, ctl, "r6_interventions")
    marks = [r["info"].get("intervention") for r in recs]
    starts = [i for i, r in enumerate(recs) if r["info"].get("intervention_start")]
    assert len(starts) >= 2
    for a, b in zip(starts, starts[1:]):
        assert b - a >= policies.INTERVENTION_WINDOWS
    for i in starts:
        held = [policy_key(r["policy"]) for r in recs[i:i + policies.INTERVENTION_WINDOWS]]
        assert len(set(held)) == 1
        assert all(m is not None for m in marks[i:i + policies.INTERVENTION_WINDOWS])
        assert marks[i] in ("take_from_B", "gpu_to_B", "isolate_A")
        assert is_valid(recs[i]["policy"])


def test_interventions_rate_about_one_per_minute():
    """Through `decide`, with a stub in place of R6 (the schedule is the one decide uses)."""
    ctl = policies.make("r6_interventions", scenarios.scenario_for(3106).spec, 3106)
    cur = adapters.initial_policy()
    ctl.r6 = type("Stub", (), {"decide": lambda self, tel, current, history: (current, {})})()
    infos = [ctl.decide({}, cur, [])[1] for _ in range(10 * 120)]      # ten minutes of windows
    starts = [i for i, f in enumerate(infos) if f.get("intervention_start")]
    assert 5 <= len(starts) <= 16
    assert all(b - a >= policies.INTERVENTION_WINDOWS for a, b in zip(starts, starts[1:]))
    held = sum(f.get("intervention") is not None for f in infos)
    assert held == sum(min(policies.INTERVENTION_WINDOWS, len(infos) - i) for i in starts)


def test_interventions_deterministic():
    spec = scenarios.scenario_for(3107).spec
    a, b = (policies.make("r6_interventions", spec, 3107) for _ in range(2))
    assert [a._next_window() for _ in range(600)] == [b._next_window() for _ in range(600)]


# ------------------------------------------------------------------ store
def test_columns_are_sql_friendly():
    for c in history.WINDOW_COLUMNS:
        assert "." not in c
    assert set(history.STATE_COLUMNS) == set(STATE_NAMES)
    assert history.STATE_COLUMNS["a.p99_ms"] == "a_p99_ms"
    assert history.POLICY_COLUMNS["share_A"] == "policy_share_A"


def test_training_windows_round_trip(store_root):
    root, eps = store_root
    st = history.Store(root)
    df = st.training_windows([3100, 3101])
    assert isinstance(df, pd.DataFrame)
    for seed, (sc, recs) in eps.items():
        part = df[df.episode == seed].sort_values("window")
        assert list(part.window) == list(range(len(recs)))
        want = np.stack([encode_state(r["tel"]) for r in recs])
        got = part[[history.STATE_COLUMNS[n] for n in STATE_NAMES]].to_numpy()
        np.testing.assert_allclose(got, want, rtol=1e-6)
        pol = np.stack([encode_policy(r["applied"]) for r in recs])
        np.testing.assert_allclose(part[[history.POLICY_COLUMNS[n] for n in POLICY_NAMES]].to_numpy(), pol)
        assert list(part.a_queue_overdue) == [r["tel"]["a"]["queue_overdue"] for r in recs]
        assert list(part.slo_v2_violation) == list(incidents.violations(recs))
        assert list(part.deg_active) == [r["tel"]["phase_b"]["active"] for r in recs]
        assert list(part.alive_cpus) == [len(r["tel"]["phase_b"]["alive_cpus"]) for r in recs]
        v = [sc.spec.b.value_per_batch * r["tel"]["b"]["batches"]
             + sc.spec.c.value_per_core_ms * r["tel"]["c"]["work_core_ms"] for r in recs]
        np.testing.assert_allclose(part.value, v)
        assert set(part.source) == {recs[0]["source"]}
    # degradation columns: cpu loss from 4 s on
    p = df[df.episode == 3100].sort_values("window")
    assert not p.deg_active.iloc[0] and p.deg_active.iloc[-1]
    assert p.alive_cpus.iloc[-1] == 6 and set(p.deg_kind) == {"cpu_loss"}


def test_labels_from_store_match_records(store_root):
    root, eps = store_root
    st = history.Store(root)
    rule = incidents.IncidentRule(k=2, interval=6)
    for seed, (sc, recs) in eps.items():
        rows = st.window_rows(seed, 0, len(recs))
        a = incidents.label_series(rows.slo_v2_violation.to_numpy(), rows.deg_active.to_numpy(), rule)
        b = incidents.label(recs, rule)
        np.testing.assert_array_equal(a.in_incident, b.in_incident)
        assert a.incidents == b.incidents


def test_unfinished_shard_ignored(store_root):
    root, _ = store_root
    st = history.Store(root)
    assert sorted(st.episodes().seed) == [3100, 3101]


def test_training_refuses_non_history(store_root):
    root, _ = store_root
    st = history.Store(root)
    for bad in ([3400], [3460], [3000]):
        with pytest.raises(ValueError):
            st.training_windows(bad)


def _synthetic_shard(root, seeds, n=12):
    """Hand-made window and episode files: state = 100 * episode + window, policy_share_A = window / 100."""
    d = root / f"s{seeds[0]}"
    d.mkdir(parents=True)
    for seed in seeds:
        rows = {"episode": [seed] * n, "window": list(range(n)), "t": [0.5 * (w + 1) for w in range(n)]}
        for c in history.STATE_COLUMNS.values():
            rows[c] = [float(100 * seed + w) for w in range(n)]
        for c in history.RAW_COLUMNS.values():
            rows[c] = [0] * n
        for c in history.POLICY_COLUMNS.values():
            rows[c] = [w / 100.0 if c == "policy_share_A" else 0.0 for w in range(n)]
        rows.update(value=[1.0] * n, slo_v2_violation=[False] * n, deg_kind=["none"] * n, deg_active=[False] * n,
                    deg_factor=[0.0] * n, alive_cpus=[8] * n, source=["r6"] * n)
        pd.DataFrame(rows, columns=history.WINDOW_COLUMNS).to_parquet(d / f"windows-{seed}.parquet")
        ep = {"seed": seed, "kind": "none", "source": "r6", "windows": n,
              **{c: float(i + 1) for i, c in enumerate(history.KNOWN_COLUMNS.values())}}
        pd.DataFrame([ep]).to_parquet(d / f"episodes-{seed}.parquet")
    (d / shards.DONE).write_text("")
    return d


def test_training_refuses_missing_seeds(tmp_path):
    _synthetic_shard(tmp_path, [3100, 3101])
    st = history.Store(tmp_path)
    with pytest.raises(ValueError, match="3102"):
        st.training_windows([3100, 3102])


def test_store_refuses_held_out(tmp_path):
    _synthetic_shard(tmp_path, [3100])
    _synthetic_shard(tmp_path, [3005])
    with pytest.raises(ValueError, match="held-out"):
        history.Store(tmp_path)


def _repo(path):
    import subprocess
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=test"]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "a.txt").write_text("1")
    subprocess.run(git + ["add", "a.txt"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)


def test_provenance_ignores_untracked(tmp_path):
    _repo(tmp_path)
    (tmp_path / "new_folder").mkdir()
    (tmp_path / "new_folder" / "b.txt").write_text("x")
    head = history._git_head(tmp_path)
    assert len(head) == 40 and not head.endswith("+dirty")
    (tmp_path / "a.txt").write_text("2")
    assert history._git_head(tmp_path).endswith("+dirty")


def test_inference_query_no_future(store_root):
    root, eps = store_root
    st = history.Store(root)
    n = len(eps[3100][1])
    for upto in (0, 3, 10, n - 1):
        df = st.recent_windows(3100, upto, 8)
        assert df.window.max() == upto
        assert (df.window <= upto).all()
        assert list(df.window) == list(range(max(0, upto - 7), upto + 1))
    assert len(st.recent_windows(3100, 5, 100)) == 6


def test_analysis_queries(store_root):
    root, eps = store_root
    st = history.Store(root)
    ep = st.episodes()
    assert list(ep.seed) == [3100, 3101]
    row = ep[ep.seed == 3100].iloc[0]
    assert row.kind == "cpu_loss" and row.source == "r6" and row.windows == len(eps[3100][1])
    assert row.generator_version == history.GENERATOR_VERSION
    assert '"a"' in row.spec and row.commits
    assert list(st.episodes("source = 'random'").seed) == [3101]
    w = st.window_rows(3101, 2, 5)
    assert list(w.window) == [2, 3, 4]
    m = st.state_matrix([3100, 3101], ["a_p99_ms", "b_rate"])
    assert m.shape == (sum(len(r) for _, r in eps.values()), 2)
    with pytest.raises(ValueError):
        st.state_matrix([3100], ["a_p99_ms; drop table x"])


def test_store_is_read_only(store_root):
    root, _ = store_root
    st = history.Store(root)
    public = [n for n in dir(st) if not n.startswith("_")]
    assert not [n for n in public if any(w in n for w in ("write", "insert", "add", "delete", "put"))]


def test_overloaded_episode_completes(tmp_path):
    deg = Degradation("load_ramp", 2.0, 4.0, 3.0)    # A's rate x4 on one of the busiest history seeds
    sc, recs = _episode(3118, "r6", duration=30.0, degradation=deg)
    assert len(recs) == 60
    assert recs[-1]["tel"]["a"]["queue_overdue"] > 10_000     # really overloaded: the queue keeps growing
    _shard(tmp_path, "s0", [(sc, recs, "r6")])
    st = history.Store(tmp_path)
    assert len(st.training_windows([3118])) == 60


# ------------------------------------------------------------------ script
def test_script_refuses_bad_arguments(tmp_path):
    out = str(tmp_path / "h")
    for argv in (["--seeds", "3000-3001", "--label", "smoke", "--out", out],
                 ["--seeds", "3100", "--duration", "5", "--out", out],
                 ["--seeds", "3100", "--label", "other", "--out", out],
                 ["--seeds", "1100", "--label", "smoke", "--out", out]):
        with pytest.raises(SystemExit):
            gen.main(argv)


def test_script_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(gen.runs, "tree_dirty", lambda ignore=(): True)
    with pytest.raises(SystemExit):
        gen.main(["--seeds", "3100", "--out", str(tmp_path / "h")])


def test_script_rerun_skips_done(tmp_path, capsys):
    argv = ["--seeds", "3100-3101", "--label", "smoke", "--duration", "2", "--procs", "1",
            "--out", str(tmp_path / "h")]
    first = gen.main(argv)
    assert first["failed"] == [] and len(first["done"]) == 1
    second = gen.main(argv)
    assert second["done"] == [] and second["skipped"] == first["done"]
    st = history.Store(tmp_path / "h")
    assert sorted(st.episodes().seed) == [3100, 3101]
    assert (st.episodes().wall_s > 0).all()


def test_training_sequences_removed():
    assert not hasattr(history.Store, "training_sequences")
    assert not hasattr(history, "INPUT_COLUMNS") and not hasattr(history, "TARGET_COLUMNS")


def test_generator_uses_the_shared_guards():
    assert not hasattr(gen, "tree_dirty") and not hasattr(gen, "LABELS")
    for bad in ([3000], [1100], [2050]):
        with pytest.raises(ValueError):
            gen.check_seeds(bad)
    gen.check_seeds([3100, 3420, 3460])
