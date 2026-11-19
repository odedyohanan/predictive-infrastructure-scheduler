import json
import os

import pytest

from pis import shards


def write_items(shard, out):
    """Shard function used by the tests (top level, so spawned workers can import it)."""
    (out / "items.json").write_text(json.dumps(shard.items))


def fail_odd(shard, out):
    (out / "items.json").write_text(json.dumps(shard.items))
    if shard.items[0] % 2:
        raise RuntimeError("odd shard")


CONFIG = {"kind": "test", "seeds": [1, 2, 3]}


def test_make_shards():
    sh = shards.make_shards(list(range(7)), 3)
    assert [s.items for s in sh] == [[0, 1, 2], [3, 4, 5], [6]]
    assert len({s.id for s in sh}) == 3
    with pytest.raises(ValueError):
        shards.make_shards([1], 0)


def test_all_shards_run_and_collect(tmp_path):
    sh = shards.make_shards(list(range(6)), 2)
    res = shards.run_shards(tmp_path, sh, write_items, CONFIG, procs=2)
    assert sorted(res["done"]) == sorted(s.id for s in sh)
    assert res["skipped"] == [] and res["failed"] == []
    got = shards.collect(tmp_path)
    assert [p.name for p in got] == sorted(s.id for s in sh)
    assert all((p / "done").exists() for p in got)
    assert json.loads((tmp_path / "config.json").read_text()) == CONFIG


def test_failed_shard_has_no_done_and_rerun_runs_only_missing(tmp_path):
    sh = shards.make_shards(list(range(8)), 2)          # items 0,2,4,6 start the shards: all even
    first = shards.run_shards(tmp_path, sh[:2], write_items, CONFIG, procs=1)
    assert len(first["done"]) == 2
    res = shards.run_shards(tmp_path, sh, fail_odd, CONFIG, procs=2)
    assert sorted(res["skipped"]) == sorted(s.id for s in sh[:2])
    assert sorted(res["done"]) == sorted(s.id for s in sh[2:])


def test_failing_fn_leaves_shard_without_done(tmp_path):
    sh = shards.make_shards([1, 2, 3, 4], 1)             # items 1 and 3 fail
    res = shards.run_shards(tmp_path, sh, fail_odd, CONFIG, procs=2)
    failed = {s.id for s in sh if s.items[0] % 2}
    assert set(res["failed"]) == failed
    for s in sh:
        assert (tmp_path / s.id / "done").exists() == (s.id not in failed)
    assert {p.name for p in shards.collect(tmp_path)} == {s.id for s in sh} - failed
    # rerun with a fixed function: only the failed shards run again
    res2 = shards.run_shards(tmp_path, sh, write_items, CONFIG, procs=1)
    assert set(res2["done"]) == failed
    assert set(res2["skipped"]) == {s.id for s in sh} - failed
    assert len(shards.collect(tmp_path)) == 4


def test_rerun_clears_partial_output(tmp_path):
    sh = shards.make_shards([1], 1)
    shards.run_shards(tmp_path, sh, fail_odd, CONFIG, procs=1)
    (tmp_path / sh[0].id / "stale.txt").write_text("x")
    shards.run_shards(tmp_path, sh, write_items, CONFIG, procs=1)
    assert not (tmp_path / sh[0].id / "stale.txt").exists()
    assert (tmp_path / sh[0].id / "done").exists()


def test_config_change_refused(tmp_path):
    sh = shards.make_shards([0, 2], 1)
    shards.run_shards(tmp_path, sh[:1], write_items, CONFIG, procs=1)
    changed = dict(CONFIG, seeds=[1, 2, 4])
    with pytest.raises(ValueError):
        shards.run_shards(tmp_path, sh, write_items, changed, procs=1)
    assert not (tmp_path / sh[1].id).exists()           # nothing ran
    res = shards.run_shards(tmp_path, sh, write_items, CONFIG, procs=1)
    assert res["skipped"] == [sh[0].id] and res["done"] == [sh[1].id]


def test_duplicate_shard_ids_refused(tmp_path):
    with pytest.raises(ValueError):
        shards.run_shards(tmp_path, [shards.Shard("x", [1]), shards.Shard("x", [2])],
                          write_items, CONFIG, procs=1)


def test_collect_on_missing_dir(tmp_path):
    assert shards.collect(tmp_path / "nothing") == []


def die_on_99(shard, out):
    """Kills its worker process outright (as an OOM kill or a signal would)."""
    if 99 in shard.items:
        os._exit(1)
    (out / "items.json").write_text(json.dumps(shard.items))


def test_dead_worker_fails_only_its_shard(tmp_path):
    sh = shards.make_shards([0, 2, 99, 4, 6], 1)
    res = shards.run_shards(tmp_path, sh, die_on_99, CONFIG, procs=2)
    dead = sh[2].id
    assert res["failed"] == [dead]
    assert sorted(res["done"]) == sorted(s.id for s in sh if s.id != dead)
    assert not (tmp_path / dead / "done").exists()
    res2 = shards.run_shards(tmp_path, sh, write_items, CONFIG, procs=2)
    assert res2["done"] == [dead] and len(res2["skipped"]) == 4


@pytest.mark.parametrize("bad", ["", "a/b", "..", "a..b", "x y", "/abs"])
def test_bad_shard_ids_refused(bad):
    with pytest.raises(ValueError):
        shards.Shard(bad, [1])
