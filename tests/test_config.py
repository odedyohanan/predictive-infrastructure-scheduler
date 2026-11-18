import sys

import pytest

from pis import config


def test_sibling_paths_prefers_env(monkeypatch):
    monkeypatch.setenv("SCHED_REPO", "elsewhere/sched")
    paths = config.sibling_paths()
    assert str(paths["sched"]) == "elsewhere/sched"
    assert set(paths) == {"sched", "worldmodel", "old_data"}


def test_sibling_defaults_are_relative(monkeypatch):
    for k in ("SCHED_REPO", "WORLDMODEL_REPO", "OLD_DATA"):
        monkeypatch.delenv(k, raising=False)
    assert all(not p.is_absolute() for p in config.sibling_paths().values())


def test_setup_imports_missing_repo_message_has_no_absolute_path(monkeypatch):
    monkeypatch.setenv("SCHED_REPO", "does/not/exist")
    with pytest.raises(FileNotFoundError) as e:
        config.setup_imports()
    assert "does/not/exist" in str(e.value)
    assert "/" + "Users" not in str(e.value)


def test_setup_imports_idempotent():
    config.setup_imports()
    n = len(sys.path)
    config.setup_imports()
    assert len(sys.path) == n


def test_setup_imports_error_does_not_echo_absolute_env_path(monkeypatch):
    absolute = "/" + "Users" + "/someone/missing_repo"
    monkeypatch.setenv("SCHED_REPO", absolute)
    with pytest.raises(FileNotFoundError) as e:
        config.setup_imports()
    assert absolute not in str(e.value)
    assert "SCHED_REPO" in str(e.value)
