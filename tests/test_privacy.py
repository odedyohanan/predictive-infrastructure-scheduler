import os
import subprocess
import sys
from pathlib import Path

from pis.privacy import scrub

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_privacy.py"
HOME_PREFIX = "/" + "Users" + "/"


def _run(repo: Path) -> int:
    return subprocess.run([sys.executable, str(SCRIPT), str(repo)], capture_output=True).returncode


def _repo(tmp_path: Path, content: str) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text(content)
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    return tmp_path


def test_scrub_removes_home_and_cwd():
    text = f"a {Path.home()}/x and {os.getcwd()}/y"
    out = scrub(text)
    assert str(Path.home()) not in out
    assert os.getcwd() not in out


def test_scrub_removes_email():
    addr = "someone" + "@" + "example.org"
    assert addr not in scrub(f"mail {addr} now")
    assert "<email>" in scrub(f"mail {addr} now")


def test_scrub_removes_home_style_paths():
    out = scrub("see " + HOME_PREFIX + "bob/proj/file.py ok")
    assert "bob" not in out and "<path>" in out


def test_check_privacy_flags_abs_path(tmp_path):
    repo = _repo(tmp_path, "path = " + HOME_PREFIX + "bob/code\n")
    assert _run(repo) == 1
    (repo / "f.txt").write_text("path = ../relative\n")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    assert _run(repo) == 0


def test_denylist_local_is_used(tmp_path):
    repo = _repo(tmp_path, "contains secretword here\n")
    assert _run(repo) == 0
    (repo / ".privacy-denylist.local").write_text("secretword\n")
    assert _run(repo) == 1
