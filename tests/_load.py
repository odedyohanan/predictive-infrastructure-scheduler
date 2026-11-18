"""Import one of our experiment scripts by file path (a sibling repo also has an `experiments` package)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"pis_exp_{name}", ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
