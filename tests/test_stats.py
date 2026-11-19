import math

import numpy as np
import pytest

from pis.stats import paired_diff, t_quantile


@pytest.mark.parametrize("p, df, want", [(0.975, 9, 2.2621572), (0.975, 99, 1.9842169),
                                         (0.95, 4, 2.1318468), (0.975, 1, 12.7062047), (0.5, 7, 0.0)])
def test_t_quantile_known_values(p, df, want):
    assert t_quantile(p, df) == pytest.approx(want, abs=1e-5)


def test_paired_diff_known_values():
    r = paired_diff(np.array([1.0, 2, 3, 4, 5]), np.zeros(5))
    half = 2.7764451 * math.sqrt(2.5) / math.sqrt(5)
    assert r["n"] == 5 and r["mean"] == pytest.approx(3.0)
    assert r["lo"] == pytest.approx(3.0 - half, abs=1e-5) and r["hi"] == pytest.approx(3.0 + half, abs=1e-5)
    r90 = paired_diff(np.array([1.0, 2, 3, 4, 5]), np.zeros(5), conf=0.90)
    assert r90["hi"] - r90["lo"] < r["hi"] - r["lo"]


def test_ci_contains_zero_for_identical():
    a = np.random.default_rng(0).normal(size=30)
    r = paired_diff(a, a.copy())
    assert r["lo"] <= 0.0 <= r["hi"] and r["mean"] == 0.0


def test_paired_diff_rejects_mismatch_and_handles_n1():
    with pytest.raises(ValueError):
        paired_diff(np.zeros(3), np.zeros(4))
    r = paired_diff(np.array([2.0]), np.array([1.0]))
    assert r["n"] == 1 and r["mean"] == 1.0 and math.isnan(r["lo"]) and math.isnan(r["hi"])
