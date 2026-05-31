"""Offline tests for the adaptive explore controller (matins.generate.explore)."""
from __future__ import annotations

from matins.generate.explore import adaptive_temperature


def test_cold_start_returns_base_unchanged() -> None:
    # Fewer than 2 usable taus -> no volatility signal -> base unchanged.
    assert adaptive_temperature([], 0.4) == 0.4
    assert adaptive_temperature([0.7], 0.4) == 0.4
    assert adaptive_temperature([None, None], 0.4) == 0.4


def test_volatile_explores_more_than_stable() -> None:
    base = 0.4
    stable = adaptive_temperature([0.8, 0.8, 0.8, 0.8], base)        # zero volatility
    volatile = adaptive_temperature([-1.0, 1.0, -1.0, 1.0], base)    # max volatility
    assert volatile > base                                          # uncertain/drifting -> explore
    assert stable < base                                            # stable -> exploit


def test_clamped_to_unit_band() -> None:
    # Extreme volatility with a high base stays within [0.1, 0.9].
    hot = adaptive_temperature([-1.0, 1.0, -1.0, 1.0], 0.9)
    assert 0.1 <= hot <= 0.9
    # A degenerate low base with zero volatility cannot go below the floor.
    cold = adaptive_temperature([0.5, 0.5, 0.5], 0.05)
    assert cold >= 0.1


def test_none_taus_are_ignored() -> None:
    # Mixing in None (un-ranked batches) must not change the volatility estimate.
    a = adaptive_temperature([0.2, 0.9], 0.4)
    b = adaptive_temperature([0.2, None, 0.9, None], 0.4)
    assert a == b
