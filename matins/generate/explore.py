"""Adaptive exploration: scale the explore temperature to recent uncertainty.

algo-update.md #4 -- the explore knobs should not be constant. When the system's
recent agreement with the user is volatile (uncertain taste, or active drift) it
should explore MORE; when agreement is stable it can afford to exploit. v1 uses the
single cleanest proxy already logged per batch -- the volatility of recent self-vs-
user Kendall tau -- and maps it to a temperature. Heuristic, not a posterior. (A real
Thompson-style posterior is deferred.)
"""
from __future__ import annotations

# Heuristic knobs for the volatility->temperature map. Module constants, not
# parameters: nothing varies them today, so they stay off the call surface (no
# speculative configurability) while still naming the heuristic's levers.
_GAIN = 0.5
_NEUTRAL_VOLATILITY = 0.3
_LO = 0.1
_HI = 0.9


def adaptive_temperature(recent_taus, base: float) -> float:
    """Adjust `base` up/down by how volatile recent self-vs-user tau has been.

    `recent_taus` is an iterable of past batches' tau (None entries ignored). With
    fewer than 2 usable taus (cold start) there is no volatility signal, so `base`
    is returned unchanged. Otherwise temperature rises with volatility above a
    neutral level (explore) and falls below it (exploit), clamped to [_LO, _HI].
    """
    taus = [t for t in (recent_taus or []) if t is not None]
    if len(taus) < 2:
        return base
    mean = sum(taus) / len(taus)
    volatility = (sum((t - mean) ** 2 for t in taus) / len(taus)) ** 0.5
    temp = base + _GAIN * (volatility - _NEUTRAL_VOLATILITY)
    return max(_LO, min(_HI, temp))
