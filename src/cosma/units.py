"""Length units. Everything inside COSMA is metres; conversion happens once, at import.

Three feet exist and they are not the same: the international foot (0.3048 m
exactly), the US survey foot (1200/3937 m, about 0.3048006 m) and, for old data,
others. At a State Plane easting of 6 million feet the survey/international
difference is about 12 ft, so `ft` is refused and the caller must say which one.
"""

from __future__ import annotations

import numpy as np

_FACTORS = {
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft_intl": 0.3048,
    "ft_us": 1200.0 / 3937.0,
}
_ALIASES = {
    "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "foot_intl": "ft_intl", "ift": "ft_intl", "ft_international": "ft_intl",
    "foot_us": "ft_us", "usft": "ft_us", "ftus": "ft_us", "us_survey_foot": "ft_us",
}


class UnitError(ValueError):
    pass


def unit_factor(unit: str) -> float:
    """Metres per `unit`."""
    if not isinstance(unit, str):
        raise UnitError(f"unit must be a string, got {unit!r}")
    key = unit.strip().lower()
    key = _ALIASES.get(key, key)
    if key in ("ft", "feet", "foot"):
        raise UnitError(
            f"unit {unit!r} is ambiguous: use 'ft_intl' (0.3048 m) or 'ft_us' (US survey foot, "
            "1200/3937 m). They differ by 2 ppm, which is ~12 ft at State Plane coordinates."
        )
    if key not in _FACTORS:
        raise UnitError(f"unknown length unit {unit!r}; known: {sorted(_FACTORS)}")
    return _FACTORS[key]


def to_metres(values, unit: str):
    """`values` converted from `unit` to metres (float64 array, NaN preserved)."""
    return np.asarray(values, dtype=float) * unit_factor(unit)
