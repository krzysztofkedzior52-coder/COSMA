"""Placement of a local grid frame in map (e.g. UTM) coordinates.

The grid itself lives in local coordinates starting near (0, 0): float32-safe,
and what every simulator wants. The frame says where that local system sits
on the map: origin of the local (0, 0) in map coordinates, and the azimuth of
the local +y axis (the J direction for a north-up grid), measured clockwise
from map north. The local +x axis (I direction) is at azimuth + 90 degrees.

    X = origin_x + x*cos(az) + y*sin(az)
    Y = origin_y - x*sin(az) + y*cos(az)

At azimuth 0, +x is east and +y is north (I east, J north, K down: a
left-handed grid). Rotations preserve handedness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridFrame:
    origin_x: float = 0.0
    origin_y: float = 0.0
    azimuth_deg: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.origin_x == 0.0 and self.origin_y == 0.0 and self.azimuth_deg % 360.0 == 0.0

    def _cs(self) -> tuple[float, float]:
        a = math.radians(self.azimuth_deg)
        return math.cos(a), math.sin(a)

    def to_map(self, x, y):
        c, s = self._cs()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.origin_x + x * c + y * s, self.origin_y - x * s + y * c

    def to_local(self, X, Y):
        c, s = self._cs()
        dX = np.asarray(X, dtype=float) - self.origin_x
        dY = np.asarray(Y, dtype=float) - self.origin_y
        return dX * c - dY * s, dX * s + dY * c

    def mapaxes(self, length: float = 100.0) -> tuple[float, float, float, float, float, float]:
        """The six MAPAXES numbers: a point on the grid y axis, the origin,
        a point on the grid x axis (all in map coordinates)."""
        yx, yy = self.to_map(0.0, length)
        xx, xy = self.to_map(length, 0.0)
        return (float(yx), float(yy), self.origin_x, self.origin_y, float(xx), float(xy))
