"""Well trajectories by the minimum curvature method.

Conventions (internal): x, y in the model CRS (metres); depth `tvdss` in metres,
positive down, relative to mean sea level (or whichever datum the surfaces use);
`kb` is the elevation of the depth reference (kelly bushing / rotary table) above
that datum, so TVD from KB = tvdss + kb. Inclination is measured from vertical
down (0 = vertical), azimuth clockwise from grid north.

Minimum curvature is exact for a circular arc (tested analytically). Between survey
stations the path is the arc; `at_md` interpolates along that arc, not along a chord.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TrajectoryError(ValueError):
    pass


def _unit(incl, azim):
    si = np.sin(incl)
    return np.stack([si * np.sin(azim), si * np.cos(azim), np.cos(incl)], axis=-1)  # east, north, down


def _ratio_factor(beta):
    """RF = tan(b/2)/(b/2), with the series for small b."""
    beta = np.asarray(beta, dtype=float)
    out = np.ones_like(beta)
    big = beta > 1e-6
    h = beta[big] / 2.0
    out[big] = np.tan(h) / h
    out[~big] = 1.0 + beta[~big] ** 2 / 12.0
    return out


def _dogleg(u1, u2):
    """Angle between unit vectors, stable at both ends."""
    cross = np.linalg.norm(np.cross(u1, u2), axis=-1)
    dot = np.sum(u1 * u2, axis=-1)
    return np.arctan2(cross, dot)


@dataclass(frozen=True)
class Trajectory:
    md: np.ndarray      # (n,) metres, strictly increasing
    x: np.ndarray       # (n,)
    y: np.ndarray       # (n,)
    tvdss: np.ndarray   # (n,) depth positive down below datum
    incl: np.ndarray    # (n,) radians
    azim: np.ndarray    # (n,) radians, grid north
    kb: float           # elevation of depth reference above datum, metres

    # -- construction ---------------------------------------------------------------
    @classmethod
    def from_survey(cls, md, incl_deg, azim_deg, *, x0, y0, kb, md0=0.0, tvd0=0.0,
                    azimuth_reference, convergence_deg=None):
        """Build from MD / inclination / azimuth stations.

        (x0, y0) is the wellhead, `md0` the MD of the first station position and `tvd0`
        its TVD below the depth reference (KB). The first station need not be at MD 0;
        positions before it are not extrapolated.

        `azimuth_reference` must be "grid" or "true". For "true", `convergence_deg`
        (grid convergence at the well: grid north minus true north, positive when grid
        north is east of true north) is required and grid azimuth = true - convergence.
        """
        md = np.asarray(md, float)
        inc = np.radians(np.asarray(incl_deg, float))
        az_deg = np.asarray(azim_deg, float)
        if azimuth_reference == "grid":
            if convergence_deg not in (None, 0, 0.0):
                raise TrajectoryError("convergence_deg given but azimuth_reference='grid'")
        elif azimuth_reference == "true":
            if convergence_deg is None:
                raise TrajectoryError("azimuth_reference='true' needs convergence_deg")
            az_deg = az_deg - float(convergence_deg)
        else:
            raise TrajectoryError("azimuth_reference must be 'grid' or 'true'")
        az = np.radians(az_deg)
        if not (md.shape == inc.shape == az.shape and md.ndim == 1 and md.size >= 1):
            raise TrajectoryError("md, incl, azim must be 1-D arrays of equal length")
        if not (np.all(np.isfinite(md)) and np.all(np.isfinite(inc)) and np.all(np.isfinite(az))):
            raise TrajectoryError("survey contains NaN/inf")
        if np.any(np.diff(md) <= 0):
            raise TrajectoryError("survey MD must be strictly increasing")
        if np.any(inc < 0) or np.any(inc > np.pi):
            raise TrajectoryError("inclination must be within [0, 180] degrees")
        u = _unit(inc, az)
        dmd = np.diff(md)
        beta = _dogleg(u[:-1], u[1:])
        rf = _ratio_factor(beta)
        step = (dmd * 0.5 * rf)[:, None] * (u[:-1] + u[1:])
        pos = np.vstack([[0.0, 0.0, 0.0], np.cumsum(step, axis=0)])
        return cls(
            md=md,
            x=float(x0) + pos[:, 0],
            y=float(y0) + pos[:, 1],
            tvdss=float(tvd0) - float(kb) + pos[:, 2],
            incl=inc,
            azim=az,
            kb=float(kb),
        )

    @classmethod
    def vertical(cls, x0, y0, *, kb, td):
        """Explicitly vertical well from KB (MD 0) to `td` metres MD."""
        if td <= 0:
            raise TrajectoryError("td must be positive")
        return cls.from_survey([0.0, td], [0.0, 0.0], [0.0, 0.0], x0=x0, y0=y0, kb=kb,
                               azimuth_reference="grid")

    # -- queries ----------------------------------------------------------------------
    @property
    def md_range(self):
        return float(self.md[0]), float(self.md[-1])

    def at_md(self, md):
        """(x, y, tvdss) at the given MDs, along the circular arc between stations.

        Raises outside the surveyed MD range: no extrapolation.
        """
        m = np.atleast_1d(np.asarray(md, float))
        lo, hi = self.md_range
        tol = 1e-9 * max(1.0, abs(hi))
        if np.any(~np.isfinite(m)) or np.any(m < lo - tol) or np.any(m > hi + tol):
            raise TrajectoryError(f"MD outside surveyed range [{lo}, {hi}]")
        m = np.clip(m, lo, hi)
        k = np.clip(np.searchsorted(self.md, m, side="right") - 1, 0, self.md.size - 2) if self.md.size > 1 else np.zeros(m.shape, int)
        if self.md.size == 1:
            return (np.full(m.shape, self.x[0]), np.full(m.shape, self.y[0]), np.full(m.shape, self.tvdss[0]))
        u1 = _unit(self.incl[k], self.azim[k])
        u2 = _unit(self.incl[k + 1], self.azim[k + 1])
        seg = self.md[k + 1] - self.md[k]
        t = (m - self.md[k]) / seg
        beta = _dogleg(u1, u2)
        small = beta < 1e-9
        sb = np.where(small, 1.0, np.sin(beta))
        w1 = np.where(small, 1.0 - t, np.sin((1.0 - t) * beta) / sb)
        w2 = np.where(small, t, np.sin(t * beta) / sb)
        ut = w1[:, None] * u1 + w2[:, None] * u2
        ut /= np.linalg.norm(ut, axis=1)[:, None]
        b2 = _dogleg(u1, ut)
        d = ((m - self.md[k]) * 0.5 * _ratio_factor(b2))[:, None] * (u1 + ut)
        return self.x[k] + d[:, 0], self.y[k] + d[:, 1], self.tvdss[k] + d[:, 2]

    def tvdss_at_md(self, md):
        return self.at_md(md)[2]

    def md_at_tvdss(self, tvdss):
        """MD at the given depths below datum. Requires depth to increase monotonically
        with MD over the surveyed interval (no horizontal or hooked sections), else raises:
        the answer would be ambiguous."""
        if np.any(np.diff(self.tvdss) <= 1e-6) or np.any(self.incl > np.pi / 2):
            raise TrajectoryError("trajectory has non-increasing TVD (horizontal or hooked section): "
                                  "depth to MD conversion is ambiguous; give MD instead")
        z = np.atleast_1d(np.asarray(tvdss, float))
        if np.any(z < self.tvdss[0] - 1e-9) or np.any(z > self.tvdss[-1] + 1e-9):
            raise TrajectoryError(f"depth outside surveyed TVDSS range [{self.tvdss[0]}, {self.tvdss[-1]}]")
        # station-wise guess, then refine on the arc by bisection
        md = np.interp(z, self.tvdss, self.md)
        for _ in range(3):  # Newton on the arc: dz/dmd = cos(inclination) at the point, via finite difference
            zz = self.tvdss_at_md(md)
            eps = 1e-3
            hi = np.minimum(md + eps, self.md[-1])
            lo = np.maximum(md - eps, self.md[0])
            dz = (self.tvdss_at_md(hi) - self.tvdss_at_md(lo)) / np.maximum(hi - lo, 1e-12)
            md = np.clip(md - (zz - z) / np.where(dz > 1e-9, dz, 1.0), self.md[0], self.md[-1])
        return md
