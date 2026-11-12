"""Variogram models, experimental variograms (wrapping gstools), and fitting.

Model conventions (the GSLIB ones, so numbers carry over to/from other tools):
  * ranges are PRACTICAL ranges: gamma reaches (about) its sill at the range;
      spherical   g(r) = 1.5 r - 0.5 r^3  (r < 1), 1 beyond
      exponential g(r) = 1 - exp(-3 r)
      gaussian    g(r) = 1 - exp(-3 r^2)
    with r = sqrt((u/a_major)^2 + (v/a_minor)^2 + (w/a_vertical)^2);
  * azimuth is the direction of the MAJOR horizontal axis, degrees clockwise from north
    (the same convention as well azimuths); u runs along it, v is horizontal and
    perpendicular to it, w is vertical (depth). Layer-parallel anisotropy only: no dip/plunge;
  * gamma(h) = nugget + sum of structure contributions, gamma(0) = 0; the sill is
    nugget + sum of contributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm, rankdata

TYPES = ("spherical", "exponential", "gaussian")
TRANSFORMS = ("none", "normal_score", "indicator")
KINDS = ("major", "minor", "vertical")


class VariogramError(ValueError):
    pass


def shape(kind: str, r):
    r = np.asarray(r, float)
    if kind == "spherical":
        return np.where(r < 1.0, 1.5 * r - 0.5 * r ** 3, 1.0)
    if kind == "exponential":
        return 1.0 - np.exp(-3.0 * r)
    if kind == "gaussian":
        return 1.0 - np.exp(-3.0 * r ** 2)
    raise VariogramError(f"unknown variogram type {kind!r}; known: {TYPES}")


@dataclass(frozen=True)
class Structure:
    type: str
    contribution: float
    range_major: float
    range_minor: float
    range_vert: float

    def check(self):
        if self.type not in TYPES:
            raise VariogramError(f"unknown variogram type {self.type!r}; known: {TYPES}")
        if not (np.isfinite(self.contribution) and self.contribution > 0):
            raise VariogramError("a structure's contribution must be positive")
        for nm in ("range_major", "range_minor", "range_vert"):
            v = getattr(self, nm)
            if not (np.isfinite(v) and v > 0):
                raise VariogramError(f"{nm} must be positive")
        if self.range_minor > self.range_major * (1 + 1e-9):
            raise VariogramError("the minor range cannot exceed the major range (swap the axes and turn the azimuth by 90 degrees)")


@dataclass(frozen=True)
class VariogramModel:
    nugget: float
    structures: tuple
    azimuth: float = 0.0                  # degrees clockwise from north, of the major axis
    transform: str = "none"               # data the variogram was computed on: "none" (raw), "normal_score", or "indicator" (0/1 facies)
    variable: str = ""

    def check(self):
        if not (np.isfinite(self.nugget) and self.nugget >= 0):
            raise VariogramError("nugget cannot be negative")
        if not self.structures:
            raise VariogramError("a model needs at least one structure")
        if len(self.structures) > 3:
            raise VariogramError("at most three structures are supported")
        for s in self.structures:
            s.check()
        if not np.isfinite(self.azimuth):
            raise VariogramError("azimuth must be a number")
        if self.transform not in TRANSFORMS:
            raise VariogramError(f"transform must be one of {TRANSFORMS}")
        return self

    @property
    def sill(self) -> float:
        return self.nugget + sum(s.contribution for s in self.structures)

    def gamma_xyz(self, dx, dy, dz):
        """Semivariance for separation vectors (east, north, down)."""
        dx, dy, dz = (np.asarray(a, float) for a in (dx, dy, dz))
        az = np.radians(self.azimuth)
        u = dx * np.sin(az) + dy * np.cos(az)
        v = dx * np.cos(az) - dy * np.sin(az)
        h = np.sqrt(dx * dx + dy * dy + dz * dz)
        g = np.zeros_like(h)
        for s in self.structures:
            r = np.sqrt((u / s.range_major) ** 2 + (v / s.range_minor) ** 2 + (dz / s.range_vert) ** 2)
            g = g + s.contribution * shape(s.type, r)
        return np.where(h > 0, g + self.nugget, 0.0)

    def gamma_along(self, h, kind: str):
        """Semivariance along the major axis, the minor axis or the vertical."""
        h = np.asarray(h, float)
        z = np.zeros_like(h)
        az = np.radians(self.azimuth)
        if kind == "major":
            return self.gamma_xyz(h * np.sin(az), h * np.cos(az), z)
        if kind == "minor":
            return self.gamma_xyz(h * np.cos(az), -h * np.sin(az), z)
        if kind == "vertical":
            return self.gamma_xyz(z, z, h)
        raise VariogramError(f"kind must be one of {KINDS}")

    def range_of(self, kind: str) -> float:
        """Largest practical range among the structures in the given direction."""
        return max({"major": s.range_major, "minor": s.range_minor, "vertical": s.range_vert}[kind] for s in self.structures)


# -- GSLIB parameter-block text ------------------------------------------------------------------------------------------
_GSLIB_TYPES = {1: "spherical", 2: "exponential", 3: "gaussian"}


def to_gslib(m: VariogramModel) -> str:
    inv = {v: k for k, v in _GSLIB_TYPES.items()}
    out = [f"{len(m.structures)} {m.nugget:.6g}      -nst, nugget constant"]
    for s in m.structures:
        out.append(f"{inv[s.type]} {s.contribution:.6g} {m.azimuth:.6g} 0 0   -it, cc, ang1, ang2, ang3")
        out.append(f"{s.range_major:.6g} {s.range_minor:.6g} {s.range_vert:.6g}   -a_hmax, a_hmin, a_vert")
    return "\n".join(out) + "\n"


def from_gslib(text: str, *, variable: str = "", transform: str = "none") -> VariogramModel:
    """Parse the GSLIB variogram block used in sgsim/sisim parameter files. Anything the model
    here cannot represent is refused rather than approximated: power models, dip/plunge angles,
    and structures with different azimuths."""
    rows = []
    for ln, line in enumerate(text.splitlines(), start=1):
        body = line.split("-", 1)[0].split("#", 1)[0].strip() if not line.strip().startswith("-") else ""
        if not body:
            continue
        toks = body.replace(",", " ").split()
        try:
            rows.append([float(t) for t in toks])
        except ValueError:
            raise VariogramError(f"line {ln}: {line.strip()!r} is not a GSLIB variogram line") from None
    if not rows or len(rows[0]) < 2:
        raise VariogramError("first line must be 'nst nugget'")
    nst = int(rows[0][0])
    nug = rows[0][1]
    if nst < 1 or len(rows) != 1 + 2 * nst:
        raise VariogramError(f"expected {1 + 2 * nst} lines for nst={nst}, found {len(rows)}")
    structs, azs = [], set()
    for k in range(nst):
        a, b = rows[1 + 2 * k], rows[2 + 2 * k]
        if len(a) < 5 or len(b) < 3:
            raise VariogramError(f"structure {k + 1}: needs 'it cc ang1 ang2 ang3' and 'a_hmax a_hmin a_vert'")
        it = int(a[0])
        if it not in _GSLIB_TYPES:
            raise VariogramError(f"structure {k + 1}: model type {it} is not supported (1 spherical, 2 exponential, 3 gaussian)")
        if abs(a[3]) > 1e-9 or abs(a[4]) > 1e-9:
            raise VariogramError(f"structure {k + 1}: dip/plunge angles ({a[3]:g}, {a[4]:g}) are not supported (layer-parallel anisotropy only)")
        azs.add(round(a[2] % 360.0, 9))
        structs.append(Structure(_GSLIB_TYPES[it], a[1], b[0], b[1], b[2]))
    if len(azs) > 1:
        raise VariogramError(f"structures have different azimuths {sorted(azs)}: not supported")
    return VariogramModel(nug, tuple(structs), azs.pop(), transform, variable).check()


# -- experimental variograms --------------------------------------------------------------------------------------------------
@dataclass
class Experimental:
    kind: str                       # "major" | "minor" | "vertical"
    lag: np.ndarray                 # bin centres, metres
    gamma: np.ndarray               # NaN where there are no pairs
    npairs: np.ndarray
    variance: float                 # variance of the data used (the usual reference sill)
    settings: dict = field(default_factory=dict)

    def usable(self, min_pairs: int = 1):
        ok = np.isfinite(self.gamma) & (self.npairs >= min_pairs)
        return self.lag[ok], self.gamma[ok], self.npairs[ok]


def normal_score(values) -> np.ndarray:
    """Rank-based normal-score transform (ties share their average rank)."""
    v = np.asarray(values, float)
    r = rankdata(v, method="average")
    return norm.ppf((r - 0.5) / v.size)


def _estimate(pos, vals, edges, direction, tol, bandwidth, sampling_size, seed):
    import gstools as gs

    kw = dict(direction=[direction], angles_tol=tol, return_counts=True, estimator="matheron")
    if bandwidth is not None:
        kw["bandwidth"] = bandwidth
    if sampling_size is not None and sampling_size < len(vals):
        kw.update(sampling_size=sampling_size, sampling_seed=seed)
    _, g, c = gs.vario_estimate(pos, vals, edges, **kw)
    return np.asarray(g, float)[0] if np.ndim(g) > 1 else np.asarray(g, float), np.asarray(c)[0] if np.ndim(c) > 1 else np.asarray(c)


def _horizontal(p, v, groups, edges, azimuth, tol_deg, bandwidth, vtol, max_points, seed):
    """Matheron semivariance over PAIRS FROM DIFFERENT WELLS that lie in the same layer
    (|dz| <= vtol) and whose horizontal separation points along `azimuth` (either way)
    within `tol_deg`; optionally with perpendicular offset <= bandwidth."""
    n = v.size
    if max_points is not None and n > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
        p, v = p[keep], v[keep]
        groups = None if groups is None else groups[keep]
    nb = len(edges) - 1
    num, cnt = np.zeros(nb), np.zeros(nb, dtype=np.int64)
    u = np.array([np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))])
    cos_tol = np.cos(np.radians(tol_deg))
    gid = np.arange(v.size) if groups is None else np.unique(groups, return_inverse=True)[1]
    step = max(1, int(2e7 // max(v.size, 1)))
    for a in range(0, v.size, step):
        b = min(a + step, v.size)
        dx = p[None, :, 0] - p[a:b, None, 0]
        dy = p[None, :, 1] - p[a:b, None, 1]
        dz = np.abs(p[None, :, 2] - p[a:b, None, 2])
        h = np.hypot(dx, dy)
        along = np.abs(dx * u[0] + dy * u[1])
        ok = (np.arange(v.size)[None, :] > np.arange(a, b)[:, None]) & (gid[None, :] != gid[a:b, None]) & (dz <= vtol) & (h >= edges[0]) \
            & (h < edges[-1]) & (along >= h * cos_tol)
        if bandwidth is not None:
            ok &= np.sqrt(np.maximum(h * h - along * along, 0.0)) <= bandwidth
        if not ok.any():
            continue
        k = np.searchsorted(edges, h[ok], side="right") - 1
        sq = 0.5 * (v[None, :] - v[a:b, None])[ok] ** 2
        num += np.bincount(k, sq, minlength=nb)[:nb]
        cnt += np.bincount(k, minlength=nb)[:nb]
    return np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan), cnt


def experimental_variogram(points, values, kind: str, *, lag: float, nlags: int, azimuth: float = 0.0,
                           tolerance_deg: float = 22.5, bandwidth: float | None = None, groups=None,
                           vertical_tolerance: float | None = None,
                           max_points: int | None = 5000, seed: int = 1) -> Experimental:
    """Matheron semivariance from points (x east, y north, z down in the stratigraphic frame).

    'major'/'minor': horizontal directions at `azimuth` / azimuth+90 (within `tolerance_deg`).
    Only pairs from DIFFERENT wells in the same layer (|dz| <= `vertical_tolerance`, which must
    be given) are used, because pairs at different depths measure the vertical, not the
    horizontal, continuity. 'vertical': computed per group (well) and pooled, so pairs never
    straddle two wells. Bins are centred on lag, 2 lag, ... nlags * lag.
    """
    if kind not in KINDS:
        raise VariogramError(f"kind must be one of {KINDS}")
    if not (lag > 0 and nlags >= 1):
        raise VariogramError("lag must be positive and nlags at least 1")
    p = np.asarray(points, float)
    v = np.asarray(values, float)
    ok = np.isfinite(v) & np.all(np.isfinite(p), axis=1)
    p, v = p[ok], v[ok]
    g_ids = None if groups is None else np.asarray(groups)[ok]
    if v.size < 2:
        raise VariogramError("need at least two valid samples")
    edges = lag * (np.arange(nlags + 1) + 0.5)
    settings = dict(lag=lag, nlags=nlags, azimuth=azimuth, tolerance_deg=tolerance_deg, bandwidth=bandwidth,
                    vertical_tolerance=vertical_tolerance)
    if kind != "vertical":
        if vertical_tolerance is None or not vertical_tolerance > 0:
            raise VariogramError("horizontal variograms need a positive vertical tolerance (the thickness of the layer "
                                 "pairs are taken from)")
        az = azimuth + (90.0 if kind == "minor" else 0.0)
        g, c = _horizontal(p, v, g_ids, edges, az, tolerance_deg, bandwidth, vertical_tolerance, max_points, seed)
        return Experimental(kind, lag * np.arange(1, nlags + 1), g, c.astype(int), float(np.var(v)), settings)
    tol = np.radians(tolerance_deg)
    d = (0.0, 0.0, 1.0)
    if g_ids is not None:
        num, cnt = np.zeros(nlags), np.zeros(nlags)
        for gid in np.unique(g_ids):
            m = g_ids == gid
            if m.sum() < 2:
                continue
            g, c = _estimate((p[m, 0], p[m, 1], p[m, 2]), v[m], edges, d, tol, bandwidth, max_points, seed)
            g = np.nan_to_num(g)
            num += g * c
            cnt += c
        gamma = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        npairs = cnt.astype(int)
    else:
        g, c = _estimate((p[:, 0], p[:, 1], p[:, 2]), v, edges, d, tol, bandwidth, max_points, seed)
        npairs = c.astype(int)
        gamma = np.where(npairs > 0, g, np.nan)
    return Experimental(kind, lag * np.arange(1, nlags + 1), gamma, npairs, float(np.var(v)), settings)


# -- fitting ----------------------------------------------------------------------------------------------------------------------------
def fit_direction(exp: Experimental, vtype: str, *, nugget: float | None = None, sill: float | None = None,
                  min_pairs: int = 1) -> dict:
    """Fit nugget, contribution and range of ONE structure to one directional experimental
    variogram by pair-weighted least squares. If `nugget` / `sill` (total) are given they are
    held fixed (used for the second and third directions). Returns
    {'nugget', 'contribution', 'range', 'rmse'}."""
    lag, gam, n = exp.usable(min_pairs)
    if lag.size < 3:
        raise VariogramError("need at least three lags with pairs to fit")
    w = np.sqrt(n.astype(float))
    top = float(np.median(gam[-max(2, lag.size // 3):]))
    free_nug = nugget is None
    free_sill = sill is None
    lo, hi, x0 = [], [], []
    if free_nug:
        lo.append(0.0); hi.append(max(top, gam.max()) * 1.0); x0.append(min(float(gam[0]), top) * 0.2)
    if free_sill:
        lo.append(1e-12); hi.append(max(gam.max(), top) * 3.0); x0.append(max(top - (x0[0] if free_nug else nugget), 1e-9))
    lo.append(float(lag[0]) * 0.1); hi.append(float(lag[-1]) * 20.0); x0.append(float(lag[-1]) / 2.0)

    def unpack(x):
        i = 0
        nug = x[i] if free_nug else nugget
        i += int(free_nug)
        c = x[i] if free_sill else max(sill - nug, 1e-12)
        i += int(free_sill)
        return nug, c, x[i]

    def resid(x):
        nug, c, a = unpack(x)
        return w * (nug + c * shape(vtype, lag / a) - gam)

    sol = least_squares(resid, x0, bounds=(lo, hi))
    nug, c, a = unpack(sol.x)
    rmse = float(np.sqrt(np.sum(w * w * (resid(sol.x) / w) ** 2) / np.sum(w * w)))
    return dict(nugget=float(nug), contribution=float(c), range=float(a), rmse=rmse)


# -- JSON ----------------------------------------------------------------------------------------------------------------------------
def to_dict(m: VariogramModel) -> dict:
    return dict(nugget=m.nugget, azimuth=m.azimuth, transform=m.transform, variable=m.variable,
                structures=[dict(type=s.type, contribution=s.contribution, range_major=s.range_major,
                                 range_minor=s.range_minor, range_vert=s.range_vert) for s in m.structures])


def from_dict(d: dict) -> VariogramModel:
    try:
        return VariogramModel(float(d["nugget"]), tuple(Structure(s["type"], float(s["contribution"]), float(s["range_major"]),
                                                                   float(s["range_minor"]), float(s["range_vert"]))
                                                         for s in d["structures"]),
                              float(d.get("azimuth", 0.0)), d.get("transform", "none"), d.get("variable", "")).check()
    except (KeyError, TypeError) as exc:
        raise VariogramError(f"variogram JSON is incomplete or malformed: {exc!r}") from None


def export_variogram(m: VariogramModel, path) -> str:
    """Write a model as JSON (.json) or a GSLIB parameter block (anything else). Returns the format."""
    import json
    from pathlib import Path
    path = Path(path)
    m.check()
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(to_dict(m), indent=2), encoding="utf-8")
        return "json"
    path.write_text(to_gslib(m), encoding="utf-8")
    return "gslib"
