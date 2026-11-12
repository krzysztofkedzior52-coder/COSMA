"""Grid geometry: horizons -> layered corner-point grid (vertical pillars, no fault stepping yet).

Everything is explicit: each interval needs a layering choice and each horizon a contact type.

Layering modes (per interval):
  fixed_count      n layers, thickness proportional to the interval thickness in each column
  fixed_thickness  layers of about ``dz``: n = ceil(thickest / dz); each column uses
                   ceil(t/dz) of them, equally thick, the rest collapse at the base
  follow_top       layers of exactly ``dz`` stacked down from the top (parallel to the top);
                   what does not fit above the base collapses at the base
  follow_base      layers of exactly ``dz`` stacked up from the base; collapse at the top

Contact types, applied to the horizon at the top of an interval, after any throw:
  conformable  nothing is clipped; an inverted pair is a data error, repaired and reported
  erosional    horizons below cannot rise above this horizon (older beds truncated)
  onlap        horizons above cannot sink below this horizon (younger beds terminate on it)
Cells thinner than ``min_thickness`` are made inactive; their geometry is kept so the
volume lost is reportable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundary import GridSpec
from .cornerpoint import CornerPointGrid

MODES = ("fixed_count", "fixed_thickness", "follow_top", "follow_base")
CONTACTS = ("conformable", "erosional", "onlap")
DEFAULT_MIN_THICKNESS = 0.1
MAX_CELLS = 50_000_000


class BuildError(ValueError):
    """Inconsistent or incomplete grid-building input."""


@dataclass(frozen=True)
class Layering:
    mode: str
    n: int | None = None      # fixed_count
    dz: float | None = None   # the other three

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise BuildError(f"layering mode {self.mode!r} is not one of {', '.join(MODES)}")
        if self.mode == "fixed_count":
            if self.n is None or int(self.n) != self.n or self.n < 1:
                raise BuildError("fixed_count needs a whole number of layers n >= 1")
            if self.dz is not None:
                raise BuildError("fixed_count takes n, not dz")
        else:
            if self.dz is None or not np.isfinite(self.dz) or self.dz <= 0:
                raise BuildError(f"{self.mode} needs a layer thickness dz > 0")
            if self.n is not None:
                raise BuildError(f"{self.mode} takes dz, not n")


@dataclass
class BuildReport:
    layers_per_interval: list
    layer_interval: np.ndarray         # (nz,) interval index of every layer
    clipped_corners: list              # per horizon: corners moved by a contact rule
    inverted_corners: list             # per horizon: corners repaired in a conformable pair
    undefined_columns: int             # active_xy columns switched off: horizon undefined
    thin_cells: int                    # active-column cells made inactive (< min_thickness)
    lost_volume: float                 # their geometric volume, m3
    total_volume: float                # geometric volume of all cells in active columns, m3
    warnings: list = field(default_factory=list)


def pillars_to_corners(a, nx: int, ny: int) -> np.ndarray:
    """(ny+1, nx+1) pillar values -> (ny, 2, nx, 2) per-cell-corner values."""
    j = np.arange(ny)[:, None, None, None] + np.arange(2)[None, :, None, None]
    i = np.arange(nx)[None, None, :, None] + np.arange(2)[None, None, None, :]
    return np.asarray(a, float)[j, i]


def _as_corners(a, nx: int, ny: int) -> np.ndarray:
    a = np.asarray(a, float)
    if a.shape == (ny + 1, nx + 1):
        return pillars_to_corners(a, nx, ny)
    if a.shape == (ny, 2, nx, 2):
        return a
    raise BuildError(f"horizon array has shape {a.shape}; expected pillar shape {(ny + 1, nx + 1)} "
                     f"or corner shape {(ny, 2, nx, 2)}")


def _check_inputs(spec, horizons, contacts, layerings, min_thickness):
    nh = len(horizons)
    if nh < 2:
        raise BuildError("need at least two horizons (a top and a base)")
    ni = nh - 1
    if len(layerings) != ni:
        raise BuildError(f"{ni} intervals but {len(layerings)} layering choices; every interval needs one")
    if len(contacts) != ni:
        raise BuildError(f"{ni} intervals but {len(contacts)} contact types")
    for lay in layerings:
        if not isinstance(lay, Layering):
            raise BuildError("layerings must be Layering objects (no default is assumed)")
    for c in contacts:
        if c not in CONTACTS:
            raise BuildError(f"contact {c!r} is not one of {', '.join(CONTACTS)}")
    if not (np.isfinite(min_thickness) and min_thickness >= 0):
        raise BuildError("min_thickness must be >= 0")


def sample_horizons(spec: GridSpec, surfaces) -> list:
    """Sample surfaces (map coordinates, depth down) at the pillar positions. NaN where a
    surface is undefined or does not cover the pillar."""
    X, Y = np.meshgrid(spec.x_edges, spec.y_edges)
    mx, my = spec.frame.to_map(X, Y)
    return [np.asarray(s.sample(mx, my), float) for s in surfaces]


def apply_contacts(horizons, contacts):
    """Return (clipped horizons, per-horizon clipped counts, per-horizon inverted counts)."""
    d = [np.array(h, float) for h in horizons]
    nh = len(d)
    clipped = [0] * nh
    for h in range(1, nh - 1):
        c = contacts[h]  # contact at the top of interval h = horizon h
        if c == "erosional":
            for j in range(h + 1, nh):
                new = np.fmax(d[j], d[h])
                clipped[j] += int(np.sum(new != d[j]))
                d[j] = new
        elif c == "onlap":
            for j in range(h):
                new = np.fmin(d[j], d[h])
                clipped[j] += int(np.sum(new != d[j]))
                d[j] = new
    inverted = [0] * nh
    for j in range(1, nh):
        new = np.fmax(d[j], d[j - 1])
        inverted[j] += int(np.sum(new != d[j]))
        d[j] = new
    return d, clipped, inverted


def _layer_count(lay: Layering, tmax: float) -> int:
    if lay.mode == "fixed_count":
        return int(lay.n)
    return max(1, int(np.ceil(tmax / lay.dz - 1e-9)))


def _boundaries(lay: Layering, n: int, top, base):
    """Layer boundary depths, shape top.shape + (n + 1,)."""
    t = base - top
    k = np.arange(n + 1)
    if lay.mode == "fixed_count":
        return top[..., None] + t[..., None] * (k / n)
    if lay.mode == "fixed_thickness":
        ncol = np.clip(np.ceil(t / lay.dz - 1e-9), 1, n)
        return top[..., None] + t[..., None] * np.minimum(k, ncol[..., None]) / ncol[..., None]
    if lay.mode == "follow_top":
        return top[..., None] + np.minimum(k * lay.dz, t[..., None])
    return base[..., None] - np.minimum((n - k) * lay.dz, t[..., None])


def build_grid(spec: GridSpec, horizons, contacts, layerings,
               min_thickness: float = DEFAULT_MIN_THICKNESS) -> tuple[CornerPointGrid, BuildReport]:
    """Build the grid. ``horizons``: nh depth arrays, top to base, either per pillar (ny+1, nx+1) or per cell
    corner (ny, 2, nx, 2) (which lets cells on either side of a fault disagree at their shared pillar);
    ``contacts``: per interval, the contact at its top (the first entry has no effect);
    ``layerings``: one :class:`Layering` per interval."""
    _check_inputs(spec, horizons, contacts, layerings, min_thickness)
    nx, ny = spec.nx, spec.ny
    ni = len(horizons) - 1
    raw = [_as_corners(h, nx, ny) for h in horizons]
    undefined = np.zeros((ny, 2, nx, 2), bool)
    for h in raw:
        undefined |= ~np.isfinite(h)
    filled = [np.where(np.isfinite(h), h, np.nanmean(h) if np.isfinite(h).any() else 0.0) for h in raw]
    d, clipped, inverted = apply_contacts(filled, contacts)
    warnings: list[str] = []

    # a cell is usable only when all four of its corners are defined for every horizon
    col_ok = ~undefined.any(axis=(1, 3))          # (ny, nx)
    col_ok = col_ok.T                              # (nx, ny)
    undefined_cols = int(np.sum(spec.active_xy & ~col_ok))
    if undefined_cols:
        warnings.append(f"{undefined_cols} active columns have an undefined horizon at a corner and were made inactive")
    use = ~undefined

    counts, zs = [], []
    for i in range(ni):
        t = d[i + 1] - d[i]
        tmax = float(t[use].max()) if use.any() else 0.0
        n = _layer_count(layerings[i], tmax)
        counts.append(n)
        zs.append(_boundaries(layerings[i], n, d[i], d[i + 1])[..., :-1])
    nz = sum(counts)
    if nx * ny * nz > MAX_CELLS:
        raise BuildError(f"{nx} x {ny} x {nz} = {nx * ny * nz:,} cells exceeds the limit of {MAX_CELLS:,}")
    zb = np.concatenate(zs + [d[-1][..., None]], axis=-1)  # (ny, 2, nx, 2, nz+1)
    layer_interval = np.repeat(np.arange(ni), counts)

    # pillar z range: every corner that touches the pillar
    zmin = np.full((ny + 1, nx + 1), np.inf)
    zmax = np.full((ny + 1, nx + 1), -np.inf)
    for jj in (0, 1):
        for ii in (0, 1):
            top = zb[:, jj, :, ii, 0]
            bot = zb[:, jj, :, ii, -1]
            sl = (slice(jj, jj + ny), slice(ii, ii + nx))
            zmin[sl] = np.minimum(zmin[sl], top)
            zmax[sl] = np.maximum(zmax[sl], bot)
    coord = np.empty((ny + 1, nx + 1, 6))
    coord[..., 0] = coord[..., 3] = spec.x_edges[None, :]
    coord[..., 1] = coord[..., 4] = spec.y_edges[:, None]
    coord[..., 2] = zmin
    coord[..., 5] = zmax

    zcorn = np.empty((nz, 2, ny, 2, nx, 2))
    for t_ in (0, 1):
        zcorn[:, t_] = np.transpose(zb[..., t_:t_ + nz], (4, 0, 1, 2, 3))

    thick = _cell_thickness(zcorn, nx, ny, nz)
    col_active = spec.active_xy & col_ok
    thin = thick < min_thickness
    actnum = (col_active[:, :, None] & ~thin).astype(np.int32)
    thin_in_cols = col_active[:, :, None] & thin
    vol = thick * spec.column_area[:, :, None]
    lost = float(vol[thin_in_cols].sum())
    total = float(vol[np.broadcast_to(col_active[:, :, None], thin.shape)].sum())
    if thin_in_cols.any():
        warnings.append(f"{int(thin_in_cols.sum())} cells thinner than {min_thickness:g} m were made inactive "
                        f"({lost:,.1f} m3 of {total:,.1f} m3)")
    for h, n_ in enumerate(inverted):
        if n_:
            warnings.append(f"horizon {h}: {n_} cell corners lay above horizon {h - 1}; "
                            "the pair was repaired to zero thickness (check the surfaces)")

    grid = CornerPointGrid(nx, ny, nz, coord, zcorn, actnum, {}, j_north=True, frame=spec.frame)
    report = BuildReport(counts, layer_interval, clipped, inverted, undefined_cols,
                         int(thin_in_cols.sum()), lost, total, warnings)
    return grid, report


def _cell_thickness(zcorn, nx, ny, nz):
    dz = zcorn[:, 1] - zcorn[:, 0]            # (nz, ny, 2, nx, 2)
    mean = dz.mean(axis=(2, 4))               # (nz, ny, nx)
    return np.transpose(mean, (2, 1, 0))      # (nx, ny, nz)
