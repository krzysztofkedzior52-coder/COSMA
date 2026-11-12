"""Synthetic grids with analytically known answers, used as the benchmark."""

from __future__ import annotations

import numpy as np

from .boundary import GridSpec
from .cornerpoint import CornerPointGrid
from .frame import GridFrame


def make_faulted_box(
    nx: int = 6,
    ny: int = 4,
    nz: int = 5,
    dx: float = 100.0,
    dy: float = 100.0,
    dz: float = 10.0,
    top: float = 1500.0,
    x0: float = 0.0,
    y0: float = 0.0,
    dip_x: float = 0.0,
    fault_i: int | None = 3,
    throw: float = 0.0,
    j_north: bool = True,
    poro: float | np.ndarray = 0.2,
    kx: float | np.ndarray = 100.0,
    ky: float | np.ndarray | None = None,
    kz: float | np.ndarray | None = None,
    spec: GridSpec | None = None,
    frame: GridFrame | None = None,
) -> CornerPointGrid:
    """Build a box grid with vertical pillars and one vertical fault.

    Areal layout comes either from ``nx, ny, dx, dy, x0, y0`` (plain box in
    local coordinates, optionally placed on the map with ``frame``) or from a
    :class:`GridSpec` (frame, column edges and the active-column mask from a
    boundary definition). With a spec, ``nx, ny, dx, dy, x0, y0`` are ignored
    and the J axis must point north (local +y).

    The fault plane sits between cell columns ``fault_i - 1`` and ``fault_i``
    and strikes along J. Cells with ``i >= fault_i`` (the hanging wall) are
    moved down by ``throw`` metres. Layers have constant vertical thickness
    ``dz``; the top horizon dips with slope ``dip_x`` (m of depth per m of
    local x), which makes axis-order mistakes visible.

    Analytical facts used by the tests:
      * the bulk volume of cell (i, j, k) is width_i * height_j * dz (parallel
        top and bottom faces, vertical pillars), independent of throw and dip;
      * the depth of the top-west corner of cell (i, j, k) is
        top + dip_x * (x_edge_i - x_edge_0) + k * dz + (throw if
        fault_i is not None and i >= fault_i else 0).
    """
    if spec is not None:
        if not j_north:
            raise ValueError("a GridSpec grid must have J pointing along local +y")
        if frame is not None:
            raise ValueError("pass either spec or frame, not both")
        x_edges = np.asarray(spec.x_edges, dtype=float)
        y_edges = np.asarray(spec.y_edges, dtype=float)
        nx, ny = spec.nx, spec.ny
        frame = spec.frame
        active_xy = spec.active_xy
    else:
        x_edges = x0 + dx * np.arange(nx + 1)
        y_edges = y0 + dy * np.arange(ny + 1)
        frame = frame if frame is not None else GridFrame()
        active_xy = None

    if fault_i is not None and not (0 < fault_i < nx):
        raise ValueError("fault_i must lie strictly inside the grid (1..nx-1)")

    ysign = 1.0 if j_north else -1.0
    xs = x_edges
    ys = y_edges[0] + ysign * (y_edges - y_edges[0])

    # Pillars: vertical, top at z_top(x) and bottom at z_top(x) + nz*dz
    # (faulted pillars carry the footwall z; hanging-wall shifts live in ZCORN).
    coord = np.empty((ny + 1, nx + 1, 6))
    ztop_p = top + dip_x * (xs - xs[0])
    coord[..., 0] = xs[None, :]
    coord[..., 3] = xs[None, :]
    coord[..., 1] = ys[:, None]
    coord[..., 4] = ys[:, None]
    coord[..., 2] = ztop_p[None, :]
    coord[..., 5] = ztop_p[None, :] + nz * dz + (throw if fault_i is not None else 0.0)

    # ZCORN: (nz, 2, ny, 2, nx, 2). Build per-cell top depth then expand.
    ii = np.arange(nx)
    zcorn = np.empty((nz, 2, ny, 2, nx, 2))
    shift = np.where(ii >= fault_i, throw, 0.0) if fault_i is not None else np.zeros(nx)
    for t in (0, 1):  # 0 = top face of layer, 1 = bottom face
        for is_ in (0, 1):  # 0 = west edge, 1 = east edge
            zbase = top + dip_x * (x_edges[ii + is_] - xs[0])  # (nx,)
            for k in range(nz):
                z = zbase + (k + t) * dz + shift  # (nx,)
                zcorn[k, t, :, :, :, is_] = z[None, None, :]

    actnum = np.ones((nx, ny, nz), dtype=np.int32)
    if active_xy is not None:
        actnum[:] = active_xy[:, :, None].astype(np.int32)

    props = {
        "PORO": _as_cube(poro, nx, ny, nz),
        "PERMX": _as_cube(kx, nx, ny, nz),
    }
    props["PERMY"] = _as_cube(kx if ky is None else ky, nx, ny, nz)
    props["PERMZ"] = _as_cube(0.1 * np.asarray(kx) if kz is None else kz, nx, ny, nz)
    return CornerPointGrid(nx, ny, nz, coord, zcorn, actnum, props, j_north=j_north, frame=frame)


def _as_cube(v: float | np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0:
        return np.full((nx, ny, nz), float(arr))
    if arr.shape != (nx, ny, nz):
        raise ValueError(f"expected shape {(nx, ny, nz)}, got {arr.shape}")
    return arr.copy()
