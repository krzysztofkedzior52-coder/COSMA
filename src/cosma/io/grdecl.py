"""GRDECL writer (and a tiny keyword scanner used for self-checks).

The writer is deliberately our own: export is the product, and xtgeo's
writer would hide the ordering and precision choices we need to control.
Readers (xtgeo, resdata) are used in tests as independent validators.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from ..grid.cornerpoint import CornerPointGrid

_PER_LINE = 6  # values per line; keeps lines short and diffable
_GEOM_FMT = "%.10g"
_PROP_FMT = "%.8g"
_INT_PROPS = frozenset({"FIPNUM", "SATNUM", "PVTNUM", "EQLNUM", "ACTNUM", "FLUXNUM", "MULTNUM", "ROCKNUM"})


def _flatten_prop(a: np.ndarray) -> np.ndarray:
    """(nx, ny, nz) -> 1D with I fastest, then J, then K."""
    return np.asarray(a).ravel(order="F")


def _write_block(f, keyword: str, values: Iterable, fmt: str, compress: bool = True) -> None:
    """Write one keyword block, run-length compressing repeats as ``n*value``."""
    f.write(f"{keyword}\n")
    vals = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    tokens: list[str] = []
    n = len(vals)
    p = 0
    while p < n:
        q = p + 1
        if compress:
            while q < n and vals[q] == vals[p]:
                q += 1
        run = q - p
        s = fmt % vals[p] if vals.dtype.kind == "f" else str(int(vals[p]))
        tokens.append(f"{run}*{s}" if run > 1 else s)
        p = q
    for s in range(0, len(tokens), _PER_LINE):
        f.write(" ".join(tokens[s : s + _PER_LINE]) + "\n")
    f.write("/\n\n")


FLOAT32_WARN_M = 0.01  # absolute-coordinate export warns above this rounding error


def float32_coordinate_error(grid: CornerPointGrid, coordinates: str = "local") -> float:
    """Largest error (m) a consumer storing pillar x, y as float32 would
    introduce. float32 has a 0.5 m step at a northing of 4.7 million, so
    absolute UTM coordinates lose up to 0.25 m per pillar, and because pillars
    round independently, cell sizes are distorted."""
    xy = _pillar_xy(grid, coordinates)
    return float(np.abs(xy.astype(np.float32).astype(np.float64) - xy).max())


def _pillar_xy(grid: CornerPointGrid, coordinates: str) -> np.ndarray:
    if coordinates not in ("local", "absolute"):
        raise ValueError("coordinates must be 'local' or 'absolute'")
    c = grid.coord
    if coordinates == "absolute":
        top = np.stack(grid.frame.to_map(c[..., 0], c[..., 1]), axis=-1)
        bot = np.stack(grid.frame.to_map(c[..., 3], c[..., 4]), axis=-1)
    else:
        top, bot = c[..., 0:2], c[..., 3:5]
    return np.stack([top, bot], axis=-2)  # (ny+1, nx+1, 2, 2)


def write_grdecl(
    grid: CornerPointGrid,
    path: str | Path,
    properties: Iterable[str] | None = None,
    gridunit: str | None = "METRES",
    comment: str | None = None,
    coordinates: str = "local",
) -> Path:
    """Write ``grid`` to ``path`` as GRDECL.

    ``properties`` selects which of ``grid.props`` to write, in that order;
    default is all of them in insertion order. Property keyword names are the
    dict keys (any name is allowed: simulator-specific mapping comes in M6).

    ``coordinates``:
      * ``"local"`` (default): pillars stay in the grid's local frame and, when
        the frame is not the identity, ``MAPUNITS`` and ``MAPAXES`` place the
        grid on the map. This is float32-safe and is what Petrel-style
        exports do.
      * ``"absolute"``: map coordinates are written directly and no MAPAXES.
        Readers that store x, y as float32 then lose up to 0.25 m per pillar
        at UTM scale (cell volumes off by percent at 10 m cell size), so a
        warning is raised when the expected rounding error exceeds 1 cm.
    """
    path = Path(path)
    names = list(grid.props) if properties is None else list(properties)
    for n in names:
        if n not in grid.props:
            raise KeyError(f"property {n!r} not in grid")

    err32 = float32_coordinate_error(grid, coordinates)
    if coordinates == "absolute" and err32 > FLOAT32_WARN_M:
        warnings.warn(
            f"absolute coordinates: float32 consumers will round pillars by up to "
            f"{err32:.3f} m; use coordinates='local' (MAPAXES) to avoid this",
            UserWarning,
            stacklevel=2,
        )

    coord = grid.coord.copy()
    if coordinates == "absolute":
        xy = _pillar_xy(grid, "absolute")
        coord[..., 0:2], coord[..., 3:5] = xy[..., 0, :], xy[..., 1, :]
    coord = coord.reshape(-1)  # (ny+1, nx+1, 6) C order == I fastest per row of pillars
    zcorn = grid.zcorn.reshape(-1)  # C order of (nz,2,ny,2,nx,2) == Eclipse ZCORN order
    actnum = _flatten_prop(grid.actnum)
    write_mapaxes = coordinates == "local" and not grid.frame.is_identity

    with path.open("w", encoding="ascii") as f:
        f.write("-- Written by COSMA\n")
        if coordinates == "absolute":
            f.write(f"-- absolute map coordinates; float32 rounding error would be {err32:.3f} m\n")
        elif write_mapaxes:
            f.write("-- local grid coordinates; MAPAXES places the grid on the map\n")
        if comment:
            for line in comment.splitlines():
                f.write(f"-- {line}\n")
        f.write("\n")
        if write_mapaxes:
            f.write("MAPUNITS\n  METRES /\n\n")
            f.write("MAPAXES\n  " + " ".join("%.6f" % v for v in grid.frame.mapaxes(100.0)) + " /\n\n")
        f.write(f"SPECGRID\n  {grid.nx} {grid.ny} {grid.nz} 1 F /\n\n")
        if gridunit:
            f.write(f"GRIDUNIT\n  {gridunit} /\n\n")
        _write_block(f, "COORD", coord, _GEOM_FMT, compress=False)
        _write_block(f, "ZCORN", zcorn, _GEOM_FMT, compress=True)
        _write_block(f, "ACTNUM", actnum, "%d", compress=True)
        for n in names:
            _write_block(f, n, _flatten_prop(grid.props[n]), "%d" if n in _INT_PROPS else _PROP_FMT, compress=True)
    return path


def scan_keywords(path: str | Path) -> dict[str, int]:
    """Minimal independent scan: keyword -> number of values after expanding
    ``n*v`` repeats. Used to check counts without any third-party reader."""
    counts: dict[str, int] = {}
    key: str | None = None
    n = 0
    for raw in Path(path).read_text(encoding="ascii").splitlines():
        line = raw.split("--", 1)[0].strip()
        if not line:
            continue
        if key is None:
            key = line.split()[0]
            n = 0
            rest = line.split()[1:]
        else:
            rest = line.split()
        for tok in rest:
            if tok == "/":
                counts[key] = n
                key = None
                break
            if tok.endswith("/"):
                tok = tok[:-1]
                closing = True
            else:
                closing = False
            if tok:
                n += int(tok.split("*", 1)[0]) if "*" in tok else 1
            if closing:
                counts[key] = n
                key = None
                break
    return counts
