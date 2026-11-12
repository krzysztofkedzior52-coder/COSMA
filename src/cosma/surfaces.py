"""Depth surfaces (regular grids) in IRAP ASCII / IRAP binary / ZMAP+ ASCII.

A `Surface` is always stored in metres, with depth positive down relative to the
model datum (TVDSS). Nothing about units or sign is guessed: the caller states the
horizontal unit, the vertical unit and whether the file's values are positive down
(depth) or positive up (elevation).

Node (i, j) sits at
    X = xori + i*xinc*cos(rot) - j*yinc*yflip*sin(rot)
    Y = yori + i*xinc*sin(rot) + j*yinc*yflip*cos(rot)
with `rot` counter-clockwise from +x (IRAP convention) and yflip = +1 for a normal
right-handed map grid.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .units import unit_factor


class SurfaceError(ValueError):
    pass


@dataclass(frozen=True)
class Surface:
    xori: float
    yori: float
    xinc: float
    yinc: float
    rotation: float            # degrees, counter-clockwise from +x
    yflip: int                 # +1 or -1
    values: np.ndarray         # (ncol, nrow), metres, depth positive down, NaN = undefined
    name: str = ""

    def __post_init__(self):
        if self.values.ndim != 2 or min(self.values.shape) < 2:
            raise SurfaceError("surface needs at least 2 x 2 nodes")
        if not (self.xinc > 0 and self.yinc > 0):
            raise SurfaceError("xinc and yinc must be positive")
        if self.yflip not in (1, -1):
            raise SurfaceError("yflip must be +1 or -1")

    @property
    def ncol(self):
        return self.values.shape[0]

    @property
    def nrow(self):
        return self.values.shape[1]

    def node_xy(self):
        """(X, Y) arrays of shape (ncol, nrow)."""
        r = np.radians(self.rotation)
        u = (np.arange(self.ncol) * self.xinc)[:, None]
        v = (np.arange(self.nrow) * self.yinc * self.yflip)[None, :]
        return (self.xori + u * np.cos(r) - v * np.sin(r),
                self.yori + u * np.sin(r) + v * np.cos(r))

    def sample(self, x, y):
        """Bilinear depth at (x, y), NaN outside the grid or where any of the four
        surrounding nodes is undefined."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        r = np.radians(self.rotation)
        dx, dy = x - self.xori, y - self.yori
        u = dx * np.cos(r) + dy * np.sin(r)
        v = -dx * np.sin(r) + dy * np.cos(r)
        fi = u / self.xinc
        fj = v / (self.yinc * self.yflip)
        eps = 1e-9
        inside = (fi >= -eps) & (fi <= self.ncol - 1 + eps) & (fj >= -eps) & (fj <= self.nrow - 1 + eps)
        fi = np.clip(fi, 0, self.ncol - 1)
        fj = np.clip(fj, 0, self.nrow - 1)
        i0 = np.minimum(np.floor(fi).astype(int), self.ncol - 2)
        j0 = np.minimum(np.floor(fj).astype(int), self.nrow - 2)
        a, b = fi - i0, fj - j0
        z = self.values
        out = ((1 - a) * (1 - b) * z[i0, j0] + a * (1 - b) * z[i0 + 1, j0]
               + (1 - a) * b * z[i0, j0 + 1] + a * b * z[i0 + 1, j0 + 1])
        return np.where(inside, out, np.nan)

    def footprint(self):
        """Corner coordinates of the grid outline, counter-clockwise or clockwise as the grid is."""
        X, Y = self.node_xy()
        idx = [(0, 0), (-1, 0), (-1, -1), (0, -1)]
        return np.array([(X[i, j], Y[i, j]) for i, j in idx])


_FORMATS = {"irap_ascii": "irap_ascii", "irap_binary": "irap_binary", "zmap_ascii": "zmap_ascii"}
_EXT = {".irap": "irap_ascii", ".gri": "irap_binary", ".zmap": "zmap_ascii", ".dat": "zmap_ascii"}


def _sniff(path: Path) -> str:
    ext = _EXT.get(path.suffix.lower())
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"!") or head.startswith(b"@"):
        return "zmap_ascii"
    if head[:4] == b"\x00\x00\x00\x20":      # IRAP binary: first Fortran record is 32 bytes (big endian)
        return "irap_binary"
    first = head.split(None, 1)[0] if head.split() else b""
    if first == b"-996":
        return "irap_ascii"
    if ext:
        return ext
    raise SurfaceError(f"{path.name}: cannot recognise the surface format; pass format=")


def read_surface(path, *, xy_unit: str, z_unit: str, positive: str = "down", format: str | None = None) -> Surface:
    """Read a regular surface.

    xy_unit, z_unit : unit of the file's horizontal / vertical numbers (required).
    positive        : "down" if values are depths, "up" if they are elevations (then negated).
    """
    import xtgeo

    path = Path(path)
    if not path.exists():
        raise SurfaceError(f"{path}: no such file")
    if positive not in ("down", "up"):
        raise SurfaceError("positive must be 'down' or 'up'")
    fx, fz = unit_factor(xy_unit), unit_factor(z_unit)
    fmt = format or _sniff(path)
    if fmt not in _FORMATS:
        raise SurfaceError(f"unsupported surface format {fmt!r}; known: {sorted(_FORMATS)}")
    try:
        s = xtgeo.surface_from_file(str(path), fformat=fmt)
    except Exception as exc:  # xtgeo raises assorted types for malformed files
        raise SurfaceError(f"{path.name}: cannot read as {fmt}: {exc}") from exc
    if int(s.yflip) != 1:
        raise SurfaceError(f"{path.name}: yflip={int(s.yflip)} (y axis reversed). Unsupported: readers disagree on where "
                           "such a grid is anchored, so it is refused rather than risk a shifted surface.")
    vals = np.ma.filled(s.values.astype(float), np.nan) * fz
    vals[~np.isfinite(vals)] = np.nan
    if positive == "up":
        vals = -vals
    if not np.isfinite(vals).any():
        raise SurfaceError(f"{path.name}: surface has no defined nodes")
    if positive == "down" and np.nanmax(vals) <= 0:
        warnings.warn(f"{path.name}: every value is <= 0 but positive='down'. Elevations (negative below "
                      "sea level) would look like this; pass positive='up' if so.", UserWarning, stacklevel=2)
    return Surface(
        xori=float(s.xori) * fx, yori=float(s.yori) * fx,
        xinc=float(s.xinc) * fx, yinc=float(s.yinc) * fx,
        rotation=float(s.rotation), yflip=int(s.yflip), values=vals, name=path.stem,
    )
