"""Model boundary treatments. The user may pick any combination:

  closed             nothing selected: columns outside the boundary polygon are inactive, no flow through the edge
  extended buffer    columns whose centre lies within a distance of the polygon are active too (region 2)
  pore-volume mult.  columns along the outer edge of the active area get a MULTPV factor (grid include file)
  aquifer            analytic Fetkovich or Carter-Tracy aquifer on chosen outer faces (solution include file)

PV multipliers and an aquifer both add volume outside the model; choosing both is allowed but flagged.
Deck text is OPM Flow / Eclipse syntax in METRIC units (m, bar, mD, sm3/day/bar, 1/bar).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy import ndimage

from .boundary import GridSpec, validate_polygon


class BoundaryOptionError(ValueError):
    pass


FACES = ("I-", "I+", "J-", "J+")
RULES = ("center", "fraction")


def _pos(x, name):
    if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise BoundaryOptionError(f"{name} must be a positive number")


@dataclass(frozen=True)
class PVMultiplier:
    factor: float
    rings: int                      # how many cell rows in from the outer edge carry the factor

    def __post_init__(self):
        _pos(self.factor, "the pore-volume multiplier")
        if int(self.rings) != self.rings or self.rings < 1:
            raise BoundaryOptionError("the number of edge rings must be a whole number >= 1")


@dataclass(frozen=True)
class AquiferSpec:
    kind: str                       # "fetkovich" | "carter_tracy"
    faces: tuple                    # which outer faces are connected: subset of FACES
    datum_depth: float              # m
    initial_pressure: float | None  # bar; None = equilibrate with the reservoir at the datum depth
    compressibility: float          # total, 1/bar
    influx_multiplier: float
    volume: float | None = None               # fetkovich: initial water volume, m3
    productivity_index: float | None = None   # fetkovich: sm3/day/bar
    permeability: float | None = None         # carter-tracy: mD
    porosity: float | None = None             # carter-tracy: fraction
    external_radius: float | None = None      # carter-tracy: m
    thickness: float | None = None            # carter-tracy: m
    influx_angle: float | None = None         # carter-tracy: degrees
    table: int | None = None                  # carter-tracy: influence table number (1 = infinite aquifer)

    _NEED = {"fetkovich": ("volume", "productivity_index"),
             "carter_tracy": ("permeability", "porosity", "external_radius", "thickness", "influx_angle", "table")}

    def __post_init__(self):
        if self.kind not in self._NEED:
            raise BoundaryOptionError("aquifer kind must be 'fetkovich' or 'carter_tracy'")
        if not self.faces or any(f not in FACES for f in self.faces) or len(set(self.faces)) != len(self.faces):
            raise BoundaryOptionError("choose which outer faces the aquifer connects to: " + ", ".join(FACES))
        if not isinstance(self.datum_depth, (int, float)) or not math.isfinite(self.datum_depth):
            raise BoundaryOptionError("the aquifer datum depth must be a number")
        if self.initial_pressure is not None:
            _pos(self.initial_pressure, "the initial aquifer pressure")
        _pos(self.compressibility, "the aquifer compressibility")
        _pos(self.influx_multiplier, "the influx multiplier")
        for n in self._NEED[self.kind]:
            v = getattr(self, n)
            if v is None:
                raise BoundaryOptionError(f"{self.kind.replace('_', '-')} aquifer needs {n.replace('_', ' ')}")
            _pos(v, n.replace("_", " "))
        if self.kind == "carter_tracy":
            if self.porosity > 1:
                raise BoundaryOptionError("aquifer porosity is a fraction (0-1)")
            if self.influx_angle > 360:
                raise BoundaryOptionError("the influx angle cannot exceed 360 degrees")
            if int(self.table) != self.table:
                raise BoundaryOptionError("the influence table number must be a whole number")
        else:
            for n in ("permeability", "porosity", "external_radius", "thickness", "influx_angle", "table"):
                if getattr(self, n) is not None:
                    raise BoundaryOptionError(f"{n.replace('_', ' ')} belongs to the Carter-Tracy aquifer")
        if self.kind == "carter_tracy":
            for n in ("volume", "productivity_index"):
                if getattr(self, n) is not None:
                    raise BoundaryOptionError(f"{n.replace('_', ' ')} belongs to the Fetkovich aquifer")


@dataclass(frozen=True)
class BoundaryOptions:
    rule: str = "center"                 # which columns are inside the polygon
    min_fraction: float = 0.5            # for rule == "fraction"
    buffer: float | None = None          # m beyond the polygon
    pv: PVMultiplier | None = None
    aquifer: AquiferSpec | None = None

    def __post_init__(self):
        if self.rule not in RULES:
            raise BoundaryOptionError("the inside rule must be 'center' or 'fraction'")
        if not (0 < self.min_fraction <= 1):
            raise BoundaryOptionError("the minimum fraction must be in (0, 1]")
        if self.buffer is not None:
            _pos(self.buffer, "the buffer distance")

    @property
    def closed(self) -> bool:
        return self.buffer is None and self.pv is None and self.aquifer is None

    def to_state(self) -> dict:
        d = dict(rule=self.rule, min_fraction=self.min_fraction, buffer=self.buffer, pv=None, aquifer=None)
        if self.pv is not None:
            d["pv"] = dict(factor=self.pv.factor, rings=self.pv.rings)
        if self.aquifer is not None:
            a = self.aquifer
            d["aquifer"] = {k: (list(getattr(a, k)) if k == "faces" else getattr(a, k))
                            for k in ("kind", "faces", "datum_depth", "initial_pressure", "compressibility",
                                      "influx_multiplier", "volume", "productivity_index", "permeability", "porosity",
                                      "external_radius", "thickness", "influx_angle", "table")}
        return d

    @classmethod
    def from_state(cls, st: dict) -> "BoundaryOptions":
        try:
            pv = None if st["pv"] is None else PVMultiplier(st["pv"]["factor"], st["pv"]["rings"])
            aq = None
            if st["aquifer"] is not None:
                a = dict(st["aquifer"])
                a["faces"] = tuple(a["faces"])
                aq = AquiferSpec(**a)
            return cls(st["rule"], st["min_fraction"], st["buffer"], pv, aq)
        except (KeyError, TypeError) as exc:
            raise BoundaryOptionError(f"bad boundary options: {exc}") from None


# -- geometry -----------------------------------------------------------------------------------------------------------

def _local_polygon(spec: GridSpec, polygon):
    def fn(a):
        x, y = spec.frame.to_local(a[:, 0], a[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(validate_polygon(polygon), fn)


def boundary_layout(spec: GridSpec, polygon, opt: BoundaryOptions):
    """Active columns and regions for a boundary polygon. Returns (active (nx,ny) bool, region (nx,ny) int where
    1 = inside the polygon, 2 = buffer, 0 = outside, warnings)."""
    nx, ny = spec.nx, spec.ny
    lp = _local_polygon(spec, polygon)
    xc = 0.5 * (spec.x_edges[:-1] + spec.x_edges[1:])
    yc = 0.5 * (spec.y_edges[:-1] + spec.y_edges[1:])
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    warnings = []
    if opt.rule == "center":
        inside = shapely.contains_xy(lp.buffer(1e-6, join_style="mitre"), XC, YC)
    else:
        XE0, YE0 = np.meshgrid(spec.x_edges[:-1], spec.y_edges[:-1], indexing="ij")
        XE1, YE1 = np.meshgrid(spec.x_edges[1:], spec.y_edges[1:], indexing="ij")
        boxes = shapely.box(XE0.ravel(), YE0.ravel(), XE1.ravel(), YE1.ravel())
        frac = shapely.area(shapely.intersection(boxes, lp)) / shapely.area(boxes)
        inside = (frac >= opt.min_fraction - 1e-9).reshape(nx, ny)
    region = np.where(inside, 1, 0)
    if opt.buffer is not None:
        wide = lp.buffer(opt.buffer, join_style="mitre")
        in_buf = shapely.contains_xy(wide.buffer(1e-6, join_style="mitre"), XC, YC)
        region = np.where(inside, 1, np.where(in_buf, 2, 0))
        grid_box = shapely.box(spec.x_edges[0], spec.y_edges[0], spec.x_edges[-1], spec.y_edges[-1])
        lost = 1.0 - shapely.area(shapely.intersection(wide, grid_box)) / shapely.area(wide)
        if lost > 1e-6:
            warnings.append(f"the grid is too small for the buffer: {100 * lost:.1f}% of the area within "
                            f"{opt.buffer:g} m of the boundary lies outside the grid")
    if not (region > 0).any():
        raise BoundaryOptionError("no grid columns fall inside the boundary; the cell size may be too large or the "
                                  "polygon is not in the model coordinate system")
    return region > 0, region.astype(int), warnings


def edge_distance(active: np.ndarray) -> np.ndarray:
    """Taxicab distance (in columns) of every active column to the nearest inactive column or the grid edge:
    1 on the outer edge of the active area. 0 for inactive columns."""
    padded = np.pad(np.asarray(active, bool), 1, constant_values=False)
    d = ndimage.distance_transform_cdt(padded, metric="taxicab")
    return d[1:-1, 1:-1]


def edge_columns(active: np.ndarray, rings: int) -> np.ndarray:
    d = edge_distance(active)
    return (d >= 1) & (d <= rings)


def multpv_cube(active: np.ndarray, pv: PVMultiplier, nz: int) -> np.ndarray:
    cols = np.where(edge_columns(active, pv.rings), pv.factor, 1.0)
    return np.repeat(cols[:, :, None], nz, axis=2)


def face_cells(active: np.ndarray, faces) -> dict:
    """Outer faces: face name -> boolean (nx, ny) of active columns whose neighbour in that direction is inactive
    or outside the grid. 'I-' is the face towards lower I."""
    a = np.asarray(active, bool)
    p = np.pad(a, 1, constant_values=False)
    nb = {"I-": p[:-2, 1:-1], "I+": p[2:, 1:-1], "J-": p[1:-1, :-2], "J+": p[1:-1, 2:]}
    return {f: a & ~nb[f] for f in faces}


def connection_runs(active: np.ndarray, faces) -> list:
    """AQUANCON boxes: (i1, i2, j1, j2, face), 1-based, merging neighbouring faces of the same side."""
    out = []
    for f, m in face_cells(active, faces).items():
        nx, ny = m.shape
        if f in ("I-", "I+"):
            for i in range(nx):
                j = 0
                while j < ny:
                    if m[i, j]:
                        j2 = j
                        while j2 + 1 < ny and m[i, j2 + 1]:
                            j2 += 1
                        out.append((i + 1, i + 1, j + 1, j2 + 1, f))
                        j = j2 + 1
                    else:
                        j += 1
        else:
            for j in range(ny):
                i = 0
                while i < nx:
                    if m[i, j]:
                        i2 = i
                        while i2 + 1 < nx and m[i2 + 1, j]:
                            i2 += 1
                        out.append((i + 1, i2 + 1, j + 1, j + 1, f))
                        i = i2 + 1
                    else:
                        i += 1
    return out


# -- deck text ---------------------------------------------------------------------------------------------------------

def _g(x) -> str:
    return f"{x:.10g}"


def multpv_text(cube: np.ndarray, pv: PVMultiplier) -> str:
    nx, ny, nz = cube.shape
    flat = np.transpose(cube, (2, 1, 0)).reshape(-1)          # I fastest, then J, then K
    toks, i = [], 0
    while i < flat.size:
        j = i
        while j + 1 < flat.size and flat[j + 1] == flat[i]:
            j += 1
        n = j - i + 1
        toks.append(f"{n}*{_g(flat[i])}" if n > 1 else _g(flat[i]))
        i = j + 1
    lines, cur = [], ""
    for t in toks:
        if len(cur) + len(t) + 1 > 100:
            lines.append(cur)
            cur = ""
        cur += (" " if cur else "") + t
    lines.append(cur)
    return ("-- COSMA boundary: pore-volume multipliers (factor %s on the outer %d ring(s) of columns)\n"
            "-- Place this file's contents in the GRID section (INCLUDE). Cell order: I fastest, then J, then K.\n"
            "MULTPV\n  %s /\n") % (_g(pv.factor), pv.rings, "\n  ".join(lines))


def aquifer_text(a: AquiferSpec, active: np.ndarray, nz: int) -> str:
    runs = connection_runs(active, a.faces)
    if not runs:
        raise BoundaryOptionError("the aquifer has no outer faces to connect to on the chosen sides")
    head = ("-- COSMA boundary: %s aquifer on faces %s (grid index directions), METRIC units\n"
            "-- Place this file's contents in the SOLUTION section (INCLUDE).\n"
            % (a.kind.replace("_", "-"), ", ".join(a.faces)))
    p = "1*" if a.initial_pressure is None else _g(a.initial_pressure)
    if a.kind == "fetkovich":
        body = ("AQUFETP\n  -- id datum_depth init_pressure water_volume compressibility productivity_index\n"
                f"  1 {_g(a.datum_depth)} {p} {_g(a.volume)} {_g(a.compressibility)} {_g(a.productivity_index)} /\n/\n")
    else:
        body = ("AQUCT\n  -- id datum_depth init_pressure perm poro compressibility r_ext thickness angle pvt_table influence_table\n"
                f"  1 {_g(a.datum_depth)} {p} {_g(a.permeability)} {_g(a.porosity)} {_g(a.compressibility)} "
                f"{_g(a.external_radius)} {_g(a.thickness)} {_g(a.influx_angle)} 1 {int(a.table)} /\n/\n")
    con = ["AQUANCON", "  -- id i1 i2 j1 j2 k1 k2 face influx_coeff influx_multiplier"]
    for i1, i2, j1, j2, f in runs:
        con.append(f"  1 {i1} {i2} {j1} {j2} 1 {nz} '{f}' 1* {_g(a.influx_multiplier)} /")
    con.append("/")
    return head + body + "\n".join(con) + "\n"


def aqudims_text(a: AquiferSpec, active: np.ndarray) -> str:
    """RUNSPEC dimensioning OPM requires before any analytic aquifer keyword."""
    n = len(connection_runs(active, a.faces))
    return ("-- COSMA boundary: aquifer dimensions. Place this file's contents in the RUNSPEC section (INCLUDE).\n"
            "-- items: -, -, influence tables, table rows, analytic aquifers, connection records\n"
            f"AQUDIMS\n  2* 1* 1* 1 {max(1, n)} /\n")


def combination_warnings(opt: BoundaryOptions) -> list:
    out = []
    if opt.pv is not None and opt.aquifer is not None:
        out.append("pore-volume multipliers and an aquifer both add volume outside the model; together they "
                   "double-count unless you intend it")
    if opt.pv is not None and opt.buffer is not None:
        out.append("pore-volume multipliers act on the outer ring of the buffer zone, not on the original boundary")
    return out
