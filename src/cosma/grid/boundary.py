"""Model boundary -> grid placement and areal extent.

Two ways in, one result (:class:`GridSpec`):

  * manual: ``grid_spec_from_cells`` (origin, azimuth, nx, ny, dx, dy),
    ``grid_spec_from_rectangle`` (origin, azimuth, lengths) and
    ``grid_spec_from_extent`` (axis-aligned x/y limits);
  * polygon: ``grid_spec_from_polygon`` fits a rotated grid around a polygon
    and deactivates the columns outside it.

A corner-point grid is logically rectangular, so a polygon boundary can only
be honoured by deactivating columns (ACTNUM = 0). The staircase is the
geometric error of that choice and is reported as ``area_ratio``. Inactive
cells are no-flow in a simulator: an open aquifer boundary needs separate
treatment (stage 9), it does not come from this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity

from .frame import GridFrame

_LEN_TOL = 1e-6  # metres: absorbs floating-point noise when counting whole cells
_EDGE_TOL = 1e-6  # metres: a column centre this close to the boundary counts as inside


class BoundaryError(ValueError):
    """Bad polygon or inconsistent boundary definition."""


@dataclass(eq=False)
class GridSpec:
    """Areal definition of a grid: where it sits, its column edges, and which
    columns are active. ``x_edges``/``y_edges`` are LOCAL coordinates (they
    start at 0) so that graded spacing can replace uniform spacing later."""

    frame: GridFrame
    x_edges: np.ndarray
    y_edges: np.ndarray
    active_xy: np.ndarray  # (nx, ny) bool
    boundary: Polygon | None = None  # MAP coordinates, validated, CCW
    coverage_xy: np.ndarray | None = None  # fraction of each column inside boundary

    @property
    def nx(self) -> int:
        return len(self.x_edges) - 1

    @property
    def ny(self) -> int:
        return len(self.y_edges) - 1

    @property
    def column_area(self) -> np.ndarray:
        return np.outer(np.diff(self.x_edges), np.diff(self.y_edges))

    @property
    def active_area(self) -> float:
        return float(self.column_area[self.active_xy].sum())

    @property
    def area_ratio(self) -> float | None:
        """Active column area / boundary polygon area (None for manual specs).
        1.0 is a perfect fit; the deviation is the cost of the staircase."""
        if self.boundary is None:
            return None
        return self.active_area / self.boundary.area

    def extent_map(self) -> tuple[float, float, float, float]:
        """Map-coordinate bounding box (xmin, ymin, xmax, ymax) of the grid."""
        xs, ys = np.meshgrid(self.x_edges[[0, -1]], self.y_edges[[0, -1]])
        X, Y = self.frame.to_map(xs, ys)
        return float(X.min()), float(Y.min()), float(X.max()), float(Y.max())


# -- polygon validation ---------------------------------------------------------
def validate_polygon(xy) -> Polygon:
    """Clean and validate a polygon given as an (N, 2+) array of map coordinates.

    Removes consecutive duplicate vertices and the closing vertex, requires at
    least 3 distinct vertices and a valid (non self-intersecting) ring of
    positive area, and returns it with counter-clockwise orientation. Invalid
    rings are rejected, never silently repaired, because repairing changes the
    boundary the user drew.
    """
    if isinstance(xy, Polygon):
        poly = xy
    else:
        a = np.asarray(xy, dtype=float)
        if a.ndim != 2 or a.shape[1] < 2:
            raise BoundaryError("polygon must be an (N, 2) array of x, y coordinates")
        a = a[:, :2]
        if not np.isfinite(a).all():
            raise BoundaryError("polygon contains NaN or infinite coordinates")
        keep = np.ones(len(a), dtype=bool)
        keep[1:] = np.any(np.diff(a, axis=0) != 0.0, axis=1)
        a = a[keep]
        if len(a) > 1 and np.array_equal(a[0], a[-1]):
            a = a[:-1]
        if len(a) < 3:
            raise BoundaryError(f"polygon needs at least 3 distinct vertices, got {len(a)}")
        poly = Polygon(a)
    if not poly.is_valid:
        raise BoundaryError(f"invalid polygon: {explain_validity(poly)}")
    if poly.area <= 0.0:
        raise BoundaryError("polygon has zero area")
    return orient(poly, 1.0)


# -- orientation -------------------------------------------------------------------
def auto_azimuth(polygon: Polygon) -> float:
    """Azimuth (deg clockwise from north) of the local +y (J) axis that gives
    the minimum-area rotated bounding rectangle, folded into [-45, 45) so the
    grid axes stay as close to north/east as the polygon allows.

    This minimises wasted (inactive) columns. It is not necessarily the best
    choice for the geology: align to the dominant fault strike instead by
    passing an explicit azimuth."""
    # Work relative to the polygon's own corner: GEOS builds the rectangle by
    # intersecting lines in the given coordinates, and at UTM scale (7-digit
    # northings) that costs ~1e-5 degrees of azimuth, i.e. ~0.5 mm over 3 km,
    # enough to add a spurious extra row of cells to an exactly fitting grid.
    x0, y0 = polygon.bounds[0], polygon.bounds[1]
    shifted = shapely.transform(polygon, lambda a: a - np.array([x0, y0]))
    mrr = shapely.minimum_rotated_rectangle(shifted)
    if not isinstance(mrr, Polygon):
        return 0.0
    p = np.asarray(mrr.exterior.coords)[:4]
    e = p[1] - p[0]
    alpha = math.degrees(math.atan2(e[0], e[1]))  # edge azimuth from north
    return float(((alpha + 45.0) % 90.0) - 45.0)


# -- manual definitions ----------------------------------------------------------------
def _edges(n: int, d: float) -> np.ndarray:
    return d * np.arange(n + 1, dtype=float)


def _check_cells(nx: int, ny: int, dx: float, dy: float) -> None:
    if nx < 1 or ny < 1:
        raise BoundaryError(f"nx and ny must be >= 1, got {nx}, {ny}")
    if not (dx > 0 and dy > 0):
        raise BoundaryError(f"cell sizes must be positive, got {dx}, {dy}")


def grid_spec_from_cells(
    origin_x: float, origin_y: float, azimuth_deg: float, nx: int, ny: int, dx: float, dy: float
) -> GridSpec:
    """Fully manual definition: map position of the grid's local origin (its
    south-west corner for azimuth 0), azimuth of the J axis, counts and sizes."""
    _check_cells(nx, ny, dx, dy)
    return GridSpec(
        frame=GridFrame(float(origin_x), float(origin_y), float(azimuth_deg)),
        x_edges=_edges(nx, dx),
        y_edges=_edges(ny, dy),
        active_xy=np.ones((nx, ny), dtype=bool),
    )


def grid_spec_from_rectangle(
    origin_x: float,
    origin_y: float,
    azimuth_deg: float,
    length_x: float,
    length_y: float,
    dx: float,
    dy: float,
    fit: str = "expand",
) -> GridSpec:
    """Rectangle given by lengths along the grid I (x) and J (y) directions.

    ``fit="expand"`` keeps dx, dy and rounds the cell counts up, so the grid
    may extend slightly past the requested lengths. ``fit="adjust"`` keeps the
    requested lengths and shrinks the cell size to fit a whole number of cells.
    """
    if fit not in ("expand", "adjust"):
        raise BoundaryError("fit must be 'expand' or 'adjust'")
    if not (length_x > 0 and length_y > 0 and dx > 0 and dy > 0):
        raise BoundaryError("lengths and cell sizes must be positive")
    nx = max(1, math.ceil((length_x - _LEN_TOL) / dx))
    ny = max(1, math.ceil((length_y - _LEN_TOL) / dy))
    if fit == "adjust":
        dx, dy = length_x / nx, length_y / ny
    return grid_spec_from_cells(origin_x, origin_y, azimuth_deg, nx, ny, dx, dy)


def grid_spec_from_extent(
    x_min: float, x_max: float, y_min: float, y_max: float, dx: float, dy: float, fit: str = "expand"
) -> GridSpec:
    """Axis-aligned map extent (azimuth 0)."""
    if not (x_max > x_min and y_max > y_min):
        raise BoundaryError("extent must have x_max > x_min and y_max > y_min")
    return grid_spec_from_rectangle(x_min, y_min, 0.0, x_max - x_min, y_max - y_min, dx, dy, fit)


# -- polygon definition ------------------------------------------------------------------
def grid_spec_from_polygon(
    polygon,
    dx: float,
    dy: float,
    azimuth: float | str = "auto",
    margin: float = 0.0,
    inside: str = "center",
    min_fraction: float = 0.5,
) -> GridSpec:
    """Fit a grid around ``polygon`` (map coordinates) and deactivate the
    columns outside it.

    azimuth:      ``"auto"`` (minimum-area rotated rectangle) or degrees
                  clockwise from north of the J axis.
    margin:       extra distance (m) added on all four sides of the polygon's
                  bounding rectangle, e.g. to leave room for boundary cells.
    inside:       ``"center"``: a column is active if its centre is inside;
                  ``"fraction"``: active if at least ``min_fraction`` of its
                  area is inside (exact polygon/cell intersection areas).
    """
    if inside not in ("center", "fraction"):
        raise BoundaryError("inside must be 'center' or 'fraction'")
    if not (dx > 0 and dy > 0):
        raise BoundaryError(f"cell sizes must be positive, got {dx}, {dy}")
    if margin < 0:
        raise BoundaryError("margin must be >= 0")
    if inside == "fraction" and not (0.0 < min_fraction <= 1.0):
        raise BoundaryError("min_fraction must be in (0, 1]")

    poly = validate_polygon(polygon)
    if isinstance(azimuth, str):
        if azimuth != "auto":
            raise BoundaryError("azimuth must be 'auto' or a number of degrees")
        az = auto_azimuth(poly)
    else:
        az = float(azimuth)

    # Bounding rectangle in a frame rotated by az but anchored at the map origin,
    # then re-anchor the frame at its lower-left corner (minus margin).
    probe = GridFrame(0.0, 0.0, az)
    pl = _to_local(poly, probe)
    bx0, by0, bx1, by1 = pl.bounds
    lx0, ly0 = bx0 - margin, by0 - margin
    ox, oy = probe.to_map(lx0, ly0)
    frame = GridFrame(float(ox), float(oy), az)
    nx = max(1, math.ceil((bx1 - bx0 + 2 * margin - _LEN_TOL) / dx))
    ny = max(1, math.ceil((by1 - by0 + 2 * margin - _LEN_TOL) / dy))
    x_edges, y_edges = _edges(nx, dx), _edges(ny, dy)

    local_poly = _to_local(poly, frame)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    # Closed-boundary rule: a centre exactly on the polygon edge is inside. Without
    # the tiny buffer, whether such a centre is "inside" is decided by floating-point
    # noise, which happens whenever the grid aligns with a polygon edge and the
    # margin is half a cell. (Columns touching the boundary get exact areas below.)
    centre_poly = local_poly.buffer(_EDGE_TOL, join_style="mitre")
    centre_in = shapely.contains_xy(centre_poly, XC, YC)

    coverage = None
    if inside == "center":
        active = centre_in
    else:
        XE0, YE0 = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
        XE1, YE1 = np.meshgrid(x_edges[1:], y_edges[1:], indexing="ij")
        boxes = shapely.box(XE0.ravel(), YE0.ravel(), XE1.ravel(), YE1.ravel())
        coverage = centre_in.ravel().astype(float)  # interior -> 1, exterior -> 0
        shapely.prepare(local_poly)
        touches = shapely.intersects(boxes, local_poly.boundary)
        if touches.any():
            idx = np.flatnonzero(touches)
            inter = shapely.area(shapely.intersection(boxes[idx], local_poly))
            coverage[idx] = inter / shapely.area(boxes[idx])
        coverage = coverage.reshape(nx, ny)
        active = coverage >= min_fraction - 1e-9

    if not active.any():
        raise BoundaryError(
            "no grid columns fall inside the boundary; cell size is probably too large "
            "for the polygon, or the polygon coordinates are not in the model CRS"
        )
    return GridSpec(frame, x_edges, y_edges, np.asarray(active, dtype=bool), poly, coverage)


def _to_local(poly: Polygon, frame: GridFrame) -> Polygon:
    def fn(a: np.ndarray) -> np.ndarray:
        x, y = frame.to_local(a[:, 0], a[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(poly, fn)
