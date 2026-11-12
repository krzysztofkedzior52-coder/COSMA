"""Facies proportion maps from raster images (PNG/JPG/TIFF/GeoTIFF/screenshots).

Pipeline (every number is entered by the user; nothing is guessed):
  1 load  2 crop + mask  3 rotate/flip/scale  4 georeference (>= 3 control points, affine, residual tolerance)
  5 colours -> facies of the project's scheme  6 class -> fraction smoothing  7 checks  8 resample to the model grid.
Steps 2-3 are applied to the pixels first, so control points are picked on the processed image.
Colours are tied only to facies that already exist in the project's scheme; a facies that does not appear in the legend
must be declared absent explicitly (its fraction is then 0 wherever the map has data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NODATA = -1


class MapImportError(ValueError):
    """The raster, a setting or the legend is not usable; the message says what to change."""


# -- 1 load ------------------------------------------------------------------------------------------------------------
@dataclass
class Raster:
    rgb: np.ndarray                    # (h, w, 3) uint8
    valid: np.ndarray                  # (h, w) bool; False = transparent / masked / outside
    geo_affine: tuple | None = None    # (a, b, c, d, e, f): X = a*col + b*row + c, Y = d*col + e*row + f (pixel centres at +0.5)

    @property
    def shape(self):
        return self.rgb.shape[:2]


def load_raster(path) -> Raster:
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
    except Exception as exc:
        raise MapImportError(f"cannot read the image: {exc}") from None
    geo = _geotiff_affine(path) if im.format == "TIFF" else None
    if im.mode in ("I;16", "I", "F", "I;16B", "I;16L"):
        raise MapImportError("16-bit / floating-point rasters hold values, not colours; export a coloured 8-bit image")
    alpha = None
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        alpha = np.asarray(rgba)[..., 3]
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    valid = np.ones(rgb.shape[:2], bool) if alpha is None else alpha > 0
    return Raster(rgb, valid, geo)


def _geotiff_affine(path):
    """Affine from ModelPixelScale + ModelTiepoint tags if present (north-up rasters only); else None."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            tags = t.pages[0].tags
            sc, tp = tags.get(33550), tags.get(33922)
            tr = tags.get(34264)
            if tr is not None:
                m = [float(v) for v in tr.value]
                return (m[0], m[1], m[3], m[4], m[5], m[7])
            if sc is None or tp is None:
                return None
            sx, sy = float(sc.value[0]), float(sc.value[1])
            i, j, _, X, Y, _ = [float(v) for v in tp.value[:6]]
            return (sx, 0.0, X - i * sx, 0.0, -sy, Y + j * sy)
    except Exception:
        return None


# -- 2 crop + mask, 3 rotate / flip / scale ----------------------------------------------------------------------------
def crop(r: Raster, x0: int, y0: int, x1: int, y1: int) -> Raster:
    h, w = r.shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise MapImportError(f"crop box must satisfy 0 <= x0 < x1 <= {w} and 0 <= y0 < y1 <= {h}")
    return Raster(r.rgb[y0:y1, x0:x1].copy(), r.valid[y0:y1, x0:x1].copy(), None)


def mask_polygon(r: Raster, poly_px, keep: str) -> Raster:
    """Pixels outside (keep='inside') or inside (keep='outside') the polygon (pixel x, y vertices) become no data."""
    if keep not in ("inside", "outside"):
        raise MapImportError("keep must be 'inside' or 'outside'")
    p = np.asarray(poly_px, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise MapImportError("a mask polygon needs at least 3 (x, y) vertices")
    from matplotlib.path import Path
    h, w = r.shape
    yy, xx = np.mgrid[0:h, 0:w]
    inside = Path(p).contains_points(np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])).reshape(h, w)
    keepm = inside if keep == "inside" else ~inside
    return Raster(r.rgb, r.valid & keepm, r.geo_affine)


def flip(r: Raster, horizontal: bool = False, vertical: bool = False) -> Raster:
    s = (slice(None, None, -1) if vertical else slice(None), slice(None, None, -1) if horizontal else slice(None))
    return Raster(r.rgb[s].copy(), r.valid[s].copy(), None)


def rotate90(r: Raster, quarter_turns: int) -> Raster:
    k = int(quarter_turns) % 4
    return Raster(np.rot90(r.rgb, k).copy(), np.rot90(r.valid, k).copy(), None)


def scale(r: Raster, factor: float) -> Raster:
    """Nearest-neighbour resize: colours stay exact (no blended colours)."""
    if not (isinstance(factor, (int, float)) and math.isfinite(factor) and factor > 0):
        raise MapImportError("the scale factor must be a number > 0")
    h, w = r.shape
    nh, nw = max(1, round(h * factor)), max(1, round(w * factor))
    iy = np.minimum((np.arange(nh) + 0.5) / factor, h - 1).astype(int)
    ix = np.minimum((np.arange(nw) + 0.5) / factor, w - 1).astype(int)
    return Raster(r.rgb[np.ix_(iy, ix)].copy(), r.valid[np.ix_(iy, ix)].copy(), None)


# -- 4 georeference ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Georef:
    affine: tuple           # X = a*col + b*row + c ; Y = d*col + e*row + f   (col,row = pixel coordinates, centres at +0.5)
    residuals: tuple        # per control point, map units
    max_residual: float
    redundant: bool         # False with exactly 3 points: the fit is exact and says nothing about the points

    def to_map(self, col, row):
        a, b, c, d, e, f = self.affine
        return a * col + b * row + c, d * col + e * row + f

    def to_pixel(self, X, Y):
        a, b, c, d, e, f = self.affine
        det = a * e - b * d
        u, v = X - c, Y - f
        return (e * u - b * v) / det, (-d * u + a * v) / det


def georeference(points, tolerance) -> Georef:
    """points: (col, row, X, Y) in processed-image pixel coordinates and map coordinates. ``tolerance`` (map units) is
    the largest residual accepted; there is no default."""
    if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and tolerance >= 0):
        raise MapImportError("enter the residual tolerance (map units, >= 0)")
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 4 or len(P) < 3 or not np.isfinite(P).all():
        raise MapImportError("at least 3 control points (col, row, X, Y) are needed")
    px, mp = P[:, :2], P[:, 2:]
    c = px - px.mean(0)
    if np.linalg.matrix_rank(c, tol=1e-9 * max(1.0, np.abs(c).max())) < 2:
        raise MapImportError("the control points lie on a line in the image; spread them out")
    A = np.column_stack([px, np.ones(len(P))])
    cx, *_ = np.linalg.lstsq(A, mp[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(A, mp[:, 1], rcond=None)
    aff = (cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])
    if abs(aff[0] * aff[4] - aff[1] * aff[3]) < 1e-12:
        raise MapImportError("the control points give a degenerate (zero-area) transformation")
    res = np.hypot(A @ cx - mp[:, 0], A @ cy - mp[:, 1])
    g = Georef(tuple(float(v) for v in aff), tuple(float(v) for v in res), float(res.max()), len(P) > 3)
    if g.max_residual > tolerance:
        worst = int(np.argmax(res))
        raise MapImportError(f"control point {worst + 1} misses the fit by {g.max_residual:.6g} map units, more than the "
                             f"tolerance {tolerance:g}; correct that point or raise the tolerance")
    return g


# -- 5 colours -> facies -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Legend:
    entries: tuple              # ((r, g, b), facies_code) ...   several colours may map to one facies
    tolerance: float            # largest RGB distance (0-441) at which a pixel takes a legend colour
    absent: tuple = ()          # facies codes the user declares absent from the map (fraction 0 where the map has data)

    def __post_init__(self):
        if not (isinstance(self.tolerance, (int, float)) and math.isfinite(self.tolerance) and 0 <= self.tolerance <= 442):
            raise MapImportError("the colour tolerance must be a number between 0 and 441 (RGB distance)")
        if not self.entries:
            raise MapImportError("tie at least one colour to a facies")
        cols = [tuple(c) for c, _ in self.entries]
        for c in cols:
            if len(c) != 3 or any(not (isinstance(v, int) and 0 <= v <= 255) for v in c):
                raise MapImportError(f"colour {c}: three whole numbers 0-255 are needed")
        if len(set(cols)) != len(cols):
            raise MapImportError("a colour is tied to two facies")
        for i in range(len(cols)):
            for j in range(i + 1):
                if i != j and self.entries[i][1] != self.entries[j][1]:
                    d = math.dist(cols[i], cols[j])
                    if d <= 2 * self.tolerance:
                        raise MapImportError(f"colours {cols[i]} and {cols[j]} (different facies) are only {d:.1f} apart; the "
                                             f"tolerance {self.tolerance:g} would make them overlap. Lower the tolerance")

    def check_scheme(self, codes):
        known = set(codes)
        for _, f in self.entries:
            if f not in known:
                raise MapImportError(f"facies code {f} is not in the project's facies; define it first (Facies menu)")
        for f in self.absent:
            if f not in known:
                raise MapImportError(f"absent facies code {f} is not in the project's facies")
        used = {f for _, f in self.entries}
        both = used & set(self.absent)
        if both:
            raise MapImportError(f"facies {sorted(both)} have a colour and are also declared absent")
        missing = sorted(known - used - set(self.absent))
        if missing:
            raise MapImportError(f"facies {missing} have no colour on the legend and are not declared absent; tie a colour "
                                 "or declare them absent (their fraction would be 0 everywhere the map has data)")


def classify_colours(r: Raster, legend: Legend):
    """-> (classes (h, w) int, NODATA where unmatched or invalid; report dict)."""
    cols = np.array([c for c, _ in legend.entries], float)
    codes = np.array([f for _, f in legend.entries])
    d = np.sqrt(((r.rgb[:, :, None, :].astype(float) - cols[None, None]) ** 2).sum(-1))
    k = d.argmin(-1)
    near = np.take_along_axis(d, k[..., None], -1)[..., 0]
    hit = near <= legend.tolerance
    out = np.where(hit & r.valid, codes[k], NODATA)
    unmatched = r.valid & ~hit
    rep = dict(valid=int(r.valid.sum()), matched=int((out != NODATA).sum()), unmatched=int(unmatched.sum()),
               per_code={int(c): int((out == c).sum()) for c in set(codes.tolist())}, top_unmatched=[])
    if unmatched.any():
        q = (r.rgb[unmatched] // 16 * 16 + 8).astype(int)
        u, n = np.unique(q, axis=0, return_counts=True)
        o = np.argsort(-n)[:5]
        rep["top_unmatched"] = [(tuple(int(v) for v in u[i]), int(n[i])) for i in o]
    return out, rep


# -- 6 class -> fraction -----------------------------------------------------------------------------------------------
def suggest_sigma(pixel_size: float, cell_size: float) -> float:
    """Gaussian sigma in PIXELS: half a model column (cell_size) expressed in pixels (pixel_size, cell_size in map units),
    never below 1 pixel. Only a suggestion; the user enters the value."""
    if not (pixel_size > 0 and cell_size > 0):
        raise MapImportError("pixel and cell sizes must be > 0")
    return max(1.0, 0.5 * cell_size / pixel_size)


def to_fractions(classes: np.ndarray, codes, sigma: float):
    """Gaussian-smoothed one-hot classes, normalised by the smoothed valid mask so no-data does not pull fractions to 0.
    -> (h, w, nfac) float, NaN where the pixel has no data and no valid neighbour within the kernel."""
    from scipy.ndimage import gaussian_filter
    if not (isinstance(sigma, (int, float)) and math.isfinite(sigma) and sigma >= 0):
        raise MapImportError("enter the smoothing radius (sigma in pixels, >= 0; 0 = none)")
    valid = classes != NODATA
    if not valid.any():
        raise MapImportError("no pixel matched a legend colour")
    oh = np.stack([(classes == c) for c in codes], -1).astype(float)
    if sigma == 0:
        out = oh.copy()
        out[~valid] = np.nan
        return out
    den = gaussian_filter(valid.astype(float), sigma, mode="constant")
    num = np.stack([gaussian_filter(oh[..., i], sigma, mode="constant") for i in range(len(codes))], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den[..., None]
    out[den < 1e-6] = np.nan
    return out


# -- 7 checks ----------------------------------------------------------------------------------------------------------
def check_fractions(fr: np.ndarray, tol: float = 1e-6) -> dict:
    ok = ~np.isnan(fr[..., 0])
    v = fr[ok]
    if not len(v):
        raise MapImportError("the fraction maps hold no data")
    s = v.sum(-1)
    rep = dict(pixels=int(ok.sum()), min=float(v.min()), max=float(v.max()), max_sum_error=float(np.abs(s - 1).max()))
    if rep["min"] < -tol or rep["max"] > 1 + tol or rep["max_sum_error"] > tol:
        raise MapImportError(f"fractions fail the checks: range {rep['min']:.6g}..{rep['max']:.6g}, "
                             f"largest |sum-1| {rep['max_sum_error']:.3g}")
    return rep


# -- 8 resample to the model grid --------------------------------------------------------------------------------------
@dataclass
class Resampled:
    fractions: np.ndarray       # (nx, ny, nfac); NaN outside the map / no data / inactive columns
    codes: tuple
    active: int
    covered: int
    uncovered: np.ndarray       # (nx, ny) bool: active columns without map data

    @property
    def coverage(self):
        return self.covered / self.active if self.active else 0.0


def resample_to_grid(fr: np.ndarray, codes, geo: Georef, spec, min_coverage: float) -> Resampled:
    """Bilinear sampling at active column centres; a column is covered only if all four pixels around it have data.
    ``min_coverage`` (0-1, entered by the user) is the least share of active columns that must be covered."""
    if not (isinstance(min_coverage, (int, float)) and 0 < min_coverage <= 1):
        raise MapImportError("enter the minimum coverage of active columns (0-1]")
    xc = 0.5 * (spec.x_edges[:-1] + spec.x_edges[1:])
    yc = 0.5 * (spec.y_edges[:-1] + spec.y_edges[1:])
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    MX, MY = spec.frame.to_map(XX, YY)
    col, row = geo.to_pixel(MX, MY)
    h, w = fr.shape[:2]
    gx, gy = col - 0.5, row - 0.5                  # pixel-centre index space
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    tx, ty = gx - x0, gy - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    out = np.zeros(XX.shape + (fr.shape[2],))
    good = inside.copy()
    for dx, dy, wt in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
        v = fr[y0c + dy, x0c + dx]
        good &= ~np.isnan(v[..., 0])
        out += np.nan_to_num(v) * wt[..., None]
    act = np.asarray(spec.active_xy, bool)
    good &= act
    out[~good] = np.nan
    res = Resampled(out, tuple(codes), int(act.sum()), int(good.sum()), act & ~good)
    if res.coverage < min_coverage:
        raise MapImportError(f"the map covers {100 * res.coverage:.1f}% of the active columns, less than the required "
                             f"{100 * min_coverage:.1f}%")
    return res


# -- recipe: the whole pipeline as data (so it can be run headless, shown, stored) ---------------------------------------
OPS = ("crop", "flip", "rotate", "scale", "mask")


@dataclass(frozen=True)
class Recipe:
    interval: str
    ops: tuple                     # ("crop", x0, y0, x1, y1) | ("flip", horizontal, vertical) | ("rotate", quarter_turns)
    #                                | ("scale", factor) | ("mask", ((x, y), ...), "inside"|"outside")
    control_points: tuple          # ((col, row, X, Y), ...) on the processed image
    tolerance: float               # residual tolerance, map units
    legend: Legend
    sigma: float                   # Gaussian sigma, pixels
    min_coverage: float            # 0-1
    source: str = ""               # file name, informational only (the image itself is not stored)

    def __post_init__(self):
        if not self.interval:
            raise MapImportError("choose the interval this map belongs to")
        for op in self.ops:
            if not op or op[0] not in OPS:
                raise MapImportError(f"unknown processing step {op!r}")


@dataclass
class ProportionMap:
    interval: str
    codes: tuple
    fractions: np.ndarray          # (nx, ny, nfac), NaN where the map has no data / inactive
    recipe_state: dict
    report: str

    @property
    def shape(self):
        return self.fractions.shape[:2]


def apply_ops(r: Raster, ops) -> Raster:
    for op in ops:
        k = op[0]
        if k == "crop":
            r = crop(r, *[int(v) for v in op[1:5]])
        elif k == "flip":
            r = flip(r, bool(op[1]), bool(op[2]))
        elif k == "rotate":
            r = rotate90(r, int(op[1]))
        elif k == "scale":
            r = scale(r, float(op[1]))
        elif k == "mask":
            r = mask_polygon(r, op[1], op[2])
        else:
            raise MapImportError(f"unknown processing step {op!r}")
    return r


def run_recipe(r: Raster, recipe: Recipe, scheme, spec, *, with_images: bool = False):
    """Steps 2-8 on a loaded raster. ``scheme``: the project's facies (objects with .code and .name). Any failure raises
    MapImportError naming the step. Returns a ProportionMap (and, with_images, the intermediate arrays for display)."""
    names = {f.code: f.name for f in scheme}
    codes = tuple(f.code for f in scheme)
    if not codes:
        raise MapImportError("define the facies first (Facies menu): map colours can only be tied to existing facies")
    recipe.legend.check_scheme(codes)
    pr = apply_ops(r, recipe.ops)
    geo = georeference(recipe.control_points, recipe.tolerance)
    classes, crep = classify_colours(pr, recipe.legend)
    if crep["matched"] == 0:
        raise MapImportError("no pixel is within the colour tolerance of any legend colour")
    fr = to_fractions(classes, codes, recipe.sigma)
    frep = check_fractions(fr)
    res = resample_to_grid(fr, codes, geo, spec, recipe.min_coverage)
    sa = res.fractions[~np.isnan(res.fractions[..., 0])]
    mean = sa.mean(0)
    lines = [f"Proportion map for interval {recipe.interval!r} from {recipe.source or 'an image'}",
             f"  image after processing: {pr.shape[1]} x {pr.shape[0]} pixels; control points: {len(recipe.control_points)}, "
             f"largest residual {geo.max_residual:.4g} map units (tolerance {recipe.tolerance:g})"
             + ("" if geo.redundant else "; exactly 3 points: the fit is exact, residuals say nothing about the points"),
             f"  colours: {crep['matched']} of {crep['valid']} valid pixels matched (tolerance {recipe.legend.tolerance:g}); "
             f"{crep['unmatched']} unmatched = no data"]
    if crep["top_unmatched"]:
        lines.append("  most common unmatched colours (rounded): " + ", ".join(f"{c} x{n}" for c, n in crep["top_unmatched"]))
    lines.append("  pixels per facies: " + ", ".join(f"{names[c]} {crep['per_code'].get(c, 0)}" for c in codes))
    if recipe.legend.absent:
        lines.append("  declared absent from the map (fraction 0): " + ", ".join(names[c] for c in recipe.legend.absent))
    lines.append(f"  smoothing sigma {recipe.sigma:g} pixels; fractions in [{frep['min']:.3g}, {frep['max']:.3g}], "
                 f"largest |sum-1| {frep['max_sum_error']:.2g}")
    lines.append(f"  model columns: {res.covered} of {res.active} active columns covered ({100 * res.coverage:.1f}%; required "
                 f"{100 * recipe.min_coverage:.1f}%); mean fractions over covered columns: "
                 + ", ".join(f"{names[c]} {100 * v:.1f}%" for c, v in zip(codes, mean)))
    pm = ProportionMap(recipe.interval, codes, res.fractions, recipe_to_state(recipe), "\n".join(lines))
    if with_images:
        return pm, dict(classes=classes, fractions=fr, processed=pr, georef=geo)
    return pm


# -- state -------------------------------------------------------------------------------------------------------------
def recipe_to_state(r: Recipe) -> dict:
    return dict(interval=r.interval, ops=[list(o) if o[0] != "mask" else ["mask", [list(p) for p in o[1]], o[2]] for o in r.ops],
                control_points=[list(p) for p in r.control_points], tolerance=r.tolerance,
                legend=dict(entries=[[list(c), f] for c, f in r.legend.entries], tolerance=r.legend.tolerance,
                            absent=list(r.legend.absent)), sigma=r.sigma, min_coverage=r.min_coverage, source=r.source)


def recipe_from_state(d) -> Recipe:
    try:
        lg = d["legend"]
        ops = tuple(("mask", tuple(tuple(p) for p in o[1]), o[2]) if o[0] == "mask" else tuple(o) for o in d["ops"])
        return Recipe(d["interval"], ops, tuple(tuple(p) for p in d["control_points"]), d["tolerance"],
                      Legend(tuple((tuple(c), f) for c, f in lg["entries"]), lg["tolerance"], tuple(lg["absent"])),
                      d["sigma"], d["min_coverage"], d.get("source", ""))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, MapImportError):
            raise
        raise MapImportError(f"bad stored proportion-map recipe: {exc}") from None
