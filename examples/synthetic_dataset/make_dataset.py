"""Synthetic test dataset for COSMA: one faulted three-interval reservoir with every kind of input COSMA imports.

    python make_dataset.py OUTPUT_DIR

Everything is invented (UTM zone 16N coordinates, no real place or well). A hidden "truth" is built first
(structure with three faults, facies from a truncated latent Gaussian field with vertical and lateral proportion
trends, porosity/shale volume correlated with facies, permeability from porosity with scatter); the files are then
read off that truth the way real data would be: wells sample it along their paths, core plugs sample it at depth,
maps are averages over the interval, images are slices.  Seeded, so the output is reproducible.

Nothing here is a model of any real basin.  The numbers are plausible, not calibrated.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import fft as sfft
from scipy.ndimage import map_coordinates
from scipy.stats import norm

SEED = 20261113
X0, Y0, LX, LY = 599500.0, 4699500.0, 6000.0, 5000.0        # surface / truth extent (UTM 16N metres)
CRS = "EPSG:32616"
NX, NY, NK = 60, 50, 160                                      # truth grid: 100 m columns, 160 proportional layers
CELL = 100.0
INTERVALS = ("Seal", "Res_A", "Res_B")
FACIES = {1: "Sand", 2: "Silt", 3: "Shale"}
KB_DEFAULT = 242.0
Z_REF = 1150.0                                                # depth at which fault traces are defined (TVDSS)

# ---------------------------------------------------------------------------------------------- structure
FAULTS = {
    # name: (P0, P1) trace points at Z_REF, dip, which side is down-dropped (+1 right of direction, -1 left), throw (m)
    "F_Main": dict(p0=(600400.0, 4700700.0), p1=(604200.0, 4703900.0), dip=86.0, side=+1, throw=28.0, taper="mid"),
    "F_Minor": dict(p0=(602300.0, 4702300.0), p1=None, dip=87.0, side=+1, throw=14.0, taper="tip"),
    "F_East": dict(p0=(603600.0, 4700100.0), p1=None, dip=88.0, side=-1, throw=4.0, taper="mid"),
}


def _finish_faults():
    m = FAULTS["F_Main"]
    # F_Minor starts exactly on F_Main (a T-junction), runs ESE, 2200 m
    m0, m1 = np.array(m["p0"]), np.array(m["p1"])
    FAULTS["F_Minor"]["p0"] = tuple(m0 + 0.5 * (m1 - m0))
    a = math.radians(100.0)
    FAULTS["F_Minor"]["p1"] = tuple(np.array(FAULTS["F_Minor"]["p0"]) + 2200.0 * np.array([math.sin(a), math.cos(a)]))
    a = math.radians(10.0)
    FAULTS["F_East"]["p1"] = tuple(np.array(FAULTS["F_East"]["p0"]) + 1500.0 * np.array([math.sin(a), math.cos(a)]))
    for f in FAULTS.values():
        p0, p1 = np.array(f["p0"]), np.array(f["p1"])
        d = p1 - p0
        L = float(np.hypot(*d))
        u = d / L
        right = np.array([u[1], -u[0]])
        n = right if f["side"] > 0 else -right            # horizontal unit vector pointing to the down-dropped side
        f.update(L=L, u=u, n=n, p0=p0, p1=p1)


_finish_faults()


def _smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def fault_offset(x, y, z):
    """Depth offset (m, positive = deeper) the faults add at (x, y) for a surface at depth z."""
    out = np.zeros_like(np.asarray(x, float))
    for f in FAULTS.values():
        rel = np.stack([x - f["p0"][0], y - f["p0"][1]])
        t = (rel[0] * f["u"][0] + rel[1] * f["u"][1]) / f["L"]
        d = rel[0] * f["n"][0] + rel[1] * f["n"][1] - (z - Z_REF) / math.tan(math.radians(f["dip"]))
        tc = np.clip(t, 0, 1)
        if f["taper"] == "mid":
            g = (4 * tc * (1 - tc)) ** 0.5
        else:
            g = (1 - tc) ** 0.6 * _smooth((t + 0.03) / 0.03)
        g = np.where((t < 0) | (t > 1), 0.0, g)
        out = out + f["throw"] * g * _smooth((d / 12.0 + 1.0) / 2.0)
    return out


_RNG0 = np.random.default_rng(SEED)
_ROUGH = [(_RNG0.uniform(0.0004, 0.0016), _RNG0.uniform(0.0004, 0.0016), _RNG0.uniform(0, 6.28), a)
          for a in (1.8, 1.2, 0.9, 0.6)]


def _rough(x, y, scale=1.0):
    return scale * sum(a * np.sin(kx * (x - X0) + ky * (y - Y0) + ph) for kx, ky, ph, a in _ROUGH)


def trend_ne(x, y):
    """0 in the SW corner, 1 in the NE corner (position along azimuth 45)."""
    u = ((x - X0) * math.sin(math.radians(45)) + (y - Y0) * math.cos(math.radians(45)))
    lo, hi = 0.0, (LX + LY) * math.sin(math.radians(45))
    return np.clip((u - lo) / (hi - lo), 0, 1)


def surface_depths(x, y):
    """TVDSS (m, positive down) of Top Seal, Top Res_A, Top Res_B, base (Basement) at (x, y). Analytic, so
    wells and files agree."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = (x - X0) * math.sin(math.radians(135)) + (y - Y0) * math.cos(math.radians(135))
    dome = -9.0 * np.cos(2 * math.pi * (x - X0 - 800.0) / 5200.0) * np.cos(2 * math.pi * (y - Y0 - 600.0) / 6500.0)
    top = 1050.0 + 0.020 * u + dome + _rough(x, y)
    s = trend_ne(x, y)
    t_seal = 55.0 + 12.0 * s + _rough(x, y + 700, 0.35)
    t_a = 68.0 + 14.0 * np.sin(2 * math.pi * (x - X0) / 7000.0) + _rough(x + 400, y, 0.45)
    t_b = 88.0 - 15.0 * s + _rough(x, y - 500, 0.5)
    # the faults displace every horizon (each at its own depth, so a dipping trace shifts with depth)
    z0 = top + fault_offset(x, y, top)
    z1 = z0 + t_seal + (fault_offset(x, y, top + t_seal) - fault_offset(x, y, top))
    z2 = z1 + t_a + (fault_offset(x, y, top + t_seal + t_a) - fault_offset(x, y, top + t_seal))
    z3 = z2 + t_b + (fault_offset(x, y, top + t_seal + t_a + t_b) - fault_offset(x, y, top + t_seal + t_a))
    return np.stack([z0, z1, z2, z3])


# ---------------------------------------------------------------------------------------------- truth fields
def _fast(n):
    return sfft.next_fast_len(int(n), real=True)


def gauss_field(shape, sig_u, sig_v, sig_z, az_deg, rng):
    """Standardised Gaussian field, Gaussian covariance with practical ranges 3.46*sigma (sigma in cells),
    major axis at azimuth az (clockwise from north, i.e. from +y towards +x)."""
    nx, ny, nz = shape
    px, py, pz = _fast(2 * nx), _fast(2 * ny), _fast(2 * nz)
    w = rng.standard_normal((px, py, pz))
    F = sfft.rfftn(w)
    kx = 2 * np.pi * sfft.fftfreq(px)[:, None, None]
    ky = 2 * np.pi * sfft.fftfreq(py)[None, :, None]
    kz = 2 * np.pi * sfft.rfftfreq(pz)[None, None, :]
    a = math.radians(az_deg)
    ku = kx * math.sin(a) + ky * math.cos(a)
    kv = kx * math.cos(a) - ky * math.sin(a)
    amp = np.exp(-0.5 * ((sig_u * ku) ** 2 + (sig_v * kv) ** 2 + (sig_z * kz) ** 2))
    f = sfft.irfftn(F * amp, s=(px, py, pz))
    f = (f - f.mean()) / f.std()
    return f[:nx, :ny, :nz].astype(np.float32)


# latent-field geometry: practical ranges in metres (horizontal) and in proportional layers (vertical)
LAT = dict(long=dict(major=1800.0, minor=900.0, vert_m=10.0), short=dict(major=600.0, minor=400.0, vert_m=2.5),
           azimuth=40.0, long_weight=0.7)
MEAN_THICKNESS = {"Seal": 61.0, "Res_A": 68.0, "Res_B": 80.0}


def make_fields(rng):
    fields = {}
    for iv in INTERVALS:
        thk = MEAN_THICKNESS[iv]
        sz_long = LAT["long"]["vert_m"] / thk * NK / 3.46
        sz_short = LAT["short"]["vert_m"] / thk * NK / 3.46
        L, S = LAT["long"], LAT["short"]
        fields[iv] = dict(
            zl=gauss_field((NX, NY, NK), L["major"] / 3.46 / CELL, L["minor"] / 3.46 / CELL, sz_long, LAT["azimuth"], rng),
            zs=gauss_field((NX, NY, NK), S["major"] / 3.46 / CELL, S["minor"] / 3.46 / CELL, sz_short, LAT["azimuth"], rng),
            g1=gauss_field((NX, NY, NK), 500 / 3.46 / CELL, 350 / 3.46 / CELL, 1.5 / thk * NK / 3.46, LAT["azimuth"], rng),
            g2=gauss_field((NX, NY, NK), 700 / 3.46 / CELL, 450 / 3.46 / CELL, 2.0 / thk * NK / 3.46, LAT["azimuth"], rng))
        f = fields[iv]
        f["z"] = (math.sqrt(LAT["long_weight"]) * f["zl"] + math.sqrt(1 - LAT["long_weight"]) * f["zs"]).astype(np.float32)
        f["z"] = (f["z"] - f["z"].mean()) / f["z"].std()
    return fields


def proportions(iv, x, y, zr):
    """(p_sand, p_silt, p_shale) at map position and relative depth zr (0 = top of interval, 1 = base)."""
    s = trend_ne(x, y)
    if iv == "Seal":
        sand, shale = 0.02 + 0 * s, 0.86 - 0.08 * s
    elif iv == "Res_A":          # coarsening upward, more sand to the NE
        sand = (0.20 + 0.30 * (1 - zr)) * (0.7 + 0.6 * s)
        shale = (0.25 - 0.15 * (1 - zr)) * (1.2 - 0.4 * s)
    else:                        # Res_B: fining upward, sand-rich base, more sand to the NE
        sand = (0.25 + 0.45 * zr) * (0.8 + 0.4 * s)
        shale = (0.20 - 0.15 * zr) * (1.15 - 0.3 * s)
    tot = sand + shale
    k = np.where(tot > 0.95, 0.95 / tot, 1.0)
    sand, shale = sand * k, shale * k
    return sand, 1.0 - sand - shale, shale


def facies_from_z(iv, x, y, zr, z):
    ps, pi_, ph = proportions(iv, x, y, zr)
    t_shale = norm.ppf(np.clip(ph, 1e-4, 1 - 1e-4))
    t_silt = norm.ppf(np.clip(ph + pi_, 1e-4, 1 - 1e-4))
    return np.where(z < t_shale, 3, np.where(z < t_silt, 2, 1)).astype(np.int8)


MU_V = {1: (0.10, 0.06), 2: (0.40, 0.08), 3: (0.82, 0.08)}       # shale volume mean, sd per facies
MU_P = {1: (0.23, 0.025), 2: (0.12, 0.020), 3: (0.050, 0.010)}   # porosity mean, sd per facies
PERM_REL = {1: (-1.02, 16.0, 0.35), 2: (-1.68, 14.0, 0.40), 3: (-3.5, 12.0, 0.30)}   # log10 k = a + b phi + N(0, s)


def props_from(fac, z_tvdss, g1, g2, w1=None, w2=None):
    """Shale volume and porosity from facies and correlated noise. g1, g2 smooth fields; w1, w2 white noise (or None)."""
    e1 = 0.75 * g1 + (0.66 * w1 if w1 is not None else 0.0)
    e2 = 0.80 * g2 + (0.60 * w2 if w2 is not None else 0.0)
    mv = np.choose(fac - 1, [MU_V[1][0], MU_V[2][0], MU_V[3][0]])
    sv = np.choose(fac - 1, [MU_V[1][1], MU_V[2][1], MU_V[3][1]])
    mp = np.choose(fac - 1, [MU_P[1][0], MU_P[2][0], MU_P[3][0]]) * (1.0 - 0.0002 * (z_tvdss - Z_REF))
    sp = np.choose(fac - 1, [MU_P[1][1], MU_P[2][1], MU_P[3][1]])
    vsh = np.clip(mv + sv * e1, 0.0, 1.0)
    phi = np.clip(mp + sp * (-0.5 * e1 + math.sqrt(0.75) * e2), 0.005, 0.40)
    return vsh, phi


class Truth:
    def __init__(self, rng):
        self.rng = rng
        self.fields = make_fields(rng)

    def _interp(self, iv, name, x, y, zr):
        c = np.stack([(x - X0) / CELL - 0.5, (y - Y0) / CELL - 0.5, zr * NK - 0.5])
        return map_coordinates(self.fields[iv][name], c, order=1, mode="nearest")

    def sample(self, x, y, z, noise_rng):
        """Truth along a path: interval index (-1 above the seal, 3 below the base), facies, vsh, phi, perm-free."""
        d = surface_depths(x, y)
        n = x.size
        iv_idx = np.full(n, -1)
        for k in range(4):
            iv_idx[z >= d[k]] = k
        fac = np.zeros(n, np.int8)
        vsh = np.zeros(n)
        phi = np.zeros(n)
        zr_all = np.full(n, np.nan)
        w1, w2 = noise_rng.standard_normal(n), noise_rng.standard_normal(n)
        for k, iv in enumerate(INTERVALS):
            m = iv_idx == k
            if not m.any():
                continue
            zr = np.clip((z[m] - d[k][m]) / (d[k + 1][m] - d[k][m]), 0, 1)
            zr_all[m] = zr
            zf = self._interp(iv, "z", x[m], y[m], zr)
            f = facies_from_z(iv, x[m], y[m], zr, zf)
            v, p = props_from(f, z[m], self._interp(iv, "g1", x[m], y[m], zr), self._interp(iv, "g2", x[m], y[m], zr),
                              w1[m], w2[m])
            fac[m], vsh[m], phi[m] = f, v, p
        over, base = iv_idx == -1, iv_idx == 3
        fac[over], vsh[over], phi[over] = 0, np.clip(0.80 + 0.05 * w1[over], 0, 1), np.clip(0.06 + 0.01 * w2[over], 0.01, 0.3)
        fac[base], vsh[base], phi[base] = 0, np.clip(0.30 + 0.03 * w1[base], 0, 1), np.clip(0.02 + 0.004 * w2[base], 0.005, 0.1)
        return iv_idx, fac, vsh, phi

    def cubes(self):
        """Facies and noise-free porosity per interval on the truth grid, in proportional layers."""
        xc = X0 + (np.arange(NX) + 0.5) * CELL
        yc = Y0 + (np.arange(NY) + 0.5) * CELL
        XX, YY = np.meshgrid(xc, yc, indexing="ij")
        zr = (np.arange(NK) + 0.5) / NK
        d = surface_depths(XX, YY)
        out = {}
        for k, iv in enumerate(INTERVALS):
            f = self.fields[iv]
            X3, Y3, Z3 = XX[:, :, None], YY[:, :, None], zr[None, None, :]
            fac = facies_from_z(iv, X3, Y3, Z3, f["z"])
            zt = d[k][:, :, None] + Z3 * (d[k + 1] - d[k])[:, :, None]
            vsh, phi = props_from(fac, zt, f["g1"], f["g2"])
            out[iv] = dict(facies=fac, phi=phi.astype(np.float32), vsh=vsh.astype(np.float32),
                           thickness=(d[k + 1] - d[k]))
        return out


# ---------------------------------------------------------------------------------------------- wells
from cosma.wells.trajectory import Trajectory   # noqa: E402  (the dataset is built with COSMA's own path maths)

WELLS = {   # name: (x, y, kb, kind, azimuth, inclination, extra)
    "W-01": (602000.0, 4701000.0, 242.0, "V", 0, 0, "type well"),
    "W-02": (600800.0, 4701500.0, 240.5, "V", 0, 0, ""),
    "W-03": (601500.0, 4702800.0, 244.0, "V", 0, 0, ""),
    "W-04": (603000.0, 4703300.0, 243.2, "V", 0, 0, ""),
    "W-05": (604200.0, 4703000.0, 245.0, "D", 300.0, 22.0, "deviated towards F_Main, does not reach it"),
    "W-06": (603700.0, 4701600.0, 241.0, "D", 0.0, 28.0, "deviated towards the north"),
    "W-07": (603300.0, 4700600.0, 239.0, "D", 45.0, 28.0, "deviated"),
    "W-08": (604500.0, 4700500.0, 238.4, "V", 0, 0, ""),
    "W-09": (601100.0, 4700500.0, 242.7, "D", 135.0, 25.0, "deviated"),
    "W-10": (602600.0, 4701900.0, 243.9, "V", 0, 0, ""),
    "W-11": (602700.0, 4700300.0, 240.0, "V", 0, 0, ""),
    "W-12": (600600.0, 4703200.0, 246.0, "V", 0, 0, "shallow: stops halfway through Res_B"),
    "W-13": (604600.0, 4701700.0, 241.8, "V", 0, 0, "shallow: stops inside Res_A"),
    "W-14": (602150.0, 4701250.0, 242.3, "V", 0, 0, "twin of W-01, about 290 m away"),
    "W-15": (603650.0, 4702150.0, 244.5, "D", 200.0, 20.0, "deviated; tops use other pick names"),
    "W-16": (601400.0, 4702250.0, 245.2, "V", 0, 0, "infill near W-03; tops use other pick names"),
}
FACIES_CURVE_WELLS = ("W-01", "W-02", "W-03", "W-04")
CORE_WELLS = ("W-01", "W-02", "W-04", "W-08", "W-10")
CORE_TVD_WELL = "W-14"
LATE_TOPS_WELLS = ("W-15", "W-16")
LATE_NAMES = {"Seal": "Caprock", "Res_A": "Upper_Res", "Res_B": "Lower_Res", "Basement": "Crystalline"}
GAPS = {"W-03": (("Res_A", 25.0, 28.0),), "W-08": (("Res_B", 40.0, 43.0), ("Seal", 10.0, 12.0))}   # bad-hole zones: PHIT/RHOB/NPHI missing


def build_survey(kind, az, inc):
    """MD / inclination / azimuth stations. Vertical: two stations. Deviated: vertical to 300 m, build 1.5 deg per
    25 m up to the target inclination, hold."""
    if kind == "V":
        return np.array([0.0, 2500.0]), np.zeros(2), np.zeros(2)
    md = [0.0, 300.0]
    incl = [0.0, 0.0]
    while incl[-1] < inc - 1e-9:
        md.append(md[-1] + 25.0)
        incl.append(min(inc, incl[-1] + 1.5))
    md += [md[-1] + 100.0, 3200.0]
    incl += [inc, inc]
    return np.array(md), np.array(incl), np.full(len(md), float(az))


def place(name):
    x, y, kb, kind, az, inc, _ = WELLS[name]
    md, incl, azim = build_survey(kind, az, inc)
    return Trajectory.from_survey(md, incl, azim, x0=x, y0=y, kb=kb, azimuth_reference="grid"), (md, incl, azim)


def first_crossing(traj, surface_index, step=0.1):
    lo, hi = traj.md_range
    md = np.arange(lo, hi, step)
    x, y, z = traj.at_md(md)
    s = z - surface_depths(x, y)[surface_index]
    idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    if idx.size == 0:
        return None
    i = idx[0]
    return float(md[i] + step * (-s[i]) / (s[i + 1] - s[i]))


def fault_crossings_in_reservoir(traj):
    """Faults whose traces the path crosses between Top Seal and the base (a well that cuts section out or repeats it)."""
    lo, hi = traj.md_range
    md = np.arange(lo, hi, 0.5)
    x, y, z = traj.at_md(md)
    d = surface_depths(x, y)
    inside = (z >= d[0]) & (z <= d[3])
    out = []
    for nm, f in FAULTS.items():
        rel = np.stack([x - f["p0"][0], y - f["p0"][1]])
        t = (rel[0] * f["u"][0] + rel[1] * f["u"][1]) / f["L"]
        s = rel[0] * f["n"][0] + rel[1] * f["n"][1] - (z - Z_REF) / math.tan(math.radians(f["dip"]))
        cross = (np.sign(s[:-1]) != np.sign(s[1:])) & inside[:-1] & (t[:-1] > 0) & (t[:-1] < 1)
        if cross.any():
            out.append(nm)
    return out


# ---------------------------------------------------------------------------------------------- writers
def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(r if isinstance(r, str) else str(x) for x in r) + "\n")


def f6(v):
    return f"{v:.3f}"


def write_surface(path, X, Y, vals, fmt, name):
    import xtgeo
    xs, ys = np.unique(X[:, 0]), np.unique(Y[0, :])
    s = xtgeo.RegularSurface(ncol=xs.size, nrow=ys.size, xinc=float(xs[1] - xs[0]), yinc=float(ys[1] - ys[0]),
                             xori=float(xs[0]), yori=float(ys[0]), rotation=0.0, values=np.ma.masked_invalid(vals), name=name)
    s.to_file(str(path), fformat=fmt)


def grid_nodes(step):
    xs = np.arange(X0, X0 + LX + 1e-6, step)
    ys = np.arange(Y0, Y0 + LY + 1e-6, step)
    return np.meshgrid(xs, ys, indexing="ij")


def write_las(path, name, md, curves, units, x, y, kb, td):
    import lasio
    las = lasio.LASFile()
    las.version.VERS.value = 2.0
    las.well.WELL.value = name
    las.well.NULL.value = -999.25
    las.well.STRT.value, las.well.STOP.value, las.well.STEP.value = float(md[0]), float(md[-1]), 0.25
    las.well.STRT.unit = las.well.STOP.unit = las.well.STEP.unit = "M"
    las.params["EKB"] = lasio.HeaderItem("EKB", unit="M", value=kb, descr="KB elevation above MSL (synthetic)")
    las.append_curve("DEPT", md, unit="M", descr="Measured depth")
    for k, v in curves.items():
        las.append_curve(k, np.where(np.isfinite(v), v, -999.25), unit=units[k], descr=k)
    with open(path, "w") as fh:
        las.write(fh, version=2.0, fmt="%.4f", len_numeric_field=-1)


README = """# COSMA synthetic dataset

One invented, faulted, three-interval saline-aquifer storage target, with every kind of input COSMA imports. Nothing here
is real: UTM zone 16N (EPSG:32616) coordinates, no real place, well or company. It is a **test fixture**, not a model of any basin.

How it was made: a hidden truth first (structure with three faults; facies from a truncated latent Gaussian field with vertical
and lateral proportion trends; shale volume and porosity tied to facies and correlated with each other; log10 permeability linear
in porosity per facies, plus scatter). The files are then read off that truth the way real data would be: wells sample it along their
paths, core plugs at depth, maps are interval averages, images are slices. `make_dataset.py` rebuilds everything (seed {seed});
`verify_with_cosma.py DATASET_DIR --pipeline` imports every file the way the dialogs do and goes on to a grid, facies, data checks
and the image imports. Both ran clean on the version of COSMA this was delivered with.

## Stratigraphy and structure

Intervals top to bottom: **Seal** (shale-dominated, ~61 m), **Res_A** (~71 m, coarsens upward, more sand to the NE),
**Res_B** (~81 m, fines upward, sand-rich base). Picks are named `Seal`, `Res_A`, `Res_B`, `Basement` (the base).
Facies codes: 1 Sand, 2 Silt, 3 Shale. Top Seal runs from about 980 to 1155 m TVDSS (mean 1071 m), dipping ~1.1 deg to the SE with a gentle dome; the base is at 1190-1350 m.

| Fault | Trace at 1150 m | Dip | Down-thrown | Max throw in the surfaces | Note |
|---|---|---|---|---|---|
| F_Main | {fm} | 86 | SE | 28 m (COSMA measures up to 25.6 m median) | 5 km, NE-SW |
| F_Minor | starts {fn} | 87 | S | 14 m, tapering to the tip (COSMA: 9.2 m) | sticks end 49 m short of F_Main: a T-junction, F_Main older |
| F_East | {fe} | 88 | W | 4 m | negligible throw on purpose: COSMA leaves it out of the geometry |

The surfaces carry the steps; their position follows each fault's dipping trace at that horizon's depth. Properties were generated
without regard to the faults (the assumption COSMA documents: properties fixed before faulting).

## Folders, and what to choose in each dialog

**1_boundary** - the same 17.12 km2 polygon in five formats. `boundary_utm16n.csv`, `boundary_irap.pol`: model CRS, no choice needed.
`boundary_wgs84.geojson`, `.kml`: lon/lat, set **target CRS = EPSG:32616**. `boundary_utm16n.shp` (+ .shx .dbf .prj): target CRS EPSG:32616.
They agree to 3e-7 in area. Import one, or all (the last stays the model boundary).

**2_type_well** - New project with a type well: W-01 header (x/y m, z m), deviation (z m, azimuth reference **grid**), tops (z m,
depth type **MD**). The chart is built from its tops: Seal, Res_A, Res_B, base Basement.

**3_surfaces** - four depth grids on 25 m nodes, 241 x 201, in three formats: `Top_Seal.irap`, `Top_Res_A.irap` (IRAP ASCII),
`Top_Res_B.zmap` (ZMAP+), `Top_Basement.gri` (IRAP binary). x/y m, z m, depth positive down. Attach: Top_Seal = Top of Seal,
Top_Res_A = Base of Seal / Top of Res_A, Top_Res_B = Base of Res_A / Top of Res_B, Top_Basement = base.

**4_wells** - 16 wells (table below). `well_headers_all.csv`, `deviation_all.csv`, `tops_14_wells.csv`, `logs/W-xx.las`,
`core_5_wells_md_percent.csv`, `core_W-14_tvd_fraction.csv`, `tops_W-15_W-16_other_names.csv`. They include W-01, so importing them after
the type well shows the "identical data already there" collision: choose **skip**.
* Tops: `Marker_X` is an extra pick that is not in the chart: map it to **ignore**. `tops_W-15_W-16_other_names.csv` uses `Caprock`,
  `Upper_Res`, `Lower_Res`, `Crystalline`: map them to Seal, Res_A, Res_B and the base (they are then remembered as aliases).
* LAS: depth unit m (the files say M). Curves `GR` (GAPI), `RHOB` (G/CC), `NPHI`, `PHIT`, `VSHALE` (V/V); `FACIES` (codes 1-3) only in
  W-01..W-04. Null -999.25. Bad-hole gaps (PHIT, RHOB, NPHI missing) in W-03 (Res_A) and W-08 (Seal, Res_B).
* Core: `core_5_wells_md_percent.csv`: z m, depth **MD**, porosity **percent**; columns `poro` (%), `perm` (mD), `grain_dens` (kept, ignored)
  for W-01, W-02, W-04, W-08, W-10. `core_W-14_tvd_fraction.csv`: depth **TVD below KB**, porosity **fraction**, `poro`, `perm`.
  Few shale plugs (as in real core), ~25% of plugs dropped as broken core.
* Datum: `kb` is the KB elevation above mean sea level (238-246 m); depths in the surfaces are TVDSS.

**5_fault_sticks** - `fault_sticks_F_Main_F_Minor_m.csv`: header `fault,stick,x,y,z`; x/y m, z m, **depth (positive down)**.
`fault_F_East_headerless_feet.txt`: headerless x y z, sticks separated by `999 999 999`, **in international feet** for x, y and z
(`ft_intl`), fault name F_East. Stick points carry +-3 m (xy) and +-2 m (z) noise. Then: trace depth 1150 m for each fault, reach 150 m,
answer the one interaction (F_Main older, which is what COSMA suggests), choose transmissibility.

**6_property_maps** - IRAP maps on 100 m nodes: `PHIT_Res_A_fraction` (Res_A, fraction), `PHIT_Res_B_percent` (Res_B, **percent**),
`SAND_FRACTION_Res_B` (fraction), `PERM_geomean_Res_B_mD` (mD). Interval averages of the truth, noise-free.

**7_variograms** - four files: `PHIT_normal_score_Res_A.json`, `VSHALE_normal_score_Res_B.json` (JSON, carry their own variable and
transform), `FACIES_1_sand_indicator_Res_B.json` (variable `FACIES[1]`, transform indicator), `PHIT_normal_score_Res_B.var` (GSLIB:
set variable PHIT, transform normal_score, range unit m). **These are the models of the latent Gaussian field behind the facies
(azimuth 40, ranges 1800/900 m and 600/400 m, vertical 10 and 2.5 m, Gaussian), scaled for the file's variable. They are not fitted to
the wells and are not the true variograms of PHIT or of the facies indicators. To test fitting, compute your own from the wells.**

**8_proportion_map_images** - `facies_map_Res_B_screenshot.png` (800 x 800, a screenshot look: title and legend below the map,
registration crosses outside the map frame, white = no data). Crop **0,0 - 800,600**, no rotate/flip/scale, control points from
`facies_map_Res_B_control_points.csv` (pixel centres at +0.5, 6 points), residual tolerance a few m. Legend colours (RGB): Sand
255 215 0, Silt 205 133 63, Shale 105 105 105, tolerance ~10. `facies_map_Res_A_geotiff.tif` is a georeferenced RGBA GeoTIFF (20 m pixels,
transparent outside the polygon): the affine comes from the file. Each image is one slice of the truth, picked so its sand share
({sand_b:.0%} Res_B, {sand_a:.0%} Res_A) is close to the interval's ({tsand_b:.0%}, {tsand_a:.0%}).

**9_reference_truth** - not for import. `typed_vpc_<interval>.csv`: 10-bin relative-position proportions from the truth (for the
"My curve" option), `truth_cubes.npz` (facies and noise-free porosity per interval, 100 m x 160 proportional layers), `truth_summary.json`.

## The wells

| Well | x | y | Type | TD (m MD) | Note |
|---|---|---|---|---|---|
{wells}

## What a good run should recover (the truth)

* Facies proportions inside the polygon (area-weighted over columns and layers): {props}.
* Facies from logs with the rules `Sand: VSHALE < 0.25; Silt: >= 0.25 and < 0.60; Shale: >= 0.60` match the truth facies in {acc:.1%} of
  reservoir log samples (sand {a11:.1%}, silt {a22:.1%}, shale {a33:.1%}). That is optimistic against real logs.
* Core relations used: log10 k = a + b*phi + N(0, s): Sand (-1.02, 16, 0.35), Silt (-1.68, 14, 0.40), Shale (-3.5, 12, 0.30), k in mD.
  The data-check regression on the imported core gives steeper slopes (about 17-21 for sand and silt) and scatter 0.41-0.45: it fits cell-scale
  porosity and logs-based facies, with misclassification and depth tolerance mixed in, so it is not expected to return the generating numbers.
* Porosity by facies (mean, sd): Sand 0.23 (0.025), Silt 0.12 (0.020), Shale 0.05 (0.010), with a small depth trend; shale volume:
  0.10 (0.06), 0.40 (0.08), 0.82 (0.08). Within a facies PHIT and VSHALE are negatively correlated (about -0.5 before white noise).

## Where this dataset is easy or unrepresentative

* The facies truth is a truncated Gaussian field: TGS matches the generating process, SIS does not, so a SIS-versus-truth comparison is not fair to SIS.
* 16 wells in 17 km2 is far denser than a CO2 storage scoping case usually has (often 1-5 wells); delete wells to see what thin data does.
* Log depths agree with picks and core exactly (no depth shifts), no tool-specific artefacts beyond white noise and the gaps above.
* The surface steps and the fault sticks agree by construction. Real interpretations rarely do.
* Faults are near-vertical (86-88 deg), the only geometry COSMA v1 handles. One well (W-06) cuts F_Minor inside the Seal; COSMA lists it in the warnings.
* Two wells (W-12, W-13) stop short, so Res_B / the base have no pick there: COSMA skips them for those intervals and says so.
* Control-point residuals in the PNG are ~0 because the coordinates were computed, not read off a real map.
"""


def write_readme(out, info):
    w = info["wells"]
    rows = "\n".join(f"| {n} | {v['x']:.0f} | {v['y']:.0f} | {v['kind']} | {v['td_md']:.0f} | "
                     f"{(v['note'] + '; ' if v['note'] else '')}{'cuts ' + ', '.join(v['crosses_faults_inside_reservoir']) + ' between top Seal and the base; ' if v['crosses_faults_inside_reservoir'] else ''}"
                     f"{'facies curve; ' if n in FACIES_CURVE_WELLS else ''}{'core (MD); ' if n in CORE_WELLS else ''}"
                     f"{'core (TVD); ' if n == CORE_TVD_WELL else ''}".rstrip("; ") + " |" for n, v in w.items())
    fp = info["facies_proportions_truth_inside_polygon"]
    props = "; ".join(f"{iv}: " + ", ".join(f"{k} {v:.0%}" for k, v in d.items()) for iv, d in fp.items())
    fr = info["facies_rules_vs_truth"]
    c = fr["confusion_row_fractions"]
    sl = info["image_slices"]
    fa = info["faults"]

    def pt(p):
        return f"({p[0]:.0f}, {p[1]:.0f}) to ({p[2]:.0f}, {p[3]:.0f})"
    (Path(out) / "README.md").write_text(README.format(
        seed=SEED, wells=rows, props=props, acc=fr["fraction_correct"], a11=c["truth Sand -> rule Sand"],
        a22=c["truth Silt -> rule Silt"], a33=c["truth Shale -> rule Shale"],
        fm=pt(fa["F_Main"]["trace_p0"] + fa["F_Main"]["trace_p1"]), fn=f"({fa['F_Minor']['trace_p0'][0]:.0f}, {fa['F_Minor']['trace_p0'][1]:.0f})",
        fe=pt(fa["F_East"]["trace_p0"] + fa["F_East"]["trace_p1"]),
        sand_b=sl["Res_B"]["sand_share_in_image"], sand_a=sl["Res_A"]["sand_share_in_image"],
        tsand_b=sl["Res_B"]["sand_share_interval"], tsand_a=sl["Res_A"]["sand_share_interval"]), encoding="utf-8")


def main(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    dirs = {k: out / k for k in ("1_boundary", "2_type_well", "3_surfaces", "4_wells", "4_wells/logs", "5_fault_sticks",
                                 "6_property_maps", "7_variograms", "8_proportion_map_images", "9_reference_truth")}
    for d in dirs.values():
        d.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    truth = Truth(rng)
    info = dict(seed=SEED, crs=CRS, faults={}, wells={}, intervals=list(INTERVALS))

    # ------------------------------------------------------------ 1 boundary polygon (model CRS, then other formats)
    poly = [(600200, 4700500), (602800, 4700050), (604900, 4700300), (605000, 4702300), (604700, 4703900),
            (602700, 4703950), (600400, 4703600), (600000, 4702000)]
    global POLY
    POLY = shapely.Polygon(poly)
    b = dirs["1_boundary"]
    write_csv(b / "boundary_utm16n.csv", ["x", "y"], [(f"{x:.1f}", f"{y:.1f}") for x, y in poly + poly[:1]])
    (b / "boundary_irap.pol").write_text("".join(f"{x:.1f} {y:.1f} 0.0\n" for x, y in poly + poly[:1]) + "999 999 999\n")
    from pyproj import Transformer
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    ll = [tr.transform(x, y) for x, y in poly + poly[:1]]
    (b / "boundary_wgs84.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=[dict(
        type="Feature", properties=dict(name="Model_AOI"), geometry=dict(type="Polygon", coordinates=[[list(p) for p in ll]]))])))
    (b / "boundary_wgs84.kml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
        "<name>Model_AOI</name><Polygon><outerBoundaryIs><LinearRing><coordinates>"
        + " ".join(f"{lon:.7f},{lat:.7f},0" for lon, lat in ll) + "</coordinates></LinearRing></outerBoundaryIs></Polygon>"
        "</Placemark></Document></kml>\n")
    import shapefile
    from pyproj import CRS as PCRS
    with shapefile.Writer(str(b / "boundary_utm16n"), shapeType=shapefile.POLYGON) as w:
        w.field("NAME", "C", 30)
        w.poly([[list(p) for p in reversed(poly + poly[:1])]])      # shapefile outer rings are clockwise
        w.record("Model_AOI")
    (b / "boundary_utm16n.prj").write_text(PCRS.from_user_input(CRS).to_wkt(version="WKT1_ESRI"))
    info["boundary"] = dict(vertices=poly, note="same polygon in 5 formats; text formats are in the model CRS, "
                                                "GeoJSON/KML are WGS84 lon/lat (target_crs EPSG:32616 required)")

    # ------------------------------------------------------------ 3 surfaces (25 m nodes)
    X, Y = grid_nodes(25.0)
    D = surface_depths(X, Y)
    s3 = dirs["3_surfaces"]
    names = ["Top_Seal", "Top_Res_A", "Top_Res_B", "Top_Basement"]
    write_surface(s3 / "Top_Seal.irap", X, Y, D[0], "irap_ascii", "Top_Seal")
    write_surface(s3 / "Top_Res_A.irap", X, Y, D[1], "irap_ascii", "Top_Res_A")
    write_surface(s3 / "Top_Res_B.zmap", X, Y, D[2], "zmap_ascii", "Top_Res_B")
    write_surface(s3 / "Top_Basement.gri", X, Y, D[3], "irap_binary", "Top_Basement")
    info["surfaces"] = {n: dict(mean_depth=float(np.mean(D[i])), min=float(D[i].min()), max=float(D[i].max()))
                        for i, n in enumerate(names)}

    # ------------------------------------------------------------ wells: paths, tops, logs, core
    headers, devs, tops_main, tops_late, core_rows, core_tvd_rows = [], [], [], [], [], []
    well_data = {}
    for name, (x, y, kb, kind, az, inc, note) in WELLS.items():
        traj, (smd, sinc, saz) = place(name)
        picks = {}
        for k, pn in enumerate(("Seal", "Res_A", "Res_B", "Basement")):
            md = first_crossing(traj, k)
            if md is not None:
                picks[pn] = md
        # TD: 15 m below the base (vertical-equivalent), or partial for the two shallow wells
        if name == "W-12":
            td = picks["Res_B"] + 55.0
        elif name == "W-13":
            td = picks["Res_A"] + 40.0
        else:
            td = picks["Basement"] + 15.0
        td = float(np.ceil(td / 5.0) * 5.0)
        picks = {k: v for k, v in picks.items() if v <= td}
        keep = smd <= td
        sm, si, sa = smd[keep], sinc[keep], saz[keep]
        if sm[-1] < td:
            sm, si, sa = np.append(sm, td), np.append(si, si[-1]), np.append(sa, sa[-1])
        traj = Trajectory.from_survey(sm, si, sa, x0=x, y0=y, kb=kb, azimuth_reference="grid")
        well_data[name] = dict(traj=traj, picks=picks, td=td, survey=(sm, si, sa))
        headers.append((name, f"{x:.1f}", f"{y:.1f}", f"{kb:.2f}", f"{td:.1f}"))
        for m, i_, a_ in zip(sm, si, sa):
            devs.append((name, f"{m:.2f}", f"{i_:.2f}", f"{a_:.2f}"))
        late = name in LATE_TOPS_WELLS
        for pn, md in picks.items():
            (tops_late if late else tops_main).append((name, LATE_NAMES[pn] if late else pn, f"{md:.2f}"))
        if name not in LATE_TOPS_WELLS and name != "W-01":            # the type well's picks define the chart: no extra marker
            tops_main.append((name, "Marker_X", f"{picks['Seal'] + 0.6 * (picks['Res_A'] - picks['Seal']):.2f}"))
        info["wells"][name] = dict(x=x, y=y, kb=kb, td_md=td, kind="deviated" if kind == "D" else "vertical",
                                   note=note, picks_md={k: round(v, 2) for k, v in picks.items()},
                                   crosses_faults_inside_reservoir=fault_crossings_in_reservoir(traj))

    # logs
    nrng = np.random.default_rng(SEED + 1)
    classification = []
    for name, wd in well_data.items():
        traj, picks, td = wd["traj"], wd["picks"], wd["td"]
        z_top = float(np.min(surface_depths(np.array([WELLS[name][0]]), np.array([WELLS[name][1]]))[0]))
        md_top = traj.md_at_tvdss(z_top - 30.0)[0] if np.all(np.diff(traj.tvdss) > 0) else 0.0
        start = float(np.floor(md_top / 5.0) * 5.0) + 0.0
        md = np.arange(start, td + 1e-9, 0.25)
        x, y, z = traj.at_md(md)
        iv, fac, vsh, phi = truth.sample(x, y, z, nrng)
        gr = 15.0 + 125.0 * vsh + nrng.normal(0, 4.0, md.size)
        rhob = 2.65 * (1 - phi) + 1.04 * phi + nrng.normal(0, 0.015, md.size)
        nphi = phi + 0.15 * vsh + nrng.normal(0, 0.01, md.size)
        phit = np.clip((2.65 - rhob) / (2.65 - 1.04), 0, 0.6)
        vshale = np.clip((gr - 15.0) / 125.0, 0.0, 1.0)
        for gz_iv, a, bb in GAPS.get(name, ()):
            if gz_iv in picks and (("Res_A", "Res_B", "Basement")[("Seal", "Res_A", "Res_B").index(gz_iv)]) in picks:
                s0 = picks[gz_iv]
                g = (md >= s0 + a) & (md <= s0 + bb)
                for arr in (phit, rhob, nphi):
                    arr[g] = np.nan
        curves = dict(GR=gr, RHOB=rhob, NPHI=nphi, PHIT=phit, VSHALE=vshale)
        units = dict(GR="GAPI", RHOB="G/CC", NPHI="V/V", PHIT="V/V", VSHALE="V/V", FACIES="")
        if name in FACIES_CURVE_WELLS:
            fc = fac.astype(float)
            fc[(iv < 0) | (iv > 2)] = np.nan
            curves["FACIES"] = fc
        write_las(dirs["4_wells"] / "logs" / f"{name}.las", name, md, curves, units, x[0], y[0], WELLS[name][2], td)
        wd["log"] = dict(md=md, x=x, y=y, z=z, iv=iv, fac=fac, vsh=vsh, phi=phi, vshale=vshale)
        res = (iv >= 0) & (iv <= 2)
        rule = np.where(vshale < 0.25, 1, np.where(vshale < 0.60, 2, 3))
        classification.append((fac[res], rule[res]))

    # core
    crng = np.random.default_rng(SEED + 2)

    def core_for(name):
        wd = well_data[name]
        traj, picks = wd["traj"], wd["picks"]
        mds = []
        for a, bpick, lo, hi in (("Res_A", "Res_B", 5.0, 45.0), ("Res_B", "Basement", 5.0, 50.0)):
            if a in picks:
                top = picks[a] + lo
                end = min(picks[a] + hi, picks.get(bpick, 1e9) - 1.0, wd["td"] - 1.0)
                mds.append(np.arange(top, end, 0.5))
        md = np.concatenate(mds)
        md = md[crng.random(md.size) > 0.25]                            # broken / unusable core
        x, y, z = traj.at_md(md)
        iv, fac, vsh, phi = truth.sample(x, y, z, crng)
        a, bb, s = (np.array([PERM_REL[f][i] for f in fac]) for i in range(3))
        logk = a + bb * phi + crng.normal(0, 1, md.size) * s
        keep = (fac != 3) | (crng.random(md.size) < 0.3)                # few shale plugs
        phi_core = np.clip(phi + crng.normal(0, 0.005, md.size), 0.001, 0.5)
        md = md + crng.normal(0, 0.05, md.size)
        gd = 2.65 + crng.normal(0, 0.01, md.size)
        return md[keep], phi_core[keep], 10 ** logk[keep], gd[keep], traj

    for name in CORE_WELLS:
        md, phi, k, gd, _ = core_for(name)
        for m, p, kk, g in zip(md, phi, k, gd):
            core_rows.append((name, f"{m:.2f}", f"{p * 100:.2f}", f"{kk:.4g}", f"{g:.3f}"))
    md, phi, k, gd, traj = core_for(CORE_TVD_WELL)
    for m, p, kk in zip(md, phi, k):
        tvd = float(traj.tvdss_at_md(np.array([m]))[0]) + WELLS[CORE_TVD_WELL][2]
        core_tvd_rows.append((CORE_TVD_WELL, f"{tvd:.2f}", f"{p:.4f}", f"{kk:.4g}"))

    th, w4 = dirs["2_type_well"], dirs["4_wells"]
    write_csv(th / "typewell_header_W-01.csv", ["well", "x", "y", "kb", "td"], [headers[0]])
    write_csv(th / "typewell_deviation_W-01.csv", ["well", "md", "incl", "azim"], [d for d in devs if d[0] == "W-01"])
    write_csv(th / "typewell_tops_W-01.csv", ["well", "top", "depth"], [t for t in tops_main if t[0] == "W-01"])
    write_csv(w4 / "well_headers_all.csv", ["well", "x", "y", "kb", "td"], headers)
    write_csv(w4 / "deviation_all.csv", ["well", "md", "incl", "azim"], devs)
    write_csv(w4 / "tops_14_wells.csv", ["well", "top", "depth"], tops_main)
    write_csv(w4 / "tops_W-15_W-16_other_names.csv", ["well", "top", "depth"], tops_late)
    write_csv(w4 / "core_5_wells_md_percent.csv", ["well", "md", "poro", "perm", "grain_dens"], core_rows)
    write_csv(w4 / "core_W-14_tvd_fraction.csv", ["well", "depth", "poro", "perm"], core_tvd_rows)

    # ------------------------------------------------------------ 5 fault sticks
    f5 = dirs["5_fault_sticks"]
    srng = np.random.default_rng(SEED + 3)

    def sticks(name, ts, zs):
        f = FAULTS[name]
        out = []
        for si, t in enumerate(ts, start=1):
            for z in zs:
                p = f["p0"] + t * f["L"] * f["u"] + (z - Z_REF) / math.tan(math.radians(f["dip"])) * f["n"]
                p = p + srng.normal(0, 3.0, 2)
                out.append((name, si, p[0], p[1], z + srng.normal(0, 2.0)))
        return out

    rows = sticks("F_Main", (0.08, 0.22, 0.38, 0.54, 0.70, 0.86), np.linspace(850, 1650, 9)) + \
        sticks("F_Minor", (0.03, 0.30, 0.55, 0.80, 0.97), np.linspace(850, 1650, 9))
    write_csv(f5 / "fault_sticks_F_Main_F_Minor_m.csv", ["fault", "stick", "x", "y", "z"],
              [(n, s, f"{x_:.2f}", f"{y_:.2f}", f"{z_:.2f}") for n, s, x_, y_, z_ in rows])
    ft = 1 / 0.3048
    erows = sticks("F_East", (0.15, 0.50, 0.85), np.linspace(850, 1650, 9))
    txt = ""
    cur = None
    for n, s, x_, y_, z_ in erows:
        if cur is not None and s != cur:
            txt += "999 999 999\n"
        cur = s
        txt += f"{x_ * ft:.2f} {y_ * ft:.2f} {z_ * ft:.2f}\n"
    (f5 / "fault_F_East_headerless_feet.txt").write_text(txt)
    for nm, f in FAULTS.items():
        info["faults"][nm] = dict(trace_p0=[round(float(v), 1) for v in f["p0"]], trace_p1=[round(float(v), 1) for v in f["p1"]],
                                  trace_depth=Z_REF, dip_deg=f["dip"], nominal_max_throw_m=f["throw"],
                                  downthrown_side_unit_vector=[round(float(v), 4) for v in f["n"]])

    # ------------------------------------------------------------ 9 truth cubes, then maps / vpc / images
    cubes = truth.cubes()
    xc = X0 + (np.arange(NX) + 0.5) * CELL
    yc = Y0 + (np.arange(NY) + 0.5) * CELL
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    np.savez_compressed(dirs["9_reference_truth"] / "truth_cubes.npz", x_centres=xc, y_centres=yc,
                        **{f"{iv}_facies": c["facies"] for iv, c in cubes.items()},
                        **{f"{iv}_phi_noise_free": c["phi"] for iv, c in cubes.items()},
                        **{f"{iv}_thickness": c["thickness"].astype(np.float32) for iv, c in cubes.items()})

    p6 = dirs["6_property_maps"]
    xs_m = np.arange(X0, X0 + LX + 1e-6, CELL)
    ys_m = np.arange(Y0, Y0 + LY + 1e-6, CELL)
    Xm, Ym = np.meshgrid(xs_m, ys_m, indexing="ij")

    def to_nodes(a):                                           # cell-centre field -> node grid (61 x 51), edge-padded
        pad = np.pad(a, 1, mode="edge")
        return 0.25 * (pad[:-1, :-1] + pad[1:, :-1] + pad[:-1, 1:] + pad[1:, 1:])

    def layer_mean(a):
        return a.mean(axis=2)
    phi_a = to_nodes(layer_mean(cubes["Res_A"]["phi"]))
    phi_b = to_nodes(layer_mean(cubes["Res_B"]["phi"]))
    sand_b = to_nodes((cubes["Res_B"]["facies"] == 1).mean(axis=2))
    logk_b = np.zeros_like(cubes["Res_B"]["phi"], float)
    for f, (a, bb, s) in PERM_REL.items():
        logk_b = np.where(cubes["Res_B"]["facies"] == f, a + bb * cubes["Res_B"]["phi"], logk_b)
    perm_b = to_nodes(10 ** layer_mean(logk_b))
    write_surface(p6 / "PHIT_Res_A_fraction.irap", Xm, Ym, phi_a, "irap_ascii", "PHIT_Res_A")
    write_surface(p6 / "PHIT_Res_B_percent.irap", Xm, Ym, phi_b * 100.0, "irap_ascii", "PHIT_Res_B")
    write_surface(p6 / "SAND_FRACTION_Res_B.irap", Xm, Ym, sand_b, "irap_ascii", "SAND_Res_B")
    write_surface(p6 / "PERM_geomean_Res_B_mD.irap", Xm, Ym, perm_b, "irap_ascii", "PERM_Res_B")

    # variograms: models of the LATENT field used to generate facies (truth), plus a typical porosity model
    from cosma.variogram import Structure, VariogramModel, export_variogram
    w_ = LAT["long_weight"]
    L, S = LAT["long"], LAT["short"]

    def lat_model(variable, transform, nugget, sill_scale=1.0, vscale=1.0):
        return VariogramModel(nugget, (
            Structure("gaussian", (1 - nugget) * w_ * sill_scale, L["major"], L["minor"], L["vert_m"] * vscale),
            Structure("gaussian", (1 - nugget) * (1 - w_) * sill_scale, S["major"], S["minor"], S["vert_m"] * vscale)),
            LAT["azimuth"], transform, variable).check()
    v7 = dirs["7_variograms"]
    export_variogram(lat_model("PHIT", "normal_score", 0.05, vscale=MEAN_THICKNESS["Res_A"] / 61.0), v7 / "PHIT_normal_score_Res_A.json")
    export_variogram(lat_model("FACIES[1]", "indicator", 0.05, sill_scale=0.24), v7 / "FACIES_1_sand_indicator_Res_B.json")
    export_variogram(lat_model("PHIT", "normal_score", 0.05), v7 / "PHIT_normal_score_Res_B.var")
    export_variogram(lat_model("VSHALE", "normal_score", 0.08), v7 / "VSHALE_normal_score_Res_B.json")

    # proportion-map images: facies at the middle of an interval, painted like a hand-made screenshot
    from PIL import Image, ImageDraw
    COL = {1: (255, 215, 0), 2: (205, 133, 63), 3: (105, 105, 105)}
    p8 = dirs["8_proportion_map_images"]
    px = 10.0            # metres per pixel in the PNG
    W, H = int(LX / px), int(LY / px)

    def slice_image(iv, zr_mid, res_m):
        gx = np.arange(X0 + res_m / 2, X0 + LX, res_m)
        gy = np.arange(Y0 + LY - res_m / 2, Y0, -res_m)               # rows run north -> south
        GX, GY = np.meshgrid(gx, gy)                                  # (rows, cols)
        k = np.full(GX.shape, zr_mid)
        c = np.stack([(GX - X0) / CELL - 0.5, (GY - Y0) / CELL - 0.5, k * NK - 0.5])
        zf = map_coordinates(truth.fields[iv]["z"], c.reshape(3, -1).reshape(3, GX.shape[0], GX.shape[1]), order=1, mode="nearest")
        fac = facies_from_z(iv, GX, GY, k, zf)
        rgb = np.zeros(GX.shape + (3,), np.uint8)
        for f, col in COL.items():
            rgb[fac == f] = col
        # outside the model polygon: no data (white)
        inside = shapely.contains_xy(POLY, GX, GY)
        rgb[~inside] = (255, 255, 255)
        return rgb, inside

    def best_slice(iv):
        """Relative depth whose sand share inside the polygon is closest to the interval's: one slice of a random field can be
        far from the interval average, which would make the 'map' contradict the wells for no good reason."""
        target = float((cubes[iv]["facies"] == 1).mean())
        best = None
        for z_ in np.arange(0.30, 0.701, 0.05):
            r_, ins = slice_image(iv, float(z_), 40.0)
            share = float((np.all(r_ == COL[1], axis=2) & ins).sum() / ins.sum())
            if best is None or abs(share - target) < best[0]:
                best = (abs(share - target), float(z_), share)
        return best[1], best[2], target

    zr_b, sh_b, tg_b = best_slice("Res_B")
    zr_a, sh_a, tg_a = best_slice("Res_A")
    info["image_slices"] = dict(Res_B=dict(relative_depth=round(zr_b, 2), sand_share_in_image=round(sh_b, 3), sand_share_interval=round(tg_b, 3)),
                                Res_A=dict(relative_depth=round(zr_a, 2), sand_share_in_image=round(sh_a, 3), sand_share_interval=round(tg_a, 3)))
    rgb, inside = slice_image("Res_B", zr_b, px)
    im = Image.new("RGB", (800, 800), (240, 240, 240))
    ox, oy = 100, 60
    im.paste(Image.fromarray(rgb), (ox, oy))
    dr = ImageDraw.Draw(im)
    dr.rectangle([ox - 1, oy - 1, ox + W, oy + H], outline=(0, 0, 0))
    ctrl = []
    for fx, fy in ((0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0), (0.5, 1)):
        cx = ox + fx * W + (-14 if fx == 0 else 14 if fx == 1 else 0)
        cy = oy + fy * H + (-14 if fy == 0 else 14)
        cx, cy = int(round(cx)), int(round(cy))
        dr.line([cx - 5, cy, cx + 5, cy], fill=(0, 0, 0))
        dr.line([cx, cy - 5, cx, cy + 5], fill=(0, 0, 0))
        u, v = cx + 0.5, cy + 0.5                                   # pixel centres sit at +0.5 in COSMA's convention
        mx = X0 + (u - ox) * px
        my = Y0 + LY - (v - oy) * px
        ctrl.append((u, v, round(mx, 1), round(my, 1)))
    dr.text((100, 625), "Res_B: facies at mid-interval (synthetic, hand-made screenshot look)", fill=(0, 0, 0))
    for i, (f, nm) in enumerate(FACIES.items()):
        dr.rectangle([100 + 130 * i, 660, 120 + 130 * i, 680], fill=COL[f], outline=(0, 0, 0))
        dr.text((126 + 130 * i, 664), nm, fill=(0, 0, 0))
    im.save(p8 / "facies_map_Res_B_screenshot.png")
    write_csv(p8 / "facies_map_Res_B_control_points.csv", ["pixel_x", "pixel_y", "map_x", "map_y"], ctrl)
    # same idea for Res_A as a georeferenced RGBA TIFF (20 m pixels, transparent outside the model polygon)
    rgb_a, inside_a = slice_image("Res_A", zr_a, 20.0)
    import tifffile
    rgba = np.dstack([rgb_a, np.where(inside_a, 255, 0).astype(np.uint8)])
    tifffile.imwrite(p8 / "facies_map_Res_A_geotiff.tif", rgba, photometric="rgb", extrasamples=[2],
                     extratags=[(33550, "d", 3, (20.0, 20.0, 0.0), True),
                                (33922, "d", 6, (0.0, 0.0, 0.0, X0, Y0 + LY, 0.0), True),
                                (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32616), True)])

    # typed vertical proportion curves (relative position, 10 bins) from the truth cubes, inside the polygon
    mask = shapely.contains_xy(POLY, XX, YY)
    vpc_info = {}
    for iv in INTERVALS:
        fac = cubes[iv]["facies"][mask]                                 # (cols, NK)
        rows = []
        for b_ in range(10):
            sl = fac[:, b_ * NK // 10:(b_ + 1) * NK // 10]
            p = [float((sl == f).mean()) for f in (1, 2, 3)]
            p = [round(v, 6) for v in p]
            p[0] = round(1.0 - p[1] - p[2], 6)
            rows.append((f"{(b_ + 0.5) / 10:.2f}", *[f"{v:.6f}" for v in p]))
        write_csv(p8.parent / "9_reference_truth" / f"typed_vpc_{iv}.csv", ["relative_position", "Sand", "Silt", "Shale"], rows)
        vpc_info[iv] = {FACIES[f]: round(float((fac == f).mean()), 4) for f in (1, 2, 3)}
    info["facies_proportions_truth_inside_polygon"] = vpc_info

    # facies rules and how well they recover the truth in the logs
    tr_f = np.concatenate([a for a, _ in classification])
    ru_f = np.concatenate([b_ for _, b_ in classification])
    conf = {f"truth {FACIES[t]} -> rule {FACIES[r]}": round(float(((tr_f == t) & (ru_f == r)).sum() / (tr_f == t).sum()), 3)
            for t in (1, 2, 3) for r in (1, 2, 3)}
    info["facies_rules_vs_truth"] = dict(
        rules="Sand: VSHALE < 0.25; Silt: VSHALE >= 0.25 and < 0.60; Shale: VSHALE >= 0.60", fraction_correct=round(float((tr_f == ru_f).mean()), 3),
        confusion_row_fractions=conf)
    info["generation"] = dict(
        latent_gaussian=LAT, facies_scheme=FACIES, mean_thickness_m=MEAN_THICKNESS,
        shale_volume_by_facies_mean_sd=MU_V, porosity_by_facies_mean_sd=MU_P,
        permeability="log10 k(mD) = a + b*phi + N(0,s); (a,b,s) by facies: " + json.dumps({FACIES[k]: v for k, v in PERM_REL.items()}))
    (dirs["9_reference_truth"] / "truth_summary.json").write_text(json.dumps(info, indent=2, default=float))
    write_readme(out, info)
    shutil.copy(Path(__file__), out / "make_dataset.py")
    vp = Path(__file__).with_name("verify_with_cosma.py")
    if vp.exists():
        shutil.copy(vp, out / "verify_with_cosma.py")
    return info


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_dataset_out")
