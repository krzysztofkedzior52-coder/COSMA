"""The logic behind the variogram editor, with no GUI: choose data, compute experimental
variograms, edit or fit a model, save it on the interval. The Qt dialog is a thin view.

Nothing is guessed: a range that has not been fitted or typed is None, and `model()` refuses
to build until every range is set. Auto-fit fills only what the data can support and says
what it could not fit.
"""

from __future__ import annotations

import numpy as np

from .fault_checks import crossing_warnings
from .project import Project, ProjectError
from .variogram import (KINDS, TYPES, Experimental, Structure, VariogramError, VariogramModel, experimental_variogram,
                        fit_direction)
from .variogram_data import collect


class VariogramSession:
    def __init__(self, project: Project, uid: str, variable: str, *, kind: str = "log", source=None,
                 frame: str = "top", transform: str = "none", wells=None, indicator: int | None = None,
                 average: bool = False, min_coverage: float = 0.5):
        """`indicator`: a facies code - the variogram is then of the 0/1 indicator of that code
        and is stored under the property name 'CURVE[code]'.

        `average`: the data are ONE value per well, the layer average (for a facies indicator the
        proportion) - the support that property maps are kriged from. Only horizontal variograms
        exist then; stored under 'NAME (layer average)'."""
        self.project, self.uid = project, uid
        self.kind, self.source, self.frame, self.wells = kind, source, frame, wells
        self.average, self.min_coverage = average, min_coverage
        if average:
            from .maps import layer_averages
            ld = layer_averages(project, uid, variable, kind=kind, source=source, indicator=indicator,
                                min_coverage=min_coverage, wells=wells)
            from .variogram_data import VarioData
            th = float(np.mean(list(ld.thickness.values())))
            self.data = VarioData(ld.interval, f"{ld.variable} (layer average)", "top", np.c_[ld.x, ld.y, np.zeros(ld.x.size)],
                                  ld.value, np.array(ld.wells, dtype=object), {w: 1 for w in ld.wells}, dict(ld.skipped),
                                  dict(ld.thickness), th, None)
            self.codes = None
            if transform == "indicator":
                raise VariogramError("layer averages are continuous: the indicator transform does not apply")
        else:
            self.data = collect(project, uid, variable, kind=kind, source=source, frame=frame, wells=wells)
            self.codes = self.data.facies_codes()
            if indicator is not None:
                self.data = self.data.as_indicator(indicator)
                transform = "indicator"
            elif transform == "indicator":
                raise VariogramError("the indicator transform needs a facies code")
        self.indicator = indicator
        self.variable = self.data.variable
        self.transform = transform
        self.nugget = 0.0
        self.azimuth = 0.0
        self.structures = [dict(type="spherical", contribution=None, major=None, minor=None, vertical=None)]
        self.settings = {k: dict(lag=None, nlags=10, tolerance_deg=22.5 if k != "vertical" else 10.0, bandwidth=None,
                                   vertical_tolerance=None)
                         for k in KINDS}
        self.exp: dict = {}
        self.notes: list = []

    # -- data ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> np.ndarray:
        return self.data.transformed(self.transform)

    @property
    def variance(self) -> float:
        return float(np.var(self.values))

    def set_transform(self, how: str):
        if self.indicator is not None and not self.average:
            raise VariogramError("an indicator variogram is of 0/1 data: no transform can be applied")
        if how not in ("none", "normal_score"):
            raise VariogramError("transform must be 'none' or 'normal_score'")
        self.transform = how
        self.exp.clear()

    def default_lag(self, kind: str) -> float | None:
        """A starting lag size from the data (median spacing for vertical, median nearest-well
        distance for horizontal). It is shown to the user as a starting point only."""
        if kind == "vertical":
            d = []
            for w in self.data.used:
                z = np.sort(self.data.points[self.data.groups == w][:, 2])
                if z.size > 1:
                    d.append(np.median(np.diff(z)))
            return float(np.median(d)) if d else None
        xy = {w: self.data.points[self.data.groups == w][0, :2] for w in self.data.used}
        if len(xy) < 2:
            return None
        pts = np.array(list(xy.values()))
        dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(dist, np.inf)
        return float(np.median(dist.min(axis=1)))

    # -- experimental ------------------------------------------------------------------------------------------
    def compute(self, kind: str, **override) -> Experimental:
        if kind not in KINDS:
            raise VariogramError(f"kind must be one of {KINDS}")
        if self.average and kind == "vertical":
            raise VariogramError("layer averages have one value per well: there is no vertical variogram")
        s = {**self.settings[kind], **override}
        self.settings[kind] = s
        if s["lag"] is None:
            s["lag"] = self.default_lag(kind)
        if kind != "vertical" and s["vertical_tolerance"] is None:
            s["vertical_tolerance"] = 1.0 if self.average else self.default_lag("vertical")   # same-layer pairs (averages: one z)
        if s["lag"] is None:
            raise VariogramError(f"{kind}: cannot suggest a lag (only one well); set the lag size yourself" if kind != "vertical"
                                 else "vertical: cannot suggest a lag; set the lag size yourself")
        e = experimental_variogram(self.data.points, self.values, kind, lag=s["lag"], nlags=int(s["nlags"]), azimuth=self.azimuth,
                                   tolerance_deg=s["tolerance_deg"], bandwidth=s["bandwidth"], groups=self.data.groups,
                                   vertical_tolerance=s["vertical_tolerance"])
        self.exp[kind] = e
        return e

    def scan_azimuths(self, step_deg: float = 15.0, nlags: int = 3) -> dict:
        """Mean semivariance of the first `nlags` lags for azimuths 0..180: the direction of
        lowest semivariance is the direction of longest continuity (the major axis). Needs
        pairs between wells; azimuths with no pairs are reported as None."""
        lag = self.settings["major"]["lag"] or self.default_lag("major")
        if lag is None:
            raise VariogramError("only one well: no horizontal pairs, so no azimuth can be inferred - set it by hand")
        out = {}
        for az in np.arange(0.0, 180.0, step_deg):
            e = experimental_variogram(self.data.points, self.values, "major", lag=lag, nlags=nlags, azimuth=float(az),
                                       tolerance_deg=min(self.settings["major"]["tolerance_deg"], step_deg / 2 + 1e-9),
                                       groups=self.data.groups,
                                       vertical_tolerance=self.settings["major"]["vertical_tolerance"] or (1.0 if self.average else self.default_lag("vertical")))
            ok = np.isfinite(e.gamma) & (e.npairs > 0)
            out[float(az)] = float(np.average(e.gamma[ok], weights=e.npairs[ok])) if ok.any() else None
        return out

    def suggest_azimuth(self, **kw):
        scan = {a: g for a, g in self.scan_azimuths(**kw).items() if g is not None}
        if len(scan) < 3:
            return None
        best = min(scan, key=scan.get)
        vals = sorted(scan.values())
        return None if vals[0] > 0.9 * vals[len(vals) // 2] else best        # no clear anisotropy: no suggestion

    def load(self, model: VariogramModel):
        """Start editing from a saved model (it must describe the same data)."""
        if model.transform != self.transform:
            raise VariogramError(f"that variogram was computed on {model.transform!r} data, this session uses {self.transform!r}")
        self.nugget, self.azimuth = model.nugget, model.azimuth
        self.structures = [dict(type=t.type, contribution=t.contribution, major=t.range_major, minor=t.range_minor,
                                vertical=t.range_vert) for t in model.structures]
        self.exp.clear()

    # -- model ---------------------------------------------------------------------------------------------------------
    def set_nugget(self, v):
        self.nugget = float(v)

    def set_azimuth(self, v):
        v = float(v)
        if v != self.azimuth:                       # horizontal experimentals belong to the azimuth they were computed at
            for k in ("major", "minor"):
                self.exp.pop(k, None)
        self.azimuth = v

    def add_structure(self, type="spherical"):
        if len(self.structures) >= 3:
            raise VariogramError("at most three structures are supported")
        self.structures.append(dict(type=type, contribution=None, major=None, minor=None, vertical=None))

    def remove_structure(self, i: int):
        if len(self.structures) == 1:
            raise VariogramError("a model needs at least one structure")
        self.structures.pop(i)

    def set_structure(self, i: int, **kw):
        bad = set(kw) - {"type", "contribution", "major", "minor", "vertical"}
        if bad:
            raise VariogramError(f"unknown structure field(s) {sorted(bad)}")
        if "type" in kw and kw["type"] not in TYPES:
            raise VariogramError(f"unknown variogram type {kw['type']!r}; known: {TYPES}")
        self.structures[i].update(kw)

    def missing(self) -> list:
        out = []
        for i, s in enumerate(self.structures, 1):
            out += [f"structure {i}: {k}" for k in ("contribution", "major", "minor", "vertical") if s[k] is None]
        return out

    def model(self) -> VariogramModel:
        m = self.missing()
        if m:
            raise VariogramError("not set yet: " + "; ".join(m))
        return VariogramModel(self.nugget, tuple(Structure(s["type"], s["contribution"], s["major"], s["minor"], s["vertical"])
                                                 for s in self.structures), self.azimuth, self.transform, self.variable).check()

    def curve(self, kind: str, hmax: float, n: int = 200):
        h = np.linspace(0.0, hmax, n)
        return h, self.model().gamma_along(h, kind)

    # -- automatic fit -------------------------------------------------------------------------------------------------------
    def auto_fit(self, vtype: str = "spherical", min_pairs: int = 5) -> list:
        """Fit ONE structure. Vertical (if computed) gives nugget and sill; major and minor
        then get their ranges with nugget and sill held. A direction whose experimental
        variogram has too few lags with pairs is left unset and reported. Returns messages."""
        msgs = []
        s = self.structures[0]
        s["type"] = vtype
        del self.structures[1:]
        if self.average:
            if "major" not in self.exp:
                raise VariogramError("compute the major experimental variogram first")
            f = fit_direction(self.exp["major"], vtype, min_pairs=min_pairs)
            self.nugget = f["nugget"]
            s["contribution"], s["major"] = f["contribution"], f["range"]
            s["vertical"] = self.data.mean_thickness
            msgs.append(f"major: nugget {f['nugget']:.4g}, contribution {f['contribution']:.4g}, range {f['range']:.4g} m "
                        f"(rmse {f['rmse']:.3g})")
            msgs.append(f"vertical range set to the mean interval thickness ({self.data.mean_thickness:.4g} m): "
                        "maps do not use it")
            sill = f["nugget"] + f["contribution"]
            e = self.exp.get("minor")
            if e is None:
                msgs.append("minor: not computed - range left unset")
                s["minor"] = None
            else:
                try:
                    g = fit_direction(e, vtype, nugget=f["nugget"], sill=sill, min_pairs=min_pairs)
                    s["minor"] = g["range"]
                    msgs.append(f"minor: range {g['range']:.4g} m (rmse {g['rmse']:.3g})")
                except VariogramError as exc:
                    s["minor"] = None
                    msgs.append(f"minor: not fitted ({exc}) - enter the range by hand")
            if s["minor"] is not None and s["minor"] > s["major"]:
                msgs.append("the fitted minor range exceeds the major: the azimuth is probably 90 degrees off - set it and refit")
                s["minor"] = None
            return msgs
        if "vertical" not in self.exp:
            raise VariogramError("compute the vertical experimental variogram first")
        f = fit_direction(self.exp["vertical"], vtype, min_pairs=min_pairs)
        self.nugget = f["nugget"]
        s["contribution"], s["vertical"] = f["contribution"], f["range"]
        msgs.append(f"vertical: nugget {f['nugget']:.4g}, contribution {f['contribution']:.4g}, range {f['range']:.4g} m "
                    f"(rmse {f['rmse']:.3g})")
        sill = f["nugget"] + f["contribution"]
        for k in ("major", "minor"):
            e = self.exp.get(k)
            if e is None:
                msgs.append(f"{k}: not computed - range left unset")
                s[k] = None
                continue
            try:
                g = fit_direction(e, vtype, nugget=f["nugget"], sill=sill, min_pairs=min_pairs)
            except VariogramError as exc:
                msgs.append(f"{k}: not fitted ({exc}) - enter the range by hand")
                s[k] = None
                continue
            s[k] = g["range"]
            msgs.append(f"{k}: range {g['range']:.4g} m (rmse {g['rmse']:.3g})")
        if s["major"] is not None and s["minor"] is not None and s["minor"] > s["major"]:
            msgs.append("the fitted minor range exceeds the major: the azimuth is probably 90 degrees off - set it and refit")
            s["minor"] = None
        return msgs

    # -- warnings --------------------------------------------------------------------------------------------------------
    def warnings(self) -> list:
        w = []
        n = len(self.data.used)
        if self.average and n >= 2:
            w.append(f"{n} wells give {n * (n - 1) // 2} pairs in total: a variogram of layer averages is only a rough guide.")
        if n < 2:
            w.append("Only one well: horizontal variograms cannot be computed; the horizontal ranges must be entered from "
                     "geological knowledge (analogues, outcrop, seismic).")
        elif n < 5:
            w.append(f"Only {n} wells: horizontal variograms rest on very few pairs and are poorly constrained.")
        if self.indicator is not None and not self.average:
            p = float(np.mean(self.data.values))
            k = int(self.data.values.sum())
            if k < 30 or p < 0.02 or p > 0.98:
                w.append(f"Facies {self.indicator} makes up {p:.1%} of the samples ({k} samples): its variogram will be very noisy.")
        w += crossing_warnings(self.data.crossings)
        if self.data.skipped:
            w.append("Wells not used: " + "; ".join(f"{k} ({v})" for k, v in self.data.skipped.items()))
        if self.transform == "normal_score":
            if self.nugget + sum(s["contribution"] or 0 for s in self.structures) > 0 and \
                    abs(self.nugget + sum(s["contribution"] or 0 for s in self.structures) - 1.0) > 0.25:
                w.append("Normal-score data have unit variance: the model's total sill is far from 1.")
        elif not self.missing():
            m = self.model()
            if abs(m.sill / self.variance - 1.0) > 0.25:
                w.append(f"The total sill ({m.sill:.4g}) differs from the data variance ({self.variance:.4g}) by more than 25%.")
        if self.data.mean_thickness < max(self.settings["vertical"]["lag"] or 0, 0) * 3:
            w.append("The interval is only a few lags thick: the vertical variogram is mostly noise.")
        return w

    # -- save ----------------------------------------------------------------------------------------------------------------------
    def conflicts(self, label: str) -> list:
        return self.project.conflicts_variogram(self.uid, self.variable, label)

    def save(self, label: str, actions: dict | None = None):
        if not label or not label.strip():
            raise ProjectError("give the variogram a name")
        curve = self.variable.replace(" (layer average)", "")
        curve = curve.split("[")[0] if self.indicator is not None else curve
        meta = dict(curve=curve, kind=self.kind, source=self.source, frame=self.frame, indicator=self.indicator,
                    wells=None if self.wells is None else list(self.wells), average=self.average,
                    min_coverage=self.min_coverage)
        return self.project.attach_variogram(self.uid, self.variable, self.model(), label.strip(), actions, meta)
