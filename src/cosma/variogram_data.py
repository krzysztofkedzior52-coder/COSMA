"""Collect the samples a variogram is computed from: one property, in one stratigraphic
interval, from every well that has both of the interval's picks.

Frame: x, y are the sample positions; z is the position inside the interval, so that
samples from wells where the interval sits at different depths line up:
  "top"           z = TVDSS of the sample minus TVDSS of the interval top   (depth below top, m)
  "proportional"  z = that depth scaled by (mean thickness / this well's thickness), i.e.
                  all wells are stretched to the mean thickness (proportional layering)
Nothing is guessed: if the property exists in several sources of a well, the source must
be named; a well missing a pick, a trajectory, or the property is reported in `skipped`
with the reason instead of silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .project import Project, ProjectError
from .fault_checks import well_crossings
from .variogram import normal_score
from .wells.io import resolve_top

FRAMES = ("top", "proportional")
SOURCES = ("log", "core")


@dataclass
class VarioData:
    interval: str
    variable: str
    frame: str
    points: np.ndarray                 # (n, 3) x, y, z
    values: np.ndarray                 # (n,)
    groups: np.ndarray                 # (n,) well name per sample
    used: dict = field(default_factory=dict)      # well -> number of samples
    skipped: dict = field(default_factory=dict)   # well -> reason
    thickness: dict = field(default_factory=dict)  # well -> interval thickness (TVDSS, m)
    mean_thickness: float = float("nan")
    indicator: int | None = None       # facies code this data is the indicator of (values are then 0/1)
    crossings: dict = field(default_factory=dict)  # well -> [Crossing]: faults its path cuts inside the interval

    def transformed(self, how: str) -> np.ndarray:
        if how == "none":
            return self.values
        if how == "indicator":
            if self.indicator is None:
                raise ProjectError("the indicator transform needs a facies code: use as_indicator(code)")
            return self.values
        if how == "normal_score":
            if self.indicator is not None:
                raise ProjectError("a normal-score transform makes no sense for a 0/1 indicator")
            return normal_score(self.values)
        raise ProjectError(f"unknown transform {how!r}")

    def facies_codes(self, max_codes: int = 50):
        """{code: proportion of samples} if the values are all whole numbers (a facies / category
        curve), else None. Proportions are over samples, not thickness-weighted."""
        v = self.values
        if self.indicator is not None or v.size == 0 or np.any(np.abs(v - np.round(v)) > 1e-9):
            return None
        codes, n = np.unique(np.round(v).astype(int), return_counts=True)
        if codes.size > max_codes:
            return None
        return {int(c): float(k) / v.size for c, k in zip(codes, n)}

    def as_indicator(self, code: int) -> "VarioData":
        """The same samples as the 0/1 indicator of one facies code."""
        codes = self.facies_codes()
        if codes is None:
            raise ProjectError(f"{self.variable!r} does not look like a facies curve (values are not whole-number codes)")
        if code not in codes:
            raise ProjectError(f"facies code {code} does not occur in {self.interval!r}; codes present: {sorted(codes)}")
        if codes[code] >= 1.0:
            raise ProjectError(f"every sample is facies {code}: an indicator without variation has no variogram")
        return replace(self, values=(np.round(self.values) == code).astype(float), variable=f"{self.variable}[{code}]",
                       indicator=int(code))


def _bounds(project: Project, uid: str):
    ch = project.chart
    iv = ch.interval(uid)
    i = ch.interval_index(uid)
    below = ch.intervals[i + 1].name if i + 1 < len(ch.intervals) else ch.base_name
    return iv.name, below


def _pick(rec, name):
    return rec.tops.get(name)


def collect(project: Project, uid: str, variable: str, *, kind: str = "log", source: dict | str | None = None,
            frame: str = "top", wells=None) -> VarioData:
    """`source`: a source name for all wells, or {well: source}; required only where the
    property is present in more than one source of a well."""
    if project.chart is None:
        raise ProjectError("define the stratigraphic chart first")
    if kind not in SOURCES:
        raise ProjectError(f"kind must be one of {SOURCES}")
    if frame not in FRAMES:
        raise ProjectError(f"frame must be one of {FRAMES}")
    top_name, base_name = _bounds(project, uid)
    data = VarioData(top_name, variable, frame, np.empty((0, 3)), np.empty(0), np.empty(0, object))
    per = {}
    for name, rec in project.wells.items():
        if wells is not None and name not in wells:
            continue
        pt, pb = _pick(rec, top_name), _pick(rec, base_name)
        if pt is None or pb is None:
            data.skipped[name] = f"no pick for {top_name if pt is None else base_name!r}"
            continue
        tr = rec.trajectory
        if tr is None:
            data.skipped[name] = "needs a header and a deviation survey"
            continue
        try:
            rt, rb = resolve_top(name, tr, pt), resolve_top(name, tr, pb)
        except Exception as exc:
            data.skipped[name] = str(exc)
            continue
        if rb.tvdss <= rt.tvdss:
            data.skipped[name] = f"interval has no thickness ({top_name!r} at {rt.tvdss:g}, {base_name!r} at {rb.tvdss:g} TVDSS)"
            continue
        holders = []
        if kind == "log":
            holders = [(s, ls.md, ls.curves[variable]) for s, ls in rec.logs.items() if variable in ls.curves]
        else:
            for s, c in rec.resolved_core().items():
                if variable in c.columns:
                    holders.append((s, c.md, c.columns[variable]))
        if not holders:
            data.skipped[name] = f"no {kind} data called {variable!r}"
            continue
        want = source.get(name) if isinstance(source, dict) else source
        if want is not None:
            holders = [h for h in holders if h[0] == want]
            if not holders:
                data.skipped[name] = f"no {kind} source {want!r} with {variable!r}"
                continue
        if len(holders) > 1:
            raise ProjectError(f"{name}: {variable!r} is in several sources ({', '.join(h[0] for h in holders)}); "
                               "name the one to use")
        _, md, val = holders[0]
        md, val = np.asarray(md, float), np.asarray(val, float)
        lo, hi = tr.md_range
        m = (md >= max(rt.md, lo)) & (md <= min(rb.md, hi)) & np.isfinite(val)
        if not m.any():
            data.skipped[name] = f"no valid {variable!r} samples between {top_name!r} and {base_name!r}"
            continue
        x, y, z = tr.at_md(md[m])
        per[name] = (x, y, z - rt.tvdss, val[m], rb.tvdss - rt.tvdss)
        if project.faults:
            cs = well_crossings(project.faults, tr, rt.md, rb.md)
            if cs:
                data.crossings[name] = cs
    if not per:
        raise ProjectError(f"no well has {variable!r} data in {top_name!r}"
                           + (": " + "; ".join(f"{w}: {r}" for w, r in data.skipped.items()) if data.skipped else ""))
    data.thickness = {w: v[4] for w, v in per.items()}
    data.mean_thickness = float(np.mean(list(data.thickness.values())))
    xs, vs, gs = [], [], []
    for w, (x, y, d, v, th) in per.items():
        z = d * (data.mean_thickness / th) if frame == "proportional" else d
        xs.append(np.c_[x, y, z])
        vs.append(v)
        gs.append(np.full(v.size, w, object))
        data.used[w] = int(v.size)
    data.points, data.values, data.groups = np.vstack(xs), np.concatenate(vs), np.concatenate(gs)
    return data


def available_properties(project: Project, kind: str = "log") -> dict:
    """{property name: sorted sources that have it} across all wells, for a chooser."""
    out: dict = {}
    for rec in project.wells.values():
        if kind == "log":
            for src, ls in rec.logs.items():
                for c in ls.curves:
                    out.setdefault(c, set()).add(src)
        elif kind == "core":
            for src, rc in rec.core.items():
                for c in rc.columns:
                    out.setdefault(c, set()).add(src)
        else:
            raise ProjectError(f"kind must be one of {SOURCES}")
    return {k: sorted(v) for k, v in sorted(out.items())}
