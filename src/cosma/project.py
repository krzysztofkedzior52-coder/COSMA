"""Project: the wells (and, later, everything else) loaded so far.

Wells store what was *received* - header, deviation survey, picks as (depth, type), core
and logs per source file - and derive trajectories and pick positions on demand.
Re-importing a corrected survey or header therefore never leaves stale coordinates
behind, and the order in which a well's data arrives does not matter.

When an import meets data the well already has (same top name, same curve over an
overlapping depth range, ...), that is reported as a `Conflict` and the caller must give
an `Action` for each: replace, skip, or import with a suffix. Nothing is overwritten or
dropped without one. Every `apply_*` is atomic: if anything is rejected, the project is
left as it was.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace

import numpy as np

from .strat import BASE_UID, StratChart, StratError
from .wells.io import WellDataError, _depth_to_position, place_survey, resolve_core, resolve_top
from .wells.model import LogSet, RawCore, RawTop, Survey
from .wells.trajectory import Trajectory

DEPTH_TOL = 0.01          # m: core samples closer than this are "the same depth"


class ProjectError(ValueError):
    pass


@dataclass
class WellRecord:
    name: str
    header: dict | None = None           # x, y, kb, td (metres)
    survey: Survey | None = None
    logs: dict = field(default_factory=dict)      # source -> LogSet
    tops: dict = field(default_factory=dict)      # top name -> RawTop
    core: dict = field(default_factory=dict)      # source -> RawCore

    @property
    def trajectory(self) -> Trajectory | None:
        if self.header is None or self.survey is None:
            return None
        return place_survey(self.survey, self.header)

    def resolved_tops(self) -> list:
        tr = self.trajectory
        if tr is None:
            raise ProjectError(f"{self.name}: needs a header and a deviation survey to place picks")
        return [resolve_top(self.name, tr, t) for t in self.tops.values()]

    def resolved_core(self) -> dict:
        tr = self.trajectory
        if tr is None:
            raise ProjectError(f"{self.name}: needs a header and a deviation survey to place core")
        return {src: resolve_core(self.name, tr, rc) for src, rc in self.core.items()}

    def missing(self) -> list:
        out = []
        if self.header is None:
            out.append("header")
        if self.survey is None:
            out.append("deviation survey")
        return out


@dataclass
class WellLink:
    mode: str            # "existing" | "new"
    name: str            # existing well to connect to, or the name of the new well


@dataclass(frozen=True)
class Conflict:
    key: str                      # "<well>|<what>|<item>"
    well: str
    what: str                     # header | survey | top | curve | core
    item: str                     # top name, curve mnemonic, core column ...
    existing: str
    new: str
    identical: bool = False
    allowed: tuple = ("replace", "skip", "suffix")
    reserved: frozenset = frozenset()     # names a suffixed item must not collide with
    base: str | None = None               # the name the suffix is appended to, if not `item`


@dataclass(frozen=True)
class Action:
    mode: str                     # "replace" | "skip" | "suffix"
    suffix: str = ""


@dataclass
class ImportResult:
    kind: str
    applied: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def suggest_links(file_wells, existing) -> dict:
    """Default connection per well in the file: exact name -> that well; a unique match after
    ignoring case and punctuation -> that well (noted, for the user to check); otherwise new."""
    out = {}
    by_norm = {}
    for e in existing:
        by_norm.setdefault(_norm(e), []).append(e)
    for w in file_wells:
        if w in existing:
            out[w] = (WellLink("existing", w), "name matches")
        elif len(by_norm.get(_norm(w), [])) == 1:
            t = by_norm[_norm(w)][0]
            out[w] = (WellLink("existing", t), f"similar name: {w!r} ~ {t!r} - check")
        else:
            out[w] = (WellLink("new", w), "no well of this name yet")
    return out


def check_action(c: Conflict, a) -> str | None:
    """Problem text if `a` is not a usable answer to conflict `c`, else None."""
    if not isinstance(a, Action) or a.mode not in ("replace", "skip", "suffix"):
        return f"{c.well} {c.what} {c.item}: choose replace, skip or import with suffix"
    if a.mode not in c.allowed:
        return f"{c.well} {c.what} {c.item}: '{a.mode}' is not possible here (allowed: {', '.join(c.allowed)})"
    if a.mode == "suffix":
        if not a.suffix.strip():
            return f"{c.well} {c.what} {c.item}: give a suffix"
        new = (c.item if c.base is None else c.base) + a.suffix
        if new in c.reserved:
            return f"{c.well} {c.what} {c.item}: {new!r} already exists - use another suffix"
    return None


def _rng(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return (float(x[ok].min()), float(x[ok].max())) if ok.any() else None


def _curve_range(md, values):
    ok = np.isfinite(np.asarray(values, float))
    return _rng(np.asarray(md)[ok]) if ok.any() else None


class Project:
    def __init__(self, name: str = ""):
        self.name = name
        self.wells: dict = {}
        self.type_well: str | None = None
        self.chart: StratChart | None = None
        self.surfaces: dict = {}        # horizon uid -> {label: Surface}
        self.primary: dict = {}         # horizon uid -> label of the surface used by the model
        self.variograms: dict = {}      # interval uid -> {variable -> {label: VariogramModel}}
        self.var_primary: dict = {}     # (interval uid, variable) -> label of the model the property step uses
        self.var_meta: dict = {}        # (interval uid, variable, label) -> how it was made (curve, kind, source, frame, indicator)
        self.maps: dict = {}            # interval uid -> {variable -> {label: PropertyMap}}
        self.map_primary: dict = {}     # (interval uid, variable) -> label of the map the property step uses
        self.faults: dict = {}          # fault name -> Fault (sticks in metres, depth positive down)
        self.fault_depths: dict = {}    # fault name -> depth (m) at which its vertical trace is taken (user-confirmed)
        self.fault_transmissibility: dict = {}   # fault name -> FaultTransmissibility (user-chosen, explicit)
        self.data_checks = None                  # DataCheckSetup: thresholds, properties, core columns (user-set)
        self.sgr_methods: dict = {}              # name -> fault_relations.Method (the user's own SGR methods)
        self.proportion_maps = {}                # interval name -> ProportionMap (resampled to the model grid)
        self.facies_sim = None                   # FaciesSimSetup
        self.realization_plan = None             # RealizationPlan
        self.proportions = None                  # ProportionSetup: vertical proportion curves per interval
        self.facies_setup = None                 # FaciesSetup: scheme, rules / code map, flags (user-confirmed)
        self.fault_throw_seen: dict = {}         # fault name -> has visible throw, from the last build (not saved)
        self.fault_reach: float | None = None   # distance (m) within which a fault end counts as meeting another fault
        self.fault_interactions: list = []      # fault_interactions.Interaction, answers confirmed by the user
        self.grid_def = None                    # grid_definition.GridDefinition, set by the user
        self.boundary = None                    # shapely Polygon in map coordinates: the model boundary
        self.crs = None                         # 'EPSG:n' (or WKT) of the model coordinates, once a confirmed import states it

    def names(self) -> list:
        return list(self.wells)

    # -- stratigraphy ---------------------------------------------------------------------------------------
    def set_type_well(self, name: str):
        if name not in self.wells:
            raise ProjectError(f"no well {name!r} in the project")
        self.type_well = name

    def build_chart(self, well: str | None = None, *, replace: bool = False) -> StratChart:
        """Build the stratigraphic chart from one well's tops (default: the type well), ordered by
        depth. Picks of different depth types are ordered through the well's trajectory."""
        if self.chart is not None and not replace:
            raise ProjectError("the project already has a stratigraphic chart; pass replace=True to rebuild it "
                               "(attached data would be detached)" if self.surfaces else
                               "the project already has a stratigraphic chart; pass replace=True to rebuild it")
        w = well or self.type_well
        if w is None or w not in self.wells:
            raise ProjectError("choose the well to build the chart from (type well)")
        rec = self.wells[w]
        if len(rec.tops) < 2:
            raise ProjectError(f"{w} has {len(rec.tops)} tops; a chart needs at least two (a top and a base)")
        types = {t.depth_type for t in rec.tops.values()}
        if len(types) == 1:
            picks = [(t.name, t.depth) for t in rec.tops.values()]
        elif rec.trajectory is not None:
            picks = [(t.name, _depth_to_position(w, rec.trajectory, t.depth, t.depth_type)[0]) for t in rec.tops.values()]
        else:
            raise ProjectError(f"{w}: tops are given in different depth types ({sorted(types)}); a header and a "
                               "deviation survey are needed to put them in order")
        try:
            self.chart = StratChart.from_picks(picks)
        except StratError as exc:
            raise ProjectError(f"{w}: {exc}") from exc
        self.type_well = w
        self.surfaces, self.primary = {}, {}
        return self.chart

    def create_chart(self, interval_names, base_name: str) -> StratChart:
        """Define the stratigraphy by hand (no type well): intervals top to bottom, and the base."""
        if self.chart is not None:
            raise ProjectError("the project already has a stratigraphic chart")
        try:
            self.chart = StratChart(list(interval_names), base_name)
        except StratError as exc:
            raise ProjectError(str(exc)) from exc
        return self.chart

    def rename_horizon(self, uid: str, new: str, *, keep_alias: bool = True):
        """Rename an interval (its top pick) or the base; picks in every well follow."""
        if self.chart is None:
            raise ProjectError("no stratigraphic chart")
        old = self.chart.rename(uid, new, keep_alias=keep_alias)
        new = new.strip()
        for rec in self.wells.values():
            if old in rec.tops:
                t = rec.tops.pop(old)
                rec.tops[new] = RawTop(new, t.depth, t.depth_type)

    def horizon_uses(self, uid: str) -> list:
        """What is attached to a horizon: wells with a pick of its name, surfaces."""
        h = self.chart.horizon(uid)
        out = [f"pick in {r.name}" for r in self.wells.values() if h.name in r.tops]
        out += [f"surface {lb}" for lb in self.surfaces.get(uid, {})]
        out += [f"variogram {v}" for v in self.variograms.get(uid, {})]
        out += [f"map {v}" for v in self.maps.get(uid, {})]
        return out

    def remove_interval(self, uid: str):
        uses = self.horizon_uses(uid)
        if uses:
            raise ProjectError(f"cannot remove: data is attached to its top ({', '.join(uses)}); "
                               "remove or move that data first")
        self.chart.remove_interval(uid)

    @staticmethod
    def rec_order_issues(rec: "WellRecord", order: dict) -> list:
        picks = [t for t in rec.tops.values() if t.name in order]
        types = {t.depth_type for t in picks}
        if len(picks) < 2:
            return []
        if len(types) == 1:
            dep = {t.name: t.depth for t in picks}
        elif rec.trajectory is not None:
            dep = {t.name: _depth_to_position(rec.name, rec.trajectory, t.depth, t.depth_type)[0] for t in picks}
        else:
            return []
        seq = sorted(dep, key=lambda n: order[n])
        return [f"{rec.name}: {b} ({dep[b]:g}) is not below {a} ({dep[a]:g}) although the chart puts it deeper"
                for a, b in zip(seq, seq[1:]) if dep[b] <= dep[a]]

    def order_issues(self) -> list:
        """Wells whose picks are not in the chart's order (or tie in depth). Faulted or repeated
        sections do this legitimately; a wrong name mapping does it too."""
        if self.chart is None:
            return []
        order = {h.name: h.index for h in self.chart.horizons()}
        return [m for rec in self.wells.values() for m in self.rec_order_issues(rec, order)]

    def map_picks(self, parsed: dict, top_map: dict) -> dict:
        """Rename picks read from a file to the chart's horizon names. `top_map`: file pick name ->
        horizon uid, or "ignore". Two picks of one well landing on one horizon is an error. The
        file's spelling is remembered as an alias."""
        if self.chart is None:
            return parsed
        names = {h.uid: h.name for h in self.chart.horizons()}
        out = {}
        for w, picks in parsed.items():
            seen, new = {}, []
            for t in picks:
                target = top_map.get(t.name)
                if target is None:
                    raise ProjectError(f"top {t.name!r} is not mapped to a horizon")
                if target == "ignore":
                    continue
                if target not in names:
                    raise ProjectError(f"top {t.name!r}: unknown horizon {target!r}")
                if target in seen:
                    raise ProjectError(f"{w}: picks {seen[target]!r} and {t.name!r} both map to {names[target]!r}")
                seen[target] = t.name
                new.append(RawTop(names[target], t.depth, t.depth_type))
            out[w] = new
        return out

    def remember_aliases(self, top_map: dict):
        if self.chart is None:
            return
        for fname, uid in top_map.items():
            if uid not in ("ignore", None) and self.chart.match_name(fname) is None:
                self.chart.add_alias(uid, fname)

    # -- surfaces on horizons ----------------------------------------------------------------------------------------
    def conflicts_surface(self, uid: str, label: str) -> list:
        have = self.surfaces.get(uid, {})
        if not have:
            return []
        h = self.chart.horizon(uid)
        prim = self.primary.get(uid)
        return [Conflict(f"chart|surface|{uid}", "-", "surface", h.label,
                         f"{prim} (+{len(have) - 1} other)" if len(have) > 1 else str(prim),
                         label, identical=False, allowed=("replace", "skip", "suffix"),
                         reserved=frozenset(have), base=label)]

    def attach_surface(self, uid: str, surface, label: str, actions: dict | None = None) -> ImportResult:
        if self.chart is None:
            raise ProjectError("define the stratigraphic chart before attaching surfaces")
        self.chart.horizon(uid)       # raises if unknown
        res = ImportResult("surface")
        acts = self._decide(self.conflicts_surface(uid, label), actions)
        a = acts.get(f"chart|surface|{uid}")
        h = self.chart.horizon(uid)
        bucket = self.surfaces.setdefault(uid, {})
        if a is None:
            bucket[label] = surface
            self.primary[uid] = label
            res.applied.append(f"{label} -> {h.label}")
        elif a.mode == "skip":
            res.applied.append(f"{label}: not attached (kept {self.primary[uid]})")
        elif a.mode == "suffix":
            bucket[label + a.suffix] = surface
            res.applied.append(f"{label} -> {h.label} as alternative {label + a.suffix!r}")
            res.notes.append(f"{h.label}: {self.primary[uid]!r} remains the surface the model uses")
        else:
            bucket.pop(self.primary[uid], None)
            bucket[label] = surface
            self.primary[uid] = label
            res.applied.append(f"{label} -> {h.label} (replaced)")
        return res

    # -- variograms on intervals ------------------------------------------------------------------------------------
    def _interval_label(self, uid: str) -> str:
        if self.chart is None:
            raise ProjectError("define the stratigraphic chart before attaching variograms")
        try:
            return self.chart.interval(uid).name
        except StratError as exc:
            raise ProjectError(f"{uid!r} is not an interval of the chart (variograms belong to intervals, not the base)") from exc

    def conflicts_variogram(self, uid: str, variable: str, label: str) -> list:
        name = self._interval_label(uid)
        have = self.variograms.get(uid, {}).get(variable, {})
        if not have:
            return []
        prim = self.var_primary[(uid, variable)]
        return [Conflict(f"chart|variogram|{uid}|{variable}", "-", "variogram", f"{name} / {variable}",
                         f"{prim} (+{len(have) - 1} other)" if len(have) > 1 else str(prim), label,
                         identical=False, allowed=("replace", "skip", "suffix"), reserved=frozenset(have), base=label)]

    def attach_variogram(self, uid: str, variable: str, model, label: str, actions: dict | None = None,
                         meta: dict | None = None) -> ImportResult:
        name = self._interval_label(uid)
        if not variable or not variable.strip():
            raise ProjectError("a variogram needs the name of the property it describes")
        variable = variable.strip()
        try:
            model.check()
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        res = ImportResult("variogram")
        confl = self.conflicts_variogram(uid, variable, label)
        acts = self._decide(confl, actions)
        a = acts.get(confl[0].key) if confl else None
        bucket = self.variograms.setdefault(uid, {}).setdefault(variable, {})
        key = (uid, variable)
        what = f"{label} -> {name} / {variable}"

        def meta_for(lab):
            self.var_meta.pop((uid, variable, lab), None)
            if meta:
                self.var_meta[(uid, variable, lab)] = dict(meta)

        if a is None:
            bucket[label] = model
            meta_for(label)
            self.var_primary[key] = label
            res.applied.append(what)
        elif a.mode == "skip":
            res.applied.append(f"{label}: not attached (kept {self.var_primary[key]})")
        elif a.mode == "suffix":
            bucket[label + a.suffix] = model
            meta_for(label + a.suffix)
            res.applied.append(f"{what} as alternative {label + a.suffix!r}")
            res.notes.append(f"{name} / {variable}: {self.var_primary[key]!r} remains the variogram the model uses")
        else:
            bucket.pop(self.var_primary[key], None)
            self.var_meta.pop((uid, variable, self.var_primary[key]), None)
            bucket[label] = model
            meta_for(label)
            self.var_primary[key] = label
            res.applied.append(what + " (replaced)")
        return res

    def variogram_for(self, uid: str, variable: str):
        """The variogram the property step uses for this interval and property, or None."""
        lab = self.var_primary.get((uid, variable))
        return None if lab is None else self.variograms[uid][variable][lab]

    # -- property maps on intervals ----------------------------------------------------------------------------------
    def conflicts_map(self, uid: str, variable: str, label: str) -> list:
        name = self._interval_label(uid)
        have = self.maps.get(uid, {}).get(variable, {})
        if not have:
            return []
        prim = self.map_primary[(uid, variable)]
        return [Conflict(f"chart|map|{uid}|{variable}", "-", "map", f"{name} / {variable}",
                         f"{prim} (+{len(have) - 1} other)" if len(have) > 1 else str(prim), label,
                         identical=False, allowed=("replace", "skip", "suffix"), reserved=frozenset(have), base=label)]

    def attach_map(self, uid: str, pmap, label: str, actions: dict | None = None) -> ImportResult:
        name = self._interval_label(uid)
        variable = (pmap.variable or "").strip()
        if not variable:
            raise ProjectError("a map needs the name of the property it describes")
        if not (pmap.unit or "").strip():
            raise ProjectError(f"{variable}: a map needs a declared unit (fraction, mD, m, ...); none is assumed")
        if not label or not label.strip():
            raise ProjectError("give the map a name")
        res = ImportResult("map")
        confl = self.conflicts_map(uid, variable, label)
        acts = self._decide(confl, actions)
        a = acts.get(confl[0].key) if confl else None
        bucket = self.maps.setdefault(uid, {}).setdefault(variable, {})
        key = (uid, variable)
        what = f"{label} -> {name} / {variable}"
        if a is None:
            bucket[label] = pmap
            self.map_primary[key] = label
            res.applied.append(what)
        elif a.mode == "skip":
            res.applied.append(f"{label}: not attached (kept {self.map_primary[key]})")
        elif a.mode == "suffix":
            bucket[label + a.suffix] = pmap
            res.applied.append(f"{what} as alternative {label + a.suffix!r}")
            res.notes.append(f"{name} / {variable}: {self.map_primary[key]!r} remains the map the model uses")
        else:
            bucket.pop(self.map_primary[key], None)
            bucket[label] = pmap
            self.map_primary[key] = label
            res.applied.append(what + " (replaced)")
        return res

    # -- faults ------------------------------------------------------------------------------------------------------
    def conflicts_faults(self, new: list) -> list:
        out = []
        for f in new:
            o = self.faults.get(f.name)
            if o is not None:
                out.append(Conflict(f"fault|{f.name}", "-", "fault", f.name, o.label(), f.label(),
                                    identical=o.same_geometry(f), allowed=("replace", "skip", "suffix"),
                                    reserved=frozenset(self.faults) | frozenset(x.name for x in new), base=f.name))
        return out

    def attach_faults(self, new: list, actions: dict | None = None) -> ImportResult:
        names = [f.name for f in new]
        if len(set(names)) != len(names):
            raise ProjectError("two faults with the same name in one import")
        res = ImportResult("fault_sticks")
        confl = {c.key: c for c in self.conflicts_faults(new)}
        acts = self._decide(list(confl.values()), actions)
        for f in new:
            a = acts.get(f"fault|{f.name}")
            if a is None:
                self.faults[f.name] = f
                res.applied.append(f"{f.name}: {f.label()}")
            elif a.mode == "skip":
                res.applied.append(f"{f.name}: not imported (kept the existing fault)")
            elif a.mode == "suffix":
                g = replace(f, name=f.name + a.suffix)
                self.faults[g.name] = g
                res.applied.append(f"{f.name}: imported as {g.name!r}")
            else:
                self.faults[f.name] = f
                self._forget_fault(f.name)
                res.applied.append(f"{f.name}: replaced")
        return res

    def set_boundary(self, polygon):
        from .grid.boundary import validate_polygon
        self.boundary = validate_polygon(polygon)

    def remove_fault(self, name: str):
        if name not in self.faults:
            raise ProjectError(f"no fault called {name!r}")
        del self.faults[name]
        self._forget_fault(name)

    def _forget_fault(self, name: str):
        self.fault_depths.pop(name, None)
        self.fault_transmissibility.pop(name, None)
        self.fault_throw_seen.pop(name, None)
        self.fault_interactions = [i for i in self.fault_interactions if name not in i.key]

    def set_fault_depth(self, name: str, depth: float):
        if name not in self.faults:
            raise ProjectError(f"no fault called {name!r}")
        if not (isinstance(depth, (int, float)) and depth == depth and abs(depth) != float("inf")):
            raise ProjectError("the depth of a fault trace must be a number")
        if self.fault_depths.get(name) != float(depth):
            self.fault_depths[name] = float(depth)
            self.fault_interactions = [i for i in self.fault_interactions if name not in i.key]

    def set_data_checks(self, setup):
        from .datacheck import DataCheckSetup
        if setup is not None and not isinstance(setup, DataCheckSetup):
            raise ProjectError("the data-check setup must be a DataCheckSetup")
        self.data_checks = setup

    def add_sgr_method(self, method):
        from .fault_relations import RelationError, check_new_method
        try:
            check_new_method(method, self.sgr_methods)
        except RelationError as exc:
            raise ProjectError(str(exc)) from None
        self.sgr_methods[method.name] = method        # same name replaces: choices keep their own copy of the expression

    def remove_sgr_method(self, name: str):
        if name not in self.sgr_methods:
            raise ProjectError(f"no SGR method called {name!r}")
        del self.sgr_methods[name]

    def set_proportion_map(self, pm):
        from .proportion_maps import ProportionMap
        if not isinstance(pm, ProportionMap):
            raise ProjectError("expected a ProportionMap")
        self.proportion_maps[pm.interval] = pm

    def remove_proportion_map(self, interval):
        if interval not in self.proportion_maps:
            raise ProjectError(f"no proportion map for interval {interval!r}")
        del self.proportion_maps[interval]

    def set_realization_plan(self, plan):
        from .realizations import RealizationPlan
        if plan is not None and not isinstance(plan, RealizationPlan):
            raise ProjectError("the realization plan must be a RealizationPlan")
        self.realization_plan = plan

    def set_facies_sim(self, setup):
        from .facies_sim import FaciesSimSetup
        if setup is not None and not isinstance(setup, FaciesSimSetup):
            raise ProjectError("the facies simulation setup must be a FaciesSimSetup")
        self.facies_sim = setup

    def set_proportions(self, setup):
        from .proportions import ProportionSetup
        if setup is not None and not isinstance(setup, ProportionSetup):
            raise ProjectError("the proportion setup must be a ProportionSetup")
        self.proportions = setup

    def set_facies_setup(self, setup):
        from .facies import FaciesSetup
        if setup is not None and not isinstance(setup, FaciesSetup):
            raise ProjectError("the facies setup must be a FaciesSetup")
        self.facies_setup = setup

    def set_fault_transmissibility(self, name: str, choice):
        from .fault_transmissibility import FaultTransmissibility
        if name not in self.faults:
            raise ProjectError(f"no fault called {name!r}")
        if not isinstance(choice, FaultTransmissibility):
            raise ProjectError("the transmissibility choice must be a FaultTransmissibility")
        self.fault_transmissibility[name] = choice

    def refresh_fault_interactions(self, reach: float):
        """Find where the fault traces (each at its chosen depth) cross or meet, keeping earlier answers where the
        same pair still meets in the same way. Returns (interactions, dropped answers, problems)."""
        from .fault_interactions import carry_answers, find_interactions
        traces, problems = {}, []
        for n, f in self.faults.items():
            if n not in self.fault_depths:
                problems.append(f"{n}: no trace depth chosen")
                continue
            tr = f.trace_at(self.fault_depths[n])
            if len(tr.xy) < 2:
                problems.append(f"{n}: no trace at depth {self.fault_depths[n]:g} (fewer than two sticks reach it)")
                continue
            traces[n] = tr.xy
        new = find_interactions(traces, reach)
        new, dropped = carry_answers(new, self.fault_interactions, reach)
        self.fault_reach = float(reach)
        self.fault_interactions = new
        return new, dropped, problems

    def confirm_fault_interaction(self, key: tuple, answer: str):
        from .fault_interactions import set_answer
        self.fault_interactions = set_answer(self.fault_interactions, tuple(key), answer)

    def fault_age_order(self) -> list:
        """Fault names, oldest first. Refuses while any interaction is unconfirmed or traces lack a depth."""
        from .fault_interactions import age_order, unconfirmed
        lacking = [n for n in self.faults if n not in self.fault_depths]
        if lacking:
            raise ProjectError("choose a trace depth for: " + ", ".join(lacking))
        if self.fault_reach is None and len(self.faults) > 1:
            raise ProjectError("the fault interactions have not been checked yet")
        open_ = unconfirmed(self.fault_interactions)
        if open_:
            raise ProjectError("confirm how these faults relate: " + "; ".join(i.describe() for i in open_))
        return age_order(list(self.faults), self.fault_interactions)

    def map_for(self, uid: str, variable: str):
        lab = self.map_primary.get((uid, variable))
        return None if lab is None else self.maps[uid][variable][lab]

    # -- link handling ---------------------------------------------------------------------------
    def _check_links(self, links: dict):
        used = {}
        for fw, l in links.items():
            if l.mode == "existing":
                if l.name not in self.wells:
                    raise ProjectError(f"{fw}: cannot connect to {l.name!r}: no such well in the project")
            elif l.mode == "new":
                if not l.name or l.name in self.wells:
                    raise ProjectError(f"{fw}: a new well needs an unused name; {l.name!r} "
                                       f"{'is empty' if not l.name else 'already exists'}")
            else:
                raise ProjectError(f"{fw}: unknown link mode {l.mode!r}")
            if l.name in used:
                raise ProjectError(f"{fw} and {used[l.name]} would both go to well {l.name!r}")
            used[l.name] = fw

    def _atomic(self, links, fn) -> ImportResult:
        self._check_links(links)
        touched = {l.name for l in links.values()}
        snap = {n: copy.deepcopy(self.wells[n]) for n in touched if n in self.wells}
        created = [n for n in touched if n not in self.wells]
        try:
            return fn()
        except Exception:
            for n in created:
                self.wells.pop(n, None)
            self.wells.update(snap)
            raise

    def _target(self, link: WellLink) -> WellRecord:
        if link.mode == "new":
            self.wells[link.name] = WellRecord(link.name)
        return self.wells[link.name]

    def _rec(self, link: WellLink):
        """The well a link points to, or None while it is still to be created."""
        return self.wells.get(link.name) if link.mode == "existing" else None

    def _revalidate(self, rec: WellRecord):
        if rec.trajectory is None:
            return
        if rec.tops:
            rec.resolved_tops()
        if rec.core:
            rec.resolved_core()

    def _decide(self, conflicts, actions) -> dict:
        actions = actions or {}
        out = {}
        for c in conflicts:
            problem = check_action(c, actions.get(c.key))
            if problem:
                raise ProjectError(problem)
            out[c.key] = actions[c.key]
        return out

    # -- conflict detection ---------------------------------------------------------------------------
    def conflicts_headers(self, headers: dict, links: dict) -> list:
        out = []
        for fw, h in headers.items():
            rec = self._rec(links[fw])
            if rec is not None and rec.header is not None:
                o = rec.header
                out.append(Conflict(f"{rec.name}|header|header", rec.name, "header", "header",
                                   f"x {o['x']:.2f}, y {o['y']:.2f}, KB {o['kb']:.2f} m",
                                   f"x {h['x']:.2f}, y {h['y']:.2f}, KB {h['kb']:.2f} m",
                                   identical=all(o[k] == h[k] for k in ("x", "y", "kb")),
                                   allowed=("replace", "skip")))
        return out

    def conflicts_surveys(self, surveys: dict, links: dict) -> list:
        out = []
        for fw, sv in surveys.items():
            rec = self._rec(links[fw])
            if rec is not None and rec.survey is not None:
                o = rec.survey
                same = (o.md.shape == sv.md.shape and np.array_equal(o.md, sv.md) and np.array_equal(o.incl, sv.incl)
                        and np.array_equal(o.azim, sv.azim))
                out.append(Conflict(f"{rec.name}|survey|survey", rec.name, "survey", "deviation survey",
                                   f"{o.md.size} stations, MD {o.md[0]:g}-{o.md[-1]:g} m",
                                   f"{sv.md.size} stations, MD {sv.md[0]:g}-{sv.md[-1]:g} m",
                                   identical=same, allowed=("replace", "skip")))
        return out

    @staticmethod
    def _desc(t: RawTop) -> str:
        return f"{t.depth:.2f} m {t.depth_type.upper()}"

    def conflicts_tops(self, tops: dict, links: dict) -> list:
        out = []
        for fw, picks in tops.items():
            rec = self._rec(links[fw])
            if rec is None:
                continue
            reserved = frozenset(rec.tops) | {t.name for t in picks}
            for t in picks:
                if t.name not in rec.tops:
                    continue
                o = rec.tops[t.name]
                same = False
                if o.depth_type == t.depth_type:
                    same = abs(o.depth - t.depth) < 1e-6
                elif rec.trajectory is not None:
                    try:
                        same = abs(_depth_to_position(rec.name, rec.trajectory, o.depth, o.depth_type)[0]
                                   - _depth_to_position(rec.name, rec.trajectory, t.depth, t.depth_type)[0]) < 1e-3
                    except WellDataError:
                        same = False
                out.append(Conflict(f"{rec.name}|top|{t.name}", rec.name, "top", t.name,
                                   self._desc(o), self._desc(t), identical=same, reserved=reserved))
        return out

    def conflicts_logs(self, logs: LogSet, fw: str, link: WellLink) -> list:
        rec = self._rec(link)
        if rec is None:
            return []
        out = []
        reserved = frozenset(c for s in rec.logs.values() for c in s.curves) | frozenset(logs.curves)
        for name, vals in logs.curves.items():
            nr = _curve_range(logs.md, vals)
            if nr is None:
                continue
            for src, ls in rec.logs.items():
                for en, ev in ls.curves.items():
                    if en.casefold() != name.casefold():
                        continue
                    er = _curve_range(ls.md, ev)
                    if er is None or er[1] < nr[0] or nr[1] < er[0]:
                        continue
                    lap = min(er[1], nr[1]) - max(er[0], nr[0])
                    same = (ls.md.shape == logs.md.shape and np.array_equal(ls.md, logs.md)
                            and np.array_equal(np.asarray(ev, float), np.asarray(vals, float), equal_nan=True))
                    out.append(Conflict(f"{rec.name}|curve|{name}", rec.name, "curve", name,
                                       f"{en} in {src}: {er[0]:.1f}-{er[1]:.1f} m MD",
                                       f"{name}: {nr[0]:.1f}-{nr[1]:.1f} m MD (overlap {lap:.1f} m)",
                                       identical=same, reserved=reserved))
                    break
                else:
                    continue
                break
        return out

    def conflicts_core(self, core: dict, fw_links: dict) -> list:
        out = []
        for fw, rc in core.items():
            rec = self._rec(fw_links[fw])
            if rec is None:
                continue
            reserved = frozenset(c for s in rec.core.values() for c in s.columns) | frozenset(rc.columns)
            for cname, vals in rc.columns.items():
                for src, ec in rec.core.items():
                    en = next((e for e in ec.columns if e.casefold() == cname.casefold()), None)
                    if en is None:
                        continue
                    ed = self._core_md(rec, ec)
                    nd = self._core_md(rec, rc)
                    if ed is None or nd is None:       # depth types differ and no trajectory: cannot compare
                        out.append(Conflict(f"{rec.name}|core|{cname}", rec.name, "core", cname,
                                           f"{en} in {src} ({ec.depth_type})", f"{cname} ({rc.depth_type}) - depths not comparable yet",
                                           reserved=reserved))
                        break
                    er, nr = _rng(ed), _rng(nd)
                    if er is None or nr is None or er[1] < nr[0] or nr[1] < er[0]:
                        continue
                    same_depth = int(sum(np.any(np.abs(ed - d) < DEPTH_TOL) for d in nd))
                    ident = (ed.shape == nd.shape and np.allclose(np.sort(ed), np.sort(nd), atol=DEPTH_TOL)
                             and np.array_equal(np.asarray(ec.columns[en], float)[np.argsort(ed)],
                                                np.asarray(vals, float)[np.argsort(nd)], equal_nan=True))
                    out.append(Conflict(f"{rec.name}|core|{cname}", rec.name, "core", cname,
                                       f"{en} in {src}: {len(ed)} samples, {er[0]:.1f}-{er[1]:.1f} m",
                                       f"{cname}: {len(nd)} samples, {nr[0]:.1f}-{nr[1]:.1f} m; {same_depth} at the same depth",
                                       identical=ident, reserved=reserved))
                    break
        return out

    @staticmethod
    def _core_md(rec: WellRecord, rc: RawCore):
        """Core depths as MD if that can be determined, else None."""
        if rc.depth_type == "md":
            return np.asarray(rc.depth, float)
        tr = rec.trajectory
        if tr is None:
            return None
        try:
            return np.array([_depth_to_position(rec.name, tr, float(d), rc.depth_type)[0] for d in rc.depth])
        except WellDataError:
            return None

    # -- imports ------------------------------------------------------------------------------------------
    def apply_headers(self, headers: dict, links: dict, actions: dict | None = None) -> ImportResult:
        res = ImportResult("well_headers")
        acts = self._decide(self.conflicts_headers(headers, links), actions)

        def go():
            for fw, h in headers.items():
                l = links[fw]
                existing = self._rec(l) is not None and self.wells[l.name].header is not None
                if existing and acts[f"{l.name}|header|header"].mode == "skip":
                    res.applied.append(f"{fw} -> {l.name}: header kept (new one skipped)")
                    continue
                rec = self._target(l)
                rec.header = dict(h)
                self._revalidate(rec)
                if existing:
                    res.applied.append(f"{fw} -> updated header of {l.name}")
                    res.notes.append(f"{l.name}: header replaced; trajectory and pick positions recomputed")
                else:
                    res.applied.append(f"{fw} -> {'new well' if l.mode == 'new' else 'header added to'} {l.name}")
            return res
        return self._atomic(links, go)

    def apply_surveys(self, surveys: dict, links: dict, actions: dict | None = None) -> ImportResult:
        res = ImportResult("deviation")
        acts = self._decide(self.conflicts_surveys(surveys, links), actions)

        def go():
            for fw, sv in surveys.items():
                l = links[fw]
                existing = self._rec(l) is not None and self.wells[l.name].survey is not None
                if existing and acts[f"{l.name}|survey|survey"].mode == "skip":
                    res.applied.append(f"{fw} -> {l.name}: survey kept (new one skipped)")
                    continue
                rec = self._target(l)
                rec.survey = sv
                self._revalidate(rec)
                res.applied.append(f"{fw} -> {l.name}: survey of {sv.md.size} stations" + (" (replaced)" if existing else ""))
                if rec.header is None:
                    res.notes.append(f"{l.name}: no header yet - trajectory cannot be placed until one is imported")
            return res
        return self._atomic(links, go)

    def apply_tops(self, tops: dict, links: dict, actions: dict | None = None) -> ImportResult:
        res = ImportResult("tops")
        acts = self._decide(self.conflicts_tops(tops, links), actions)

        def go():
            for fw, picks in tops.items():
                l = links[fw]
                rec = self._target(l)
                n_added = 0
                for t in picks:
                    a = acts.get(f"{l.name}|top|{t.name}")
                    if a is None or a.mode == "replace":
                        if a is not None:
                            res.notes.append(f"{l.name}: top {t.name} replaced")
                        rec.tops[t.name] = t
                    elif a.mode == "suffix":
                        nm = t.name + a.suffix
                        rec.tops[nm] = RawTop(nm, t.depth, t.depth_type)
                        res.notes.append(f"{l.name}: top {t.name} imported as {nm}")
                    else:
                        res.notes.append(f"{l.name}: top {t.name} skipped (kept existing)")
                        continue
                    n_added += 1
                self._revalidate(rec)
                res.applied.append(f"{fw} -> {l.name}: {n_added} of {len(picks)} tops imported")
                if rec.trajectory is None:
                    res.notes.append(f"{l.name}: picks stored but unplaced until a header and survey exist "
                                     f"(missing: {', '.join(rec.missing())})")
            return res
        return self._atomic(links, go)

    def _free_key(self, d: dict, key: str) -> str:
        if key not in d:
            return key
        n = 2
        while f"{key}#{n}" in d:
            n += 1
        return f"{key}#{n}"

    def apply_core(self, core: dict, links: dict, actions: dict | None = None, source: str = "core") -> ImportResult:
        res = ImportResult("core")
        acts = self._decide(self.conflicts_core(core, links), actions)

        def go():
            for fw, rc in core.items():
                l = links[fw]
                rec = self._target(l)
                cols = {}
                for cname, vals in rc.columns.items():
                    a = acts.get(f"{l.name}|core|{cname}")
                    if a is None:
                        cols[cname] = vals
                    elif a.mode == "skip":
                        res.notes.append(f"{l.name}: core column {cname} skipped (kept existing)")
                    elif a.mode == "suffix":
                        cols[cname + a.suffix] = vals
                        res.notes.append(f"{l.name}: core column {cname} imported as {cname + a.suffix}")
                    else:
                        for src, ec in list(rec.core.items()):
                            for en in [e for e in ec.columns if e.casefold() == cname.casefold()]:
                                ec = RawCore(ec.depth, ec.depth_type, {k: v for k, v in ec.columns.items() if k != en})
                                rec.core[src] = ec
                            if not rec.core[src].columns:
                                del rec.core[src]
                        cols[cname] = vals
                        res.notes.append(f"{l.name}: core column {cname} replaced")
                if cols:
                    rec.core[self._free_key(rec.core, source)] = RawCore(rc.depth, rc.depth_type, cols)
                self._revalidate(rec)
                res.applied.append(f"{fw} -> {l.name}: {len(cols)} of {len(rc.columns)} core columns, {rc.depth.size} samples")
                if rec.trajectory is None:
                    res.notes.append(f"{l.name}: core stored but unplaced (missing: {', '.join(rec.missing())})")
            return res
        return self._atomic(links, go)

    def apply_logs(self, source: str, logs: LogSet, fw: str, link: WellLink, actions: dict | None = None) -> ImportResult:
        res = ImportResult("las")
        acts = self._decide(self.conflicts_logs(logs, fw, link), actions)

        def go():
            rec = self._target(link)
            curves, units = {}, {}
            for name, vals in logs.curves.items():
                a = acts.get(f"{link.name}|curve|{name}")
                if a is None:
                    new = name
                elif a.mode == "skip":
                    res.notes.append(f"{link.name}: curve {name} skipped (kept existing)")
                    continue
                elif a.mode == "suffix":
                    new = name + a.suffix
                    res.notes.append(f"{link.name}: curve {name} imported as {new}")
                else:
                    for src, ls in list(rec.logs.items()):
                        keep = {k: v for k, v in ls.curves.items() if k.casefold() != name.casefold()}
                        if len(keep) != len(ls.curves):
                            if keep:
                                rec.logs[src] = LogSet(ls.md, keep, {k: ls.units.get(k, "") for k in keep})
                            else:
                                del rec.logs[src]
                    new = name
                    res.notes.append(f"{link.name}: curve {name} replaced")
                curves[new] = vals
                units[new] = logs.units.get(name, "")
            if curves:
                rec.logs[self._free_key(rec.logs, source)] = LogSet(logs.md, curves, units)
            res.applied.append(f"{fw} -> {link.name}: {len(curves)} of {len(logs.curves)} curves from {source}")
            return res
        return self._atomic({fw: link}, go)
