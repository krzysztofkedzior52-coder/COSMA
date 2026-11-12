"""Suggest where new single-layer data sits in the stratigraphic chart.

Two kinds of evidence, both shown to the user, neither decisive:
  * name: "Top_Seal.irap" -> top of Seal; "Base_Seal.irap" -> base of Seal, which is the
    same horizon as the top of the interval below (one object, see strat.py);
  * depth: the surface sampled at wells that have a pick for the horizon; the median
    absolute difference to the pick depths (TVDSS) says how well it fits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .strat import BASE_UID


@dataclass
class Suggestion:
    uid: str
    label: str
    name_match: bool = False
    misfit: float | None = None      # median |surface - pick|, metres
    bias: float | None = None        # median (surface - pick), metres
    n_wells: int = 0
    tol: float = 5.0

    @property
    def fits(self) -> bool:
        return self.misfit is not None and self.misfit <= self.tol

    @property
    def text(self) -> str:
        bits = []
        if self.name_match:
            bits.append("name matches")
        if self.misfit is None:
            bits.append("no well picks to compare")
        else:
            bits.append(f"{self.n_wells} well(s): surface {self.bias:+.1f} m vs picks (median |diff| {self.misfit:.1f} m)")
        return "; ".join(bits)


@dataclass
class Placement:
    candidates: list = field(default_factory=list)       # best first
    suggested: Suggestion | None = None
    warnings: list = field(default_factory=list)


_PREFIX = re.compile(r"^(top|tops|base|bs|bot|bottom)[\s_\-\.]+(.*)$", re.I)
_CAMEL = re.compile(r"^(Top|Tops|Base|Bs|Bot|Bottom)(?=[A-Z0-9])(.*)$")      # TopSeal, BaseRes - not "Basement"


def parse_hint(stem: str):
    """('top' | 'base', name part) from a file name stem."""
    m = _PREFIX.match(stem.strip()) or _CAMEL.match(stem.strip())
    if not m or not m.group(2).strip():
        return "top", stem.strip()
    return ("base" if m.group(1).lower() in ("base", "bs", "bot", "bottom") else "top"), m.group(2).strip()


def name_match_uid(chart, stem: str):
    """Horizon uid the file name points at, or None."""
    whole = chart.match_name(stem)
    if whole is not None:
        return whole
    mode, core = parse_hint(stem)
    uid = chart.match_name(core)
    if uid is None:
        return None
    if mode == "top" or uid == BASE_UID:
        return uid
    i = chart.interval_index(uid)
    hs = chart.horizons()
    return hs[i + 1].uid


def _pick_xyz(project, horizon):
    out = []
    for rec in project.wells.values():
        t = rec.tops.get(horizon.name)
        tr = rec.trajectory
        if t is None or tr is None:
            continue
        from .wells.io import _depth_to_position
        md, tvdss, x, y = _depth_to_position(rec.name, tr, t.depth, t.depth_type)
        out.append((x, y, tvdss))
    return out


def _tolerance(project, chart, horizon) -> float:
    """5 m, or a quarter of the thinner neighbouring interval at the type well if that is larger."""
    tw = project.wells.get(project.type_well) if project.type_well else None
    if tw is None or tw.trajectory is None:
        return 5.0
    from .wells.io import _depth_to_position
    hs = chart.horizons()
    z = {}
    for h in hs:
        t = tw.tops.get(h.name)
        if t is not None:
            z[h.index] = _depth_to_position(tw.name, tw.trajectory, t.depth, t.depth_type)[1]
    th = [abs(z[a] - z[b]) for a, b in ((horizon.index - 1, horizon.index), (horizon.index, horizon.index + 1))
          if a in z and b in z]
    return max(5.0, 0.25 * min(th)) if th else 5.0


def suggest_horizons(project, surface, stem: str) -> Placement:
    chart = project.chart
    named = name_match_uid(chart, stem)
    cands = []
    for h in chart.horizons():
        s = Suggestion(h.uid, h.label, name_match=(h.uid == named), tol=_tolerance(project, chart, h))
        pts = _pick_xyz(project, h)
        if pts:
            xy = np.array(pts)
            z = surface.sample(xy[:, 0], xy[:, 1])
            ok = np.isfinite(z)
            if ok.any():
                diff = z[ok] - xy[ok, 2]
                s.n_wells, s.bias, s.misfit = int(ok.sum()), float(np.median(diff)), float(np.median(np.abs(diff)))
        cands.append(s)
    # name match with no contradicting depth evidence first, then depth fit, then the rest by misfit
    cands.sort(key=lambda s: (not (s.name_match and (s.misfit is None or s.fits)), not s.fits,
                              s.misfit if s.misfit is not None else float("inf")))
    out = Placement(cands)
    best = cands[0]
    fitting = [s for s in cands if s.fits]
    if best.name_match and (best.misfit is None or best.fits):
        out.suggested = best
    elif best.fits and not best.name_match and (len(fitting) == 1 or fitting[1].misfit >= 2 * best.misfit):
        out.suggested = best
    for s in cands:
        if s.name_match and s.misfit is not None and not s.fits:
            out.warnings.append(f"The file name points at {s.label}, but the surface is {s.bias:+.1f} m from the picks "
                                f"there (tolerance {s.tol:.1f} m): wrong horizon, datum or units?")
    return out
