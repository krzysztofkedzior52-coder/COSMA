"""Facies simulation set-up and run: proportions (curves, optionally combined with maps) + hard facies + SIS or truncated Gaussian.

Nothing is defaulted: the method, its settings, the seed and - when proportion maps exist - what to do where a map has no data
and how many cells may be left empty by curve and map together are all entered by the user.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sis import SisError, SisSettings, simulate_facies
from .tgs import TgsError, TgsSettings, simulate_tgs

METHODS = ("sis", "tgs")


class FaciesSimError(ValueError):
    pass


@dataclass(frozen=True)
class FaciesSimSetup:
    method: str                    # 'sis' | 'tgs'
    sis: SisSettings | None
    tgs: TgsSettings | None
    seed: int
    outside_map: str | None = None        # 'curve' | 'stop'; needed when the project has proportion maps
    max_incompatible: float | None = None  # 0-1; needed when the project has proportion maps

    def __post_init__(self):
        if self.method not in METHODS:
            raise FaciesSimError(f"method must be one of {', '.join(METHODS)}")
        if self.method == "sis" and not isinstance(self.sis, SisSettings):
            raise FaciesSimError("SIS needs its settings")
        if self.method == "tgs" and not isinstance(self.tgs, TgsSettings):
            raise FaciesSimError("truncated Gaussian simulation needs its settings")
        if not (isinstance(self.seed, int) and not isinstance(self.seed, bool) and 0 <= self.seed < 2**31):
            raise FaciesSimError("the seed must be a whole number between 0 and 2^31 - 1")
        if self.outside_map not in (None, "curve", "stop"):
            raise FaciesSimError("outside_map must be 'curve' or 'stop'")
        if self.max_incompatible is not None and not (isinstance(self.max_incompatible, (int, float)) and 0 <= self.max_incompatible <= 1):
            raise FaciesSimError("max_incompatible must be a number between 0 and 1")


def to_state(s):
    if s is None:
        return None
    d = dict(method=s.method, seed=s.seed, outside_map=s.outside_map, max_incompatible=s.max_incompatible, sis=None, tgs=None)
    if s.sis is not None:
        d["sis"] = dict(max_neighbours=s.sis.max_neighbours, search_factor=s.sis.search_factor, levels=s.sis.levels)
    if s.tgs is not None:
        t = s.tgs
        d["tgs"] = dict(order=list(t.order), variogram=t.variogram, max_neighbours=t.max_neighbours, search_factor=t.search_factor,
                        levels=t.levels, gibbs_sweeps=t.gibbs_sweeps)
    return d


def from_state(d):
    if d is None:
        return None
    try:
        sis = SisSettings(**d["sis"]) if d.get("sis") else None
        tg = d.get("tgs")
        tgs = TgsSettings(tuple(tg["order"]), tg["variogram"], tg["max_neighbours"], tg["search_factor"], tg["levels"],
                          tg["gibbs_sweeps"]) if tg else None
        return FaciesSimSetup(d["method"], sis, tgs, d["seed"], d.get("outside_map"), d.get("max_incompatible"))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, (FaciesSimError, SisError, TgsError)):
            raise FaciesSimError(str(exc)) from None
        raise FaciesSimError(f"bad facies simulation setup: {exc}") from None


def prepare(project, model, setup: FaciesSimSetup):
    """-> (local proportions {interval: (nx, ny, nlayers, nfac)}, hard facies, notes). Raises FaciesSimError."""
    from .facies import FaciesError, hard_facies
    from .proportions import ProportionError, build_curves, combine_with_maps, layer_proportions
    if project.facies_setup is None:
        raise FaciesSimError("define the facies first (Facies menu)")
    if project.proportions is None:
        raise FaciesSimError("define and save the vertical proportion curves first (Facies menu)")
    notes = []
    try:
        res = build_curves(project, model, project.proportions)
        if not res.ok:
            raise FaciesSimError("the proportion curves have problems or conflicts; see Facies > Vertical proportion curves")
        maps = {k: m for k, m in project.proportion_maps.items() if k in model.interval_names}
        if maps:
            if setup.outside_map is None or setup.max_incompatible is None:
                raise FaciesSimError("the project has proportion maps: say what to do where a map has no data and how many cells "
                                     "may be left empty by curve and map together")
            props, text = combine_with_maps(model, res, maps, setup.outside_map, setup.max_incompatible)
            notes.append(text)
        else:
            props = layer_proportions(model, res)
        hard = hard_facies(project, project.facies_setup, model.grid)
    except (ProportionError, FaciesError) as exc:
        raise FaciesSimError(str(exc)) from None
    return props, hard, notes


def run(project, model, setup: FaciesSimSetup, prepared=None, seed=None, progress=None):
    """-> (SisResult, notes). Raises FaciesSimError with the reason for anything that is missing or inconsistent.
    ``prepared`` (the result of prepare) avoids rebuilding the proportions; ``seed`` overrides the setup's seed.
    ``progress(label, done, total)`` is called between intervals and may raise progress.Cancelled."""
    props, hard, notes = prepared if prepared is not None else prepare(project, model, setup)
    seed = setup.seed if seed is None else seed
    try:
        if setup.method == "sis":
            out = simulate_facies(project, model, setup.sis, props, hard, seed, progress)
        else:
            out = simulate_tgs(project, model, setup.tgs, props, hard, seed, progress)
    except (SisError, TgsError) as exc:
        raise FaciesSimError(str(exc)) from None
    return out, notes
