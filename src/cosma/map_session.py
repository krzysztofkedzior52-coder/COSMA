"""Logic behind the property-map dialog (no GUI): choose data, a variogram and a grid; compute
a kriged or thickness map; save it on the interval. Nothing is assumed: the grid, the
unit and the variogram are chosen by the user; the session only offers candidates and
describes what each choice means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .maps import MapError, MapGrid, PropertyMap, grid_from_surface, grid_from_wells, interval_bounds, krige_map, thickness_map, layer_averages
from .project import Project, ProjectError
from .variogram import VariogramModel
from .variogram_data import available_properties

UNITS = ("fraction", "mD", "m", "other")
AVG = " (layer average)"


@dataclass(frozen=True)
class VariogramChoice:
    key: str                 # variable key it is stored under
    label: str
    model: VariogramModel
    average: bool
    primary: bool

    @property
    def text(self) -> str:
        return f"{self.key} / {self.label}" + ("" if self.average else "  [point-scale: see warning]") + ("  (used)" if self.primary else "")


class MapSession:
    def __init__(self, project: Project, uid: str):
        if project.chart is None:
            raise ProjectError("define the stratigraphic chart first")
        project.chart.interval(uid)
        self.project, self.uid = project, uid
        self.result: PropertyMap | None = None
        self.notes: list = []

    @property
    def interval(self) -> str:
        return self.project.chart.interval(self.uid).name

    def properties(self, kind: str = "log") -> dict:
        return available_properties(self.project, kind)

    @staticmethod
    def key(variable: str, indicator: int | None) -> str:
        return variable if indicator is None else f"{variable}[{indicator}]"

    def variogram_choices(self, variable: str, indicator: int | None = None) -> list:
        """Stored variograms of this interval for the property: layer-average ones first (they are the
        right support for kriging averages), then point-scale ones."""
        k = self.key(variable, indicator)
        out = []
        for key, average in ((k + AVG, True), (k, False)):
            for lab, m in self.project.variograms.get(self.uid, {}).get(key, {}).items():
                out.append(VariogramChoice(key, lab, m, average, self.project.var_primary.get((self.uid, key)) == lab))
        return out

    def warning_for(self, choice: VariogramChoice) -> str | None:
        if choice.average:
            return None
        return ("This variogram was computed from raw log samples, but the map is kriged from one average per well. "
                "Averaging removes small-scale variance, so the nugget share is overstated: the map will be smoother than "
                "it should be and the kriging variance is on the wrong scale. Compute a 'layer averages' variogram if you can.")

    def surface_grid(self, which: str = "top") -> MapGrid:
        top, base = interval_bounds(self.project, self.uid)
        s = {"top": top, "base": base}[which] if which in ("top", "base") else None
        if s is None:
            raise MapError(f"the {which} horizon of {self.interval} has no surface to take the grid from")
        return grid_from_surface(s)

    def wells_grid(self, inc: float, margin: float) -> MapGrid:
        xy = []
        for rec in self.project.wells.values():
            tr = rec.trajectory
            if tr is not None:
                xy.append((tr.x[0], tr.y[0]))
        if not xy:
            raise MapError("no well with a header and survey to place a grid around")
        return grid_from_wells(np.array(xy), inc, margin)

    def compute(self, variable: str, choice: VariogramChoice, grid: MapGrid, unit: str, *, kind: str = "log", source=None,
                indicator: int | None = None, min_coverage: float = 0.5, wells=None, mask: str = "thickness",
                bounds=None) -> PropertyMap:
        if unit not in UNITS:
            raise MapError(f"unit must be one of {UNITS}: none is assumed")
        pm, notes = krige_map(self.project, self.uid, variable, choice.model, grid, kind=kind, source=source, indicator=indicator,
                              min_coverage=min_coverage, wells=wells, mask=mask, bounds=bounds, unit=unit)
        w = self.warning_for(choice)
        if w:
            notes.insert(0, w)
        pm.info["variogram_label"] = choice.label
        pm.info["variogram_key"] = choice.key
        pm.info["variogram_average"] = choice.average
        self.result, self.notes = pm, notes
        return pm

    def thickness(self, grid: MapGrid) -> PropertyMap:
        self.result, self.notes = thickness_map(self.project, self.uid, grid), []
        c = self.result.info.get("crossed_nodes", 0)
        if c:
            self.notes.append(f"{c} node(s) where the base surface is above the top surface were left undefined.")
        return self.result

    def save(self, label: str, actions: dict | None = None):
        if self.result is None:
            raise ProjectError("compute a map first")
        return self.project.attach_map(self.uid, self.result, label.strip() if label else label, actions)

    def conflicts(self, label: str) -> list:
        if self.result is None:
            return []
        return self.project.conflicts_map(self.uid, self.result.variable, label)
