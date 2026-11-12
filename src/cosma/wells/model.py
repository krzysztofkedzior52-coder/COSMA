"""Well data containers. Metres, depth positive down below datum (TVDSS) unless named otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trajectory import Trajectory


@dataclass(frozen=True)
class Top:
    well: str
    name: str
    md: float
    tvdss: float
    x: float
    y: float


@dataclass(frozen=True)
class LogSet:
    md: np.ndarray                      # (n,) metres
    curves: dict                        # name -> (n,) float array, NaN = null
    units: dict = field(default_factory=dict)   # name -> unit string as in the file (informational)


@dataclass(frozen=True)
class CoreSamples:
    well: str
    md: np.ndarray                      # (n,) metres
    columns: dict                       # name -> (n,) array; porosity as fraction, permeability in mD


@dataclass
class Well:
    name: str
    trajectory: Trajectory
    logs: LogSet | None = None
    tops: list = field(default_factory=list)
    core: CoreSamples | None = None

    @property
    def x(self):
        return float(self.trajectory.x[0])

    @property
    def y(self):
        return float(self.trajectory.y[0])

    @property
    def kb(self):
        return self.trajectory.kb


@dataclass(frozen=True)
class Survey:
    """Deviation survey as received (metres, degrees); the trajectory is derived from it
    once the well has a header."""
    md: np.ndarray
    incl: np.ndarray
    azim: np.ndarray
    azimuth_reference: str
    convergence_deg: float | None = None


@dataclass(frozen=True)
class RawTop:
    """A pick as received: depth in metres of the stated type (md / tvd below KB / tvdss)."""
    name: str
    depth: float
    depth_type: str


@dataclass(frozen=True)
class RawCore:
    depth: np.ndarray
    depth_type: str
    columns: dict
