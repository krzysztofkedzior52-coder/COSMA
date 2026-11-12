"""Corner-point grid data model.

Conventions (see README): metres, depth positive down, COORD pillars I-fastest,
ZCORN shaped (nz, 2, ny, 2, nx, 2) in C order, properties shaped (nx, ny, nz).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frame import GridFrame


@dataclass
class CornerPointGrid:
    nx: int
    ny: int
    nz: int
    coord: np.ndarray  # (ny+1, nx+1, 6): LOCAL x,y,z top then x,y,z bottom per pillar
    zcorn: np.ndarray  # (nz, 2, ny, 2, nx, 2)
    actnum: np.ndarray  # (nx, ny, nz) int, 1 = active
    props: dict[str, np.ndarray] = field(default_factory=dict)  # each (nx, ny, nz)
    j_north: bool = True  # True: J increases toward +y (local)
    frame: GridFrame = field(default_factory=GridFrame)  # local -> map placement

    def __post_init__(self) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        if self.coord.shape != (ny + 1, nx + 1, 6):
            raise ValueError(f"coord shape {self.coord.shape}, expected {(ny + 1, nx + 1, 6)}")
        if self.zcorn.shape != (nz, 2, ny, 2, nx, 2):
            raise ValueError(f"zcorn shape {self.zcorn.shape}, expected {(nz, 2, ny, 2, nx, 2)}")
        if self.actnum.shape != (nx, ny, nz):
            raise ValueError(f"actnum shape {self.actnum.shape}, expected {(nx, ny, nz)}")
        for name, arr in self.props.items():
            if arr.shape != (nx, ny, nz):
                raise ValueError(f"property {name} shape {arr.shape}, expected {(nx, ny, nz)}")

    # -- geometry ---------------------------------------------------------
    def cell_corners(self, i: int, j: int, k: int) -> np.ndarray:
        """Return the 8 corners of cell (i, j, k) as an (8, 3) array.

        Corner order follows the ZCORN/Eclipse convention: top face first
        (SW, SE, NW, NE in index terms: (j0,i0), (j0,i1), (j1,i0), (j1,i1)),
        then bottom face in the same order. Pillars are interpolated linearly
        in z between their top and bottom points.
        """
        out = np.empty((8, 3))
        n = 0
        for t in (0, 1):
            for js in (0, 1):
                for is_ in (0, 1):
                    p = self.coord[j + js, i + is_]
                    z = self.zcorn[k, t, j, js, i, is_]
                    top = p[:3]
                    bot = p[3:]
                    dz = bot[2] - top[2]
                    f = 0.0 if dz == 0.0 else (z - top[2]) / dz
                    out[n] = top + f * (bot - top)
                    out[n, 2] = z
                    n += 1
        return out

    def cell_corners_map(self, i: int, j: int, k: int) -> np.ndarray:
        """Like cell_corners but with x, y in map coordinates (z unchanged)."""
        c = self.cell_corners(i, j, k).copy()
        c[:, 0], c[:, 1] = self.frame.to_map(c[:, 0], c[:, 1])
        return c

    def cell_volume(self, i: int, j: int, k: int) -> float:
        """Volume of a cell, as tetrahedra from the cell centroid to each of
        the 12 face triangles (each quad face split along one diagonal).

        Exact for convex cells with planar faces, which covers every M0 case
        (vertical pillars, planar horizons). It is an approximation for
        warped faces and for non-convex cells; M3 will replace it with a
        proper implementation and geometry QC.
        """
        c = self.cell_corners(i, j, k)
        # indices: top 0..3 (SW,SE,NW,NE), bottom 4..7 same order
        quads = [
            (0, 1, 3, 2),  # top
            (4, 6, 7, 5),  # bottom
            (0, 4, 5, 1),  # south (j0)
            (2, 3, 7, 6),  # north (j1)
            (0, 2, 6, 4),  # west (i0)
            (1, 5, 7, 3),  # east (i1)
        ]
        vol = 0.0
        centre = c.mean(axis=0)
        for q in quads:
            a, b, cc, d = (c[x] - centre for x in q)
            # two triangles per quad, signed tetra volume with the centre origin
            vol += abs(np.dot(a, np.cross(b, cc))) / 6.0
            vol += abs(np.dot(a, np.cross(cc, d))) / 6.0
        return float(vol)

    def total_pore_volume(self, poro_name: str = "PORO") -> float:
        """Sum of porosity * cell volume over active cells."""
        poro = self.props[poro_name]
        total = 0.0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    if self.actnum[i, j, k]:
                        total += float(poro[i, j, k]) * self.cell_volume(i, j, k)
        return total
