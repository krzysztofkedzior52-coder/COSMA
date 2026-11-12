"""A built grid goes through a real OPM Flow engine (opm-simulators Python bindings).
Skipped when the wheel is not installed (pip install opm-simulators)."""

import numpy as np
import pytest

pytest.importorskip("opm.simulators")
import subprocess  # noqa: E402
import sys  # noqa: E402

from cosma.grid.boundary import grid_spec_from_cells  # noqa: E402
from cosma.grid.builder import Layering, build_grid  # noqa: E402
from cosma.io.grdecl import write_grdecl  # noqa: E402

DECK = """RUNSPEC
EQLDIMS
  1 /
DIMENS
  {nx} {ny} {nz} /
GAS
WATER
CO2STORE
METRIC
START
  1 JAN 2025 /
WELLDIMS
  5 20 1 5 /
TABDIMS
  1 1 40 20 /
GRID
INCLUDE
  'grid.grdecl' /
PROPS
SGWFN
 0.0 0.0 1.0 0.0
 0.2 0.0 0.5 0.0
 0.6 0.5 0.0 0.0
 0.8 1.0 0.0 0.0 /
ROCK
  100.0 4.5E-5 /
SOLUTION
EQUIL
  {datum} 160.0 1000.0 0.0 1000.0 0.0 1 0 0 /
RTEMP
  60 /
SCHEDULE
WELSPECS
  'INJ' 'G' 1 1 1* GAS /
/
COMPDAT
  'INJ' 1 1 1 {nz} OPEN 1* 1* 0.2 /
/
WCONINJE
  'INJ' GAS OPEN RATE 1000.0 /
/
TSTEP
  3*30 /
END
"""

_RUNNER = """
import sys, numpy as np
from opm.simulators import GasWaterSimulator
d = sys.argv[1]
sim = GasWaterSimulator(d + "/t.DATA", ["--output-dir=" + d, "--enable-terminal-output=false"])
sim.step_init()
np.save(d + "/vol.npy", np.asarray(sim.get_cell_volumes()))
sys.exit(int(sim.run()))
"""


def opm_volumes_and_run(folder):
    """Load the deck in a fresh process (OPM keeps global state, so one simulator per process),
    return the active-cell volumes OPM computed; fails if the run does not finish with exit 0."""
    r = subprocess.run([sys.executable, "-c", _RUNNER, str(folder)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return np.load(folder / "vol.npy")


def test_built_grid_volumes_match_opm_flow(tmp_path):
    nx, ny = 6, 4
    spec = grid_spec_from_cells(0, 0, 0, nx, ny, 100.0, 100.0)
    x = np.arange(nx + 1) * 100.0
    flat = lambda z: np.full((ny + 1, nx + 1), float(z))  # noqa: E731
    h0 = np.tile(1500 + 0.05 * x, (ny + 1, 1))               # dipping top
    h1 = np.tile(1530 + 0.05 * x, (ny + 1, 1)) + 0 * flat(0)
    h2 = np.tile(1500 + 0.12 * x, (ny + 1, 1)) + 60          # crosses h1 on the east: erosional h1
    grid, rep = build_grid(spec, [h0, h1, h2], ["conformable", "erosional"],
                           [Layering("fixed_count", n=3), Layering("follow_top", dz=10)])
    assert rep.thin_cells > 0  # the test must include pinched cells
    shape = (grid.nx, grid.ny, grid.nz)
    grid.props["PORO"] = np.full(shape, 0.2)
    for name in ("PERMX", "PERMY", "PERMZ"):
        grid.props[name] = np.full(shape, 100.0)
    write_grdecl(grid, tmp_path / "grid.grdecl")
    (tmp_path / "t.DATA").write_text(DECK.format(nx=grid.nx, ny=grid.ny, nz=grid.nz, datum=1500.0))

    opm_vol = opm_volumes_and_run(tmp_path)
    mine = [grid.cell_volume(i, j, k) for k in range(grid.nz) for j in range(grid.ny)
            for i in range(grid.nx) if grid.actnum[i, j, k]]
    assert opm_vol.size == len(mine) == int(grid.actnum.sum())
    assert np.allclose(opm_vol, mine, rtol=1e-5)


def test_faulted_grid_from_surfaces_runs_in_opm_flow(tmp_path):
    from cosma.faults import Fault
    from cosma.grid.fault_sampling import corner_depths
    from cosma.surfaces import Surface

    nx, ny = 6, 4
    spec = grid_spec_from_cells(0, 0, 0, nx, ny, 100.0, 100.0)
    n = 81
    X, Y = np.meshgrid(-200.0 + 25.0 * np.arange(n), -200.0 + 25.0 * np.arange(n), indexing="ij")
    surfs = [Surface(-200.0, -200.0, 25.0, 25.0, 0.0, 1, z0 + 0.05 * X + 20.0 * (X > 312.0)) for z0 in (1500.0, 1560.0)]
    fault = Fault("F", tuple(np.array([[312.0, y, z] for z in (0.0, 3000.0)]) for y in (-100.0, 500.0)))
    corners, (rep,) = corner_depths(spec, surfs, {"F": fault}, {"F": 1500.0})
    assert rep.has_throw
    grid, _ = build_grid(spec, corners, ["conformable"], [Layering("fixed_count", n=4)])
    shape = (grid.nx, grid.ny, grid.nz)
    grid.props["PORO"] = np.full(shape, 0.2)
    for name in ("PERMX", "PERMY", "PERMZ"):
        grid.props[name] = np.full(shape, 100.0)
    write_grdecl(grid, tmp_path / "grid.grdecl")
    (tmp_path / "t.DATA").write_text(DECK.format(nx=grid.nx, ny=grid.ny, nz=grid.nz, datum=1500.0))
    opm_vol = opm_volumes_and_run(tmp_path)
    mine = [grid.cell_volume(i, j, k) for k in range(grid.nz) for j in range(grid.ny)
            for i in range(grid.nx) if grid.actnum[i, j, k]]
    assert np.allclose(opm_vol, mine, rtol=1e-5)
    assert opm_vol.sum() == pytest.approx(nx * ny * 100.0 * 100.0 * 60.0)


# -- boundary treatments through the real engine ----------------------------------------------------------------------------

_RUNNER2 = """
import sys, json
from opm.simulators import GasWaterSimulator
d = sys.argv[1]
sim = GasWaterSimulator(d + "/t.DATA", ["--output-dir=" + d, "--enable-terminal-output=false"])
sim.step_init()
p0 = [float(x) for x in sim.get_fluidstate_variable("pw")]
steps = 0
while not sim.check_simulation_finished():
    sim.step()
    steps += 1
out = dict(n=len(p0), steps=steps, p0=p0, p1=[float(x) for x in sim.get_fluidstate_variable("pw")],
           sg=[float(x) for x in sim.get_fluidstate_variable("Sg")])
sim.step_cleanup()
print("RESULT" + json.dumps(out))
"""


def opm_result(folder):
    import json
    r = subprocess.run([sys.executable, "-c", _RUNNER2, str(folder)], capture_output=True, text=True, timeout=600)
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT")]
    assert r.returncode == 0 and line, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(line[0][6:])


def boundary_model(options):
    from shapely.geometry import box
    from cosma.grid.builder import Layering as L
    from cosma.grid_definition import GridDefinition, build_model
    from cosma.project import Project
    from cosma.surfaces import Surface
    pr = Project("p")
    pr.create_chart(["Top"], "Base")
    n = 41
    for h, z in zip(pr.chart.horizons(), (1500.0, 1560.0)):
        pr.attach_surface(h.uid, Surface(-200.0, -200.0, 50.0, 50.0, 0.0, 1, np.full((n, n), z)), "s")
    pr.set_boundary(box(100.0, 100.0, 500.0, 500.0))
    gd = GridDefinition(0.0, 0.0, 0.0, 8, 8, 75.0, 75.0, 0.1, {pr.chart.intervals[0].uid: L("fixed_count", n=3)}, {}, options)
    return build_model(pr, gd)


def run_boundary_case(tmp_path, options, name):
    """Build with the given boundary options, export with COSMA's own writer (geometry, FIPNUM and the include files),
    put the includes in the sections COSMA says they belong to, and run the deck."""
    from cosma.grid_definition import write_model
    mb = boundary_model(options)
    g = mb.grid
    shape = (g.nx, g.ny, g.nz)
    g.props["PORO"] = np.full(shape, 0.2)
    for k in ("PERMX", "PERMY", "PERMZ"):
        g.props[k] = np.full(shape, 100.0)
    d = tmp_path / name
    d.mkdir()
    names = {f.name for f in write_model(mb, d / "grid.grdecl")}
    inc = lambda suffix: f"INCLUDE\n  'grid_{suffix}.inc' /\n" if f"grid_{suffix}.inc" in names else ""   # noqa: E731
    deck = DECK.format(nx=g.nx, ny=g.ny, nz=g.nz, datum=1500.0)
    deck = deck.replace("EQLDIMS\n  1 /\n", "EQLDIMS\n  1 /\n" + inc("runspec_boundary"), 1)
    deck = deck.replace("GRID\nINCLUDE\n  'grid.grdecl' /\n", "GRID\nINCLUDE\n  'grid.grdecl' /\n" + inc("grid_boundary"), 1)
    deck = deck.replace("RTEMP\n  60 /\n", "RTEMP\n  60 /\n" + inc("solution_boundary"), 1)
    deck = deck.replace("'INJ' 'G' 1 1 1* GAS", "'INJ' 'G' 4 4 1* GAS").replace(f"'INJ' 1 1 1 {g.nz}", f"'INJ' 4 4 1 {g.nz}")
    deck = deck.replace("RATE 1000.0", "RATE 100000.0")      # enough CO2 that the boundary, not datum noise, decides
    (d / "t.DATA").write_text(deck)
    return d, mb, names


def mean_rise(r):
    return float(np.mean(r["p1"]) - np.mean(r["p0"])) / 1e5          # bar


def test_every_boundary_treatment_runs_and_behaves_in_opm_flow(tmp_path):
    from cosma.grid.boundary_options import AquiferSpec, BoundaryOptions, PVMultiplier
    all_faces = ("I-", "I+", "J-", "J+")
    cases = {
        "closed": BoundaryOptions(),
        "buffer": BoundaryOptions(buffer=75.0),
        "pv": BoundaryOptions(pv=PVMultiplier(50.0, 1)),
        "fet": BoundaryOptions(aquifer=AquiferSpec("fetkovich", all_faces, 1500.0, 160.0, 1e-4, 1.0,
                                                   volume=1e9, productivity_index=500.0)),
        "ct": BoundaryOptions(aquifer=AquiferSpec("carter_tracy", all_faces, 1500.0, None, 1e-4, 1.0, permeability=200.0,
                                                  porosity=0.2, external_radius=5000.0, thickness=60.0,
                                                  influx_angle=360.0, table=1)),
    }
    res = {}
    for name, opt in cases.items():
        d, mb, names = run_boundary_case(tmp_path, opt, name)
        res[name] = opm_result(d)
        assert res[name]["n"] == int(mb.grid.actnum.sum()) and res[name]["steps"] == 3
        assert max(res[name]["sg"]) > 0.0                               # CO2 went in
    assert "FIPNUM" in (tmp_path / "buffer" / "grid.grdecl").read_text()
    closed = mean_rise(res["closed"])
    assert closed > 1.0                                                  # a closed box pressurises
    for name in ("buffer", "pv", "fet", "ct"):
        assert mean_rise(res[name]) < 0.95 * closed, (name, mean_rise(res[name]), closed)
    assert mean_rise(res["fet"]) < mean_rise(res["pv"]) < closed        # an aquifer with PI 500 beats a x50 edge ring


# -- fault transmissibility through the real engine -------------------------------------------------------------------------

def fault_case(tmp_path, name, choice, perm=100.0):
    """A closed 8 x 8 grid cut by a N-S fault with 20 m throw; CO2 is injected west of it. Returns the mean pressure
    rise (bar) in the cells east of the fault, and the model."""
    from cosma.faults import Fault
    from cosma.fault_transmissibility import FaultTransmissibility  # noqa: F401
    from cosma.grid.builder import Layering as L
    from cosma.grid_definition import GridDefinition, build_model, write_model
    from cosma.project import Project
    from cosma.surfaces import Surface
    pr = Project("p")
    pr.create_chart(["Top"], "Base")
    n = 41
    X, _ = np.meshgrid(-200.0 + 50.0 * np.arange(n), -200.0 + 50.0 * np.arange(n), indexing="ij")
    for h, z in zip(pr.chart.horizons(), (1500.0, 1560.0)):
        pr.attach_surface(h.uid, Surface(-200.0, -200.0, 50.0, 50.0, 0.0, 1, z + 20.0 * (X > 312.0)), "s")
    pr.attach_faults([Fault("F", tuple(np.array([[312.0, y, z] for z in (0.0, 3000.0)]) for y in (-100.0, 700.0)))])
    pr.set_fault_depth("F", 1500.0)
    pr.set_fault_transmissibility("F", choice)
    g = GridDefinition(0.0, 0.0, 0.0, 8, 8, 75.0, 75.0, 0.1, {pr.chart.intervals[0].uid: L("fixed_count", n=3)}, {})
    mb = build_model(pr, g)
    grid = mb.grid
    shape = (grid.nx, grid.ny, grid.nz)
    grid.props["PORO"] = np.full(shape, 0.2)
    for k in ("PERMX", "PERMY", "PERMZ"):
        grid.props[k] = np.full(shape, perm)
    d = tmp_path / name
    d.mkdir()
    write_model(mb, d / "grid.grdecl")
    deck = DECK.format(nx=grid.nx, ny=grid.ny, nz=grid.nz, datum=1500.0)
    deck = deck.replace("GRID\nINCLUDE\n  'grid.grdecl' /\n", "GRID\nINCLUDE\n  'grid.grdecl' /\nINCLUDE\n  'grid_faults.inc' /\n", 1)
    deck = deck.replace("RATE 1000.0", "RATE 100000.0")
    (d / "t.DATA").write_text(deck)
    r = opm_result(d)
    # active cells are listed I fastest, then J, then K
    idx = [(i, j, k) for k in range(grid.nz) for j in range(grid.ny) for i in range(grid.nx) if grid.actnum[i, j, k]]
    east = np.array([i >= 4 for i, _, _ in idx])
    rise = (np.array(r["p1"]) - np.array(r["p0"])) / 1e5
    return float(rise[east].mean()), float(rise[~east].mean()), mb


def test_fault_transmissibility_choices_change_flow_in_opm_flow(tmp_path):
    from cosma.fault_transmissibility import FaultTransmissibility as FT
    east_open, west_open, _ = fault_case(tmp_path, "open", FT("open"))
    east_seal, west_seal, _ = fault_case(tmp_path, "seal", FT("fixed", multiplier=0.0))
    east_part, west_part, _ = fault_case(tmp_path, "part", FT("fixed", multiplier=0.01))
    east_calc, west_calc, mb = fault_case(tmp_path, "calc", FT("calculated", method="slab", kf=0.01, thickness=0.5))
    # open: both sides feel the injection; sealing: the far side feels nothing and the near side more
    assert east_open > 5.0 and east_seal < 0.05 * east_open
    assert west_seal > west_open
    # a leaky fault sits between
    assert east_seal < east_part < east_open
    assert west_open < west_part < west_seal
    # the calculated multiplier: 0.5 m of 0.01 mD in 100 mD rock between 75 m cells
    from cosma.fault_transmissibility import calculated_multiplier
    rep = mb.fault_reports[0]
    m, _ = calculated_multiplier(mb.fault_choices["F"], rep.faces_i, rep.faces_j, mb.grid, rep)
    assert m == pytest.approx(1 / (1 + 0.5 * (100 / 0.01 - 1) / 75.0), rel=1e-9)
    # ... and the engine treats a calculated fault exactly like a fixed one with that multiplier
    east_fix, west_fix, _ = fault_case(tmp_path, "fixed_m", FT("fixed", multiplier=m))
    assert east_calc == pytest.approx(east_fix, rel=1e-3) and west_calc == pytest.approx(west_fix, rel=1e-3)
    assert east_seal < east_calc < east_open
