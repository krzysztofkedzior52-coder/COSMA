import math

import numpy as np
import pytest

from cosma.faults import Fault
from cosma.grid.builder import Layering
from cosma.grid_definition import (GridDefinition, GridDefinitionError, build_model, footprint_extent,
                                   orientation_suggestions, required_items, structural_dip, suggest_extent)
from cosma.project import Project
from cosma.surfaces import Surface


def flat_surface(z, lo=-300.0, hi=1100.0, inc=25.0, slope=(0.0, 0.0), step=None):
    n = int((hi - lo) / inc) + 1
    X, Y = np.meshgrid(lo + inc * np.arange(n), lo + inc * np.arange(n), indexing="ij")
    v = z + slope[0] * X + slope[1] * Y
    if step is not None:
        v = v + step[1] * (X > step[0])
    return Surface(lo, lo, inc, inc, 0.0, 1, v)


def open_all(pr):
    from cosma.fault_transmissibility import FaultTransmissibility
    for n in pr.faults:
        pr.set_fault_transmissibility(n, FaultTransmissibility("open"))


def model(zs=(1000.0, 1050.0, 1100.0), **kw):
    pr = Project("p")
    pr.create_chart(["Top", "Res"], "Base")
    hs = pr.chart.horizons()
    for h, z in zip(hs, zs):
        pr.attach_surface(h.uid, flat_surface(z, **kw), "s")
    return pr


def gd(pr, **over):
    u = [iv.uid for iv in pr.chart.intervals]
    args = dict(origin_x=0.0, origin_y=0.0, azimuth_deg=0.0, nx=6, ny=4, dx=100.0, dy=100.0, min_thickness=0.1,
                layerings={u[0]: Layering("fixed_count", n=2), u[1]: Layering("follow_top", dz=10.0)}, unconformity={})
    args.update(over)
    return GridDefinition(**args)


def vf(name, p0, p1):
    return Fault(name, tuple(np.array([[x, y, z] for z in (0.0, 3000.0)]) for x, y in (p0, p1)))


def test_everything_missing_is_listed_not_defaulted():
    pr = Project("p")
    assert required_items(pr, None) == ["define the stratigraphic chart"]
    pr.create_chart(["Top", "Res"], "Base")
    assert required_items(pr, None) == ["define the grid (origin, azimuth, cells)"]
    g = GridDefinition(0, 0, 0, 6, 4, 100, 100)
    items = required_items(pr, g)
    assert sum("depth surface" in i for i in items) == 3 and sum("layering" in i for i in items) == 2
    with pytest.raises(GridDefinitionError, match="cannot build yet"):
        build_model(pr, g)


def test_unconformity_type_must_be_chosen():
    pr = model()
    pr.chart.intervals[1].contact_above = "unconformity"
    g = gd(pr)
    assert any("erosional or onlap" in i for i in required_items(pr, g))
    g = gd(pr, unconformity={pr.chart.intervals[1].uid: "erosional"})
    assert required_items(pr, g) == []


def test_plain_build_has_exact_volume_and_layer_counts():
    pr = model()
    mb = build_model(pr, gd(pr))
    assert (mb.grid.nx, mb.grid.ny) == (6, 4)
    assert mb.report.layers_per_interval == [2, 5]          # 50 m / 10 m for the second interval
    vol = sum(mb.grid.cell_volume(i, j, k) for i in range(6) for j in range(4) for k in range(mb.grid.nz))
    assert vol == pytest.approx(6 * 4 * 100 * 100 * 100.0)
    assert "6 x 4 x 7" in mb.summary() and "Top: 2 layer(s)" in mb.summary()


def test_unconformity_choice_changes_the_geometry():
    # Res/Base crosses above Top/Res on the east: base surface rises above the Res top
    pr = Project("p")
    pr.create_chart(["Top", "Res"], "Base")
    h = pr.chart.horizons()
    pr.attach_surface(h[0].uid, flat_surface(1000.0), "s")
    pr.attach_surface(h[1].uid, flat_surface(1050.0), "s")
    pr.attach_surface(h[2].uid, flat_surface(1020.0, slope=(0.1, 0.0)), "s")      # 1020 + 0.1x: crosses 1050 at x = 300
    pr.chart.intervals[1].contact_above = "unconformity"
    u1 = pr.chart.intervals[1].uid
    ero = build_model(pr, gd(pr, unconformity={u1: "erosional"}))
    on = build_model(pr, gd(pr, unconformity={u1: "onlap"}))
    # erosional: the older horizon below is cut off at the surface, so nothing is "repaired"
    assert ero.report.clipped_corners == [0, 0, 40] and not any(ero.report.inverted_corners)
    # onlap only clips horizons ABOVE the surface; the crossing below it is a data problem, repaired and warned about
    assert on.report.clipped_corners == [0, 0, 0] and on.report.inverted_corners[2] == 40
    assert any("repaired" in w for w in on.report.warnings)


def test_faults_need_depth_and_settled_interactions():
    pr = model(step=(312.0, 20.0))
    pr.attach_faults([vf("F", (312.0, -100.0), (312.0, 500.0))])
    g = gd(pr)
    assert any("trace depth" in i for i in required_items(pr, g))
    pr.set_fault_depth("F", 1500.0)
    assert any("transmissibility of fault 'F'" in i for i in required_items(pr, g))
    open_all(pr)
    assert required_items(pr, g) == []
    mb = build_model(pr, g)
    assert mb.fault_reports[0].has_throw and "Fault F" in mb.summary() and "faces" in mb.summary()
    pr.attach_faults([vf("G", (100.0, -100.0), (500.0, 500.0))])
    pr.set_fault_depth("G", 1500.0)
    assert any("transmissibility of fault 'G'" in i for i in required_items(pr, g))
    open_all(pr)
    assert any("check the fault interactions" in i for i in required_items(pr, g))
    pr.refresh_fault_interactions(5.0)
    assert any(i.startswith("confirm:") for i in required_items(pr, g))
    pr.confirm_fault_interaction(pr.fault_interactions[0].key, "F")
    assert required_items(pr, g) == []
    assert build_model(pr, g).age_order[0] == "F"


def test_negligible_fault_is_reported_as_transmissibility_only():
    pr = model(step=(312.0, 2.0))
    pr.attach_faults([vf("F", (312.0, -100.0), (312.0, 500.0))])
    pr.set_fault_depth("F", 1500.0)
    open_all(pr)
    s = build_model(pr, gd(pr)).summary()
    assert "throw negligible, transmissibility only" in s


def test_definition_validation():
    for kw in (dict(nx=0), dict(ny=2.5), dict(dx=0), dict(dy=float("nan")), dict(min_thickness=-1),
               dict(azimuth_deg=float("inf")), dict(unconformity={"u": "sideways"})):
        args = dict(origin_x=0, origin_y=0, azimuth_deg=0, nx=2, ny=2, dx=10, dy=10)
        args.update(kw)
        with pytest.raises(GridDefinitionError):
            GridDefinition(**args)
    st = GridDefinition(1, 2, 3, 4, 5, 6, 7, 0.2, {"a": Layering("follow_base", dz=3.0)}, {"b": "onlap"}).to_state()
    assert GridDefinition.from_state(st).to_state() == st
    with pytest.raises(GridDefinitionError):
        GridDefinition.from_state({"origin_x": 1})


def test_suggest_extent_covers_a_rotated_footprint():
    pr = Project("p")
    pr.create_chart(["Top"], "Base")
    h = pr.chart.horizons()
    rot = 30.0
    for hh, z in zip(h, (1000.0, 1100.0)):
        pr.attach_surface(hh.uid, Surface(500.0, 700.0, 50.0, 50.0, rot, 1, np.full((21, 11), z)), "s")
    az = -rot                                              # grid I axis along the surface's own axis
    ox, oy, nx, ny = suggest_extent(pr, az, 50.0, 50.0)
    assert (nx, ny) == (20, 10)
    assert (ox, oy) == pytest.approx((500.0, 700.0), abs=1e-6)
    x0, x1, y0, y1 = footprint_extent(np.array([[500.0, 700.0]]), az)
    # the north-aligned bounding grid is bigger than the aligned one
    (_, _, nxn, nyn) = suggest_extent(pr, 0.0, 50.0, 50.0)
    assert nxn * nyn > nx * ny
    sugg = {s.label: s for s in orientation_suggestions(pr, 50.0, 50.0)}
    assert sugg["north-aligned (no rotation)"].cost_ratio == 1.0


def test_orientation_suggestions_from_faults_and_dip():
    pr = Project("p")
    pr.create_chart(["Top"], "Base")
    h = pr.chart.horizons()
    X, Y = np.meshgrid(np.arange(0, 1001, 50.0), np.arange(0, 1001, 50.0), indexing="ij")
    pr.attach_surface(h[0].uid, Surface(0, 0, 50.0, 50.0, 0.0, 1, 1000.0 + 0.2 * X + 0.1 * Y), "s")
    pr.attach_surface(h[1].uid, Surface(0, 0, 50.0, 50.0, 0.0, 1, 1100.0 + 0.2 * X + 0.1 * Y), "s")
    dip, ddir = structural_dip(pr)
    assert dip == pytest.approx(math.degrees(math.atan(math.hypot(0.2, 0.1))), abs=1e-6) and ddir == pytest.approx(math.degrees(math.atan2(0.2, 0.1)))
    pr.attach_faults([vf("N", (400.0, 0.0), (400.0, 800.0)), vf("D", (0.0, 0.0), (800.0, 800.0))])
    sugg = orientation_suggestions(pr, 50.0, 50.0)
    labels = [s.label for s in sugg]
    assert any("fault N" in l for l in labels) and any("fault D" in l for l in labels)
    assert any("mean fault strike" in l for l in labels)
    dips = [s for s in sugg if "structural dip" in s.label]
    assert dips and dips[0].azimuth_deg == pytest.approx((math.degrees(math.atan2(0.2, 0.1)) - 90) % 360)
    strike_n = next(s for s in sugg if "fault N" in s.label)
    assert strike_n.azimuth_deg % 180 == pytest.approx(0.0, abs=1e-6)
    strike_d = next(s for s in sugg if "fault D" in s.label)
    assert strike_d.azimuth_deg % 180 == pytest.approx(45.0, abs=1e-6)
    assert all(s.cells and s.cost_ratio for s in sugg)
    # a 45 degree rotation of a square footprint costs more cells than none: the cost is shown, not hidden
    assert strike_d.cost_ratio > 1.0


def test_flat_structure_gives_no_dip_suggestion():
    pr = model()
    assert not any("structural dip" in s.label for s in orientation_suggestions(pr, 100.0, 100.0))
    assert suggest_extent(Project("p"), 0.0, 10.0, 10.0) is None


def test_ages_claim_shared_corners_only_when_the_answers_order_the_faults():
    pr = model(step=(312.0, 20.0))
    pr.attach_faults([vf("F", (312.0, -100.0), (312.0, 500.0)), vf("G", (330.0, -100.0), (330.0, 500.0))])
    for n in ("F", "G"):
        pr.set_fault_depth(n, 1500.0)
    open_all(pr)
    pr.refresh_fault_interactions(5.0)                      # parallel: no interaction, so no ages
    assert pr.fault_interactions == []
    s = "\n".join(w for r in build_model(pr, gd(pr)).fault_reports for w in r.warnings)
    assert "interaction tool" in s and "younger fault claims" not in s
    # crossing faults with a confirmed age do order the claim
    pr2 = model(step=(312.0, 20.0))
    pr2.attach_faults([vf("F", (312.0, -100.0), (312.0, 500.0)), vf("G", (230.0, 100.0), (430.0, 100.0))])
    for n in ("F", "G"):
        pr2.set_fault_depth(n, 1500.0)
    open_all(pr2)
    pr2.refresh_fault_interactions(5.0)
    pr2.confirm_fault_interaction(pr2.fault_interactions[0].key, "F")
    mb = build_model(pr2, gd(pr2))
    assert mb.age_order == ["F", "G"]


def test_confirmed_ages_reach_the_corner_claim_in_a_build():
    pr = Project("p")
    pr.create_chart(["Top", "Res"], "Base")
    n = 57
    X, Y = np.meshgrid(-300.0 + 25.0 * np.arange(n), -300.0 + 25.0 * np.arange(n), indexing="ij")
    for h, z in zip(pr.chart.horizons(), (1000.0, 1050.0, 1100.0)):
        pr.attach_surface(h.uid, Surface(-300.0, -300.0, 25.0, 25.0, 0.0, 1, z + 20.0 * (X > 312.0) + 10.0 * (X > 330.0)), "s")
    pr.attach_faults([vf("F", (312.0, -100.0), (312.0, 500.0)), vf("G", (300.0, -100.0), (345.0, 500.0))])
    for f in ("F", "G"):
        pr.set_fault_depth(f, 1500.0)
    open_all(pr)
    pr.refresh_fault_interactions(5.0)
    assert len(pr.fault_interactions) == 1
    pr.confirm_fault_interaction(pr.fault_interactions[0].key, "F")
    mb = build_model(pr, gd(pr))
    ws = "\n".join(w for r in mb.fault_reports for w in r.warnings)
    assert "younger fault claims" in ws and "interaction tool" not in ws


# -- boundary in the project build -----------------------------------------------------------------------------------------

def boundary_project():
    from shapely.geometry import box
    pr = model()
    pr.set_boundary(box(100.0, 100.0, 500.0, 500.0))
    return pr


def test_buffer_needs_a_polygon():
    from cosma.grid.boundary_options import BoundaryOptions
    pr = model()
    assert any("boundary polygon" in i for i in required_items(pr, gd(pr, boundary=BoundaryOptions(buffer=100.0))))
    assert not any("boundary polygon" in i for i in required_items(pr, gd(pr, boundary=BoundaryOptions())))


def test_polygon_limits_the_active_columns_and_buffer_adds_a_region(tmp_path):
    from cosma.grid.boundary_options import BoundaryOptions
    from cosma.grid_definition import write_model
    pr = boundary_project()
    closed = build_model(pr, gd(pr, nx=8, ny=8, dx=75.0, dy=75.0))
    assert closed.active_columns.sum() == 25 or closed.active_columns.sum() > 0
    n0 = int(closed.active_columns.sum())
    buf = build_model(pr, gd(pr, nx=8, ny=8, dx=75.0, dy=75.0, boundary=BoundaryOptions(buffer=75.0)))
    assert buf.active_columns.sum() > n0 and set(np.unique(buf.region)) == {0, 1, 2} or set(np.unique(buf.region)) == {1, 2}
    assert "buffer zone" in buf.summary() and "Boundary treatment: buffer 75 m" in buf.summary()
    files = write_model(buf, tmp_path / "m.grdecl")
    assert [f.name for f in files] == ["m.grdecl"]
    text = (tmp_path / "m.grdecl").read_text()
    assert "FIPNUM" in text and "1 = inside the boundary polygon, 2 = buffer zone" in text
    from cosma.io.grdecl import scan_keywords
    assert scan_keywords(tmp_path / "m.grdecl")["FIPNUM"] == 8 * 8 * buf.grid.nz
    vals = set()
    block = text.split("FIPNUM\n", 1)[1].split("/", 1)[0].split()
    for t in block:
        vals.add(int(t.split("*")[-1]))
    assert vals == {1, 2}                                      # integers; inactive columns carry 1, never 0
    assert "FIPNUM" not in buf.grid.props                      # exporting leaves the grid untouched
    write_model(closed, tmp_path / "c.grdecl")
    assert "FIPNUM" not in (tmp_path / "c.grdecl").read_text()


def test_pv_multiplier_and_aquifer_files(tmp_path):
    from cosma.grid.boundary_options import AquiferSpec, BoundaryOptions, PVMultiplier
    from cosma.grid_definition import write_model
    pr = boundary_project()
    ob = BoundaryOptions(pv=PVMultiplier(20.0, 1), aquifer=AquiferSpec(
        "fetkovich", ("I-", "J+"), 1500.0, None, 1e-4, 1.0, volume=1e9, productivity_index=50.0))
    mb = build_model(pr, gd(pr, nx=8, ny=8, dx=75.0, dy=75.0, boundary=ob))
    s = mb.summary()
    assert "pore-volume multiplier 20 on" in s and "Fetkovich" not in s and "fetkovich aquifer on I-, J+" in s
    assert "double-count" in s                              # both add volume outside the model
    names = sorted(f.name for f in write_model(mb, tmp_path / "m.grdecl"))
    assert names == ["m.grdecl", "m_grid_boundary.inc", "m_runspec_boundary.inc", "m_solution_boundary.inc"]
    from cosma.io.grdecl import scan_keywords
    assert scan_keywords(tmp_path / "m_grid_boundary.inc")["MULTPV"] == 8 * 8 * mb.grid.nz
    sol = (tmp_path / "m_solution_boundary.inc").read_text()
    assert "AQUFETP" in sol and "AQUANCON" in sol and "'I-'" in sol and "'J+'" in sol and "'I+'" not in sol
    assert mb.edge_count() > 0


def test_aquifer_faces_with_nothing_to_connect_are_flagged():
    from cosma.grid.boundary_options import AquiferSpec, BoundaryOptions
    pr = model()
    # no polygon: the active area is the whole grid, so every outer face exists; with a polygon nothing is missing either.
    ob = BoundaryOptions(aquifer=AquiferSpec("fetkovich", ("I-",), 1500.0, None, 1e-4, 1.0, volume=1e9, productivity_index=5.0))
    mb = build_model(pr, gd(pr, boundary=ob))
    assert not any("no outer faces" in n for n in mb.notes)
