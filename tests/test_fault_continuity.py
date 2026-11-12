"""Faults must not break properties: the same layer is the same layer on both sides of a fault.

A fault that displaces the layers after they were deposited moves every pick and every sample of the wells on one side
down by the throw. Nothing about the properties changes. The experiment below makes exactly that change to a synthetic
project (an exactly simulated field) and requires every property result to stay the same."""

import copy
import math

import numpy as np
import pytest

from cosma.fault_checks import Crossing, crossing_warnings, side_of_trace, well_crossings
from cosma.faults import Fault
from cosma.maps import MapGrid, krige_map, layer_averages, thickness_map
from cosma.surfaces import Surface
from cosma.variogram import experimental_variogram
from cosma.variogram_data import collect
from cosma.variogram_session import VariogramSession
from cosma.wells.model import LogSet, RawTop, Survey
from synthetic_field import TRUTH, synthetic
from test_faults import plane_sticks

THROW = 37.3
FAULT_X = 375.0          # between the wells at x = 300 and x = 450


def with_throw(pr, throw=THROW, x_min=FAULT_X):
    """Move every pick and log sample of the wells east of the fault down by `throw`."""
    out = copy.deepcopy(pr)
    for rec in out.wells.values():
        if rec.header["x"] >= x_min:
            rec.tops = {k: RawTop(k, t.depth + throw, t.depth_type) for k, t in rec.tops.items()}
            lg = rec.logs["lg"]
            rec.logs["lg"] = LogSet(lg.md + throw, dict(lg.curves))
            rec.survey = Survey(np.array([0.0, 400.0 + throw]), np.zeros(2), np.zeros(2), "grid")
    return out


def step_surfaces(pr, throw, step_at=FAULT_X):
    u = pr.chart.match_name("Res")
    xs = -100.0 + 50.0 * np.arange(26)
    for hid, d0 in ((u, 100.0), ("base", 150.0)):
        v = d0 + np.where(xs >= step_at, throw, 0.0)[:, None] * np.ones((1, 26))
        pr.attach_surface(hid, Surface(-100.0, -100.0, 50.0, 50.0, 0.0, 1, v), "s")
    return u


def pairs_in_same_layer(points, groups, tol=2.0, cross_only_between=(300.0, 450.0)):
    """Number of well pairs on opposite sides of x = FAULT_X with samples within `tol` m of each other in z."""
    x = points[:, 0]
    west, east = x < FAULT_X, x >= FAULT_X
    n = 0
    zw, ze = points[west, 2], points[east, 2]
    for z in zw[::5]:
        n += int(np.sum(np.abs(ze - z) <= tol))
    return n


@pytest.fixture(scope="module")
def projects():
    base = synthetic(n=5, spacing=150.0, seed=11)
    return base, with_throw(base)


def test_collected_samples_are_unchanged_by_throw(projects):
    base, thrown = projects
    u = base.chart.match_name("Res")
    a, b = collect(base, u, "PHI"), collect(thrown, u, "PHI")
    assert a.points.shape == b.points.shape and (a.groups == b.groups).all()
    assert np.allclose(a.points, b.points, atol=1e-9) and np.array_equal(a.values, b.values)
    assert a.thickness == pytest.approx(b.thickness)


def test_the_experiment_would_notice_a_depth_based_implementation(projects):
    """Negative control: with absolute depth as the vertical coordinate the east/west pairs would be lost."""
    base, thrown = projects
    u = base.chart.match_name("Res")
    d = collect(thrown, u, "PHI")
    absolute = d.points.copy()
    absolute[:, 2] += 100.0 + np.where(d.points[:, 0] >= FAULT_X, THROW, 0.0)       # depth below top + depth of the top
    assert pairs_in_same_layer(absolute, d.groups) < 0.3 * pairs_in_same_layer(d.points, d.groups)         # 450 against 2280


def test_layer_averages_variograms_and_maps_are_unchanged_by_throw(projects):
    base, thrown = projects
    u = base.chart.match_name("Res")
    la, lb = layer_averages(base, u, "PHI"), layer_averages(thrown, u, "PHI")
    assert la.wells == lb.wells and np.allclose(la.value, lb.value, atol=1e-9) and np.allclose(la.x, lb.x) and np.allclose(la.y, lb.y)
    for kind, az in (("major", 0.0), ("major", 90.0)):
        ea, eb = (experimental_variogram(collect(p, u, "PHI").points, collect(p, u, "PHI").values, kind, lag=150.0, nlags=4, azimuth=az,
                                         tolerance_deg=30.0, groups=collect(p, u, "PHI").groups, vertical_tolerance=2.0)
                  for p in (base, thrown))
        assert np.array_equal(ea.npairs, eb.npairs) and np.allclose(ea.gamma, eb.gamma, equal_nan=True, atol=1e-9)
    # across the fault: the east-west variogram uses the pairs that straddle it
    e = experimental_variogram(collect(thrown, u, "PHI").points, collect(thrown, u, "PHI").values, "major", lag=150.0, nlags=3, azimuth=90.0,
                               tolerance_deg=10.0, groups=collect(thrown, u, "PHI").groups, vertical_tolerance=2.0)
    assert e.npairs[0] > 0 and np.isfinite(e.gamma[0])
    g = MapGrid(0.0, 0.0, 25.0, 25.0, 25, 25)
    ma, _ = krige_map(base, u, "PHI", TRUTH, g, unit="fraction", mask="none")
    mb, _ = krige_map(thrown, u, "PHI", TRUTH, g, unit="fraction", mask="none")
    assert np.allclose(ma.values, mb.values, atol=1e-9) and np.allclose(ma.variance, mb.variance, atol=1e-9)


def test_map_is_continuous_across_the_fault_and_thickness_is_not_disturbed(projects):
    base, thrown = projects
    thrown = copy.deepcopy(thrown)
    u = step_surfaces(thrown, THROW)
    g = MapGrid(0.0, 0.0, 12.5, 12.5, 73, 73)
    m, notes = krige_map(thrown, u, "PHI", TRUTH, g, unit="fraction")                  # masked to the interval by the stepped surfaces
    assert m.info["masked"] == 0 and np.isfinite(m.values).all()
    X, _ = g.node_xy()
    i = int(np.argmin(abs(X[:, 0] - FAULT_X)))
    jump_at_fault = np.abs(np.diff(m.values, axis=0))[i - 1:i + 1].max()
    elsewhere = np.abs(np.diff(m.values, axis=0))
    assert jump_at_fault <= 1.5 * elsewhere.max()
    assert np.abs(np.diff(m.values, axis=0))[i - 1:i + 1].mean() <= 2 * elsewhere.mean()
    t = thickness_map(thrown, u, MapGrid(0.0, 0.0, 12.5, 12.5, 73, 73))
    assert np.allclose(t.values, 50.0)                                                    # the step does not become thickness


# -- wells that cross a fault ----------------------------------------------------------------------------------
def vertical_fault(x=FAULT_X, y0=-100.0, y1=1000.0, zmax=400.0):
    return Fault("F1", tuple(np.array([[x, y, 0.0], [x, y, zmax]]) for y in np.linspace(y0, y1, 5)))


def crossing_project(dip=None):
    pr = synthetic(n=3, spacing=150.0, seed=5)                  # wells at x, y in {0, 150, 300}; Res at 100-150 m
    pr.wells["W3"].survey = Survey(np.array([0.0, 400.0]), np.full(2, 30.0), np.full(2, 90.0), "grid")    # W3: x=0,y=150 heads east at 30 deg
    return pr


def test_deviated_well_cutting_a_vertical_fault_is_reported():
    pr = crossing_project()
    pr.wells["W3"].header["x"] = 300.0 - 0.0                      # starts 75 m west of the fault at x=375
    pr.attach_faults([vertical_fault()])
    u = pr.chart.match_name("Res")
    d = collect(pr, u, "PHI")
    (c,) = d.crossings["W3"]
    assert c.fault == "F1" and c.x == pytest.approx(FAULT_X, abs=0.5) and c.tvdss == pytest.approx(75.0 / math.tan(math.radians(30)), abs=1.0)
    assert 100.0 <= c.tvdss <= 150.0 and set(d.crossings) == {"W3"}
    la = layer_averages(pr, u, "PHI")
    assert any(w.startswith("W3 crosses F1 at 130 m TVDSS inside this interval") for w in la.warnings)
    assert set(la.crossings) == {"W3"}
    m, notes = krige_map(pr, u, "PHI", TRUTH, MapGrid(0, 0, 50, 50, 8, 8), unit="fraction", mask="none")
    assert m.info["fault_crossings"]["W3"][0]["fault"] == "F1" and any("W3 crosses F1" in n for n in notes)
    s = VariogramSession(pr, u, "PHI")
    assert any("W3 crosses F1" in w for w in s.warnings())


def test_dipping_fault_crossed_by_a_vertical_well():
    pr = synthetic(n=3, spacing=150.0, seed=5)
    sticks = plane_sticks(60.0, 90.0, [-200, 100, 400, 700], length=400.0, origin=(375.0, 0.0))     # at depth z the fault is at x = 375 + z/tan(60)
    pr.attach_faults([Fault("D", sticks)])
    u = pr.chart.match_name("Res")
    pr.wells["W6"].header["x"] = 450.0                                                    # 75 m east of the fault trace at the surface
    d = collect(pr, u, "PHI")
    cs = d.crossings.get("W6")
    assert cs and len(cs) == 1 and cs[0].tvdss == pytest.approx(75.0 * math.tan(math.radians(60)), abs=1.5)    # 129.9 m, inside Res (100-150)
    assert "W0" not in d.crossings


def test_crossings_outside_the_interval_past_the_tip_and_without_faults_are_not_reported():
    pr = crossing_project()
    pr.wells["W3"].header["x"] = 300.0
    u = pr.chart.match_name("Res")
    assert collect(pr, u, "PHI").crossings == {} and layer_averages(pr, u, "PHI").warnings == []         # no faults at all
    pr.attach_faults([vertical_fault(x=300.0 + 75.0 * 4.0)])                               # a fault where the well passes below the interval
    assert collect(pr, u, "PHI").crossings == {}
    pr2 = crossing_project()
    pr2.wells["W3"].header["x"] = 300.0
    pr2.attach_faults([vertical_fault(y0=500.0, y1=900.0)])                               # the fault ends 350 m north of the well
    assert collect(pr2, pr2.chart.match_name("Res"), "PHI").crossings == {}
    pr3 = crossing_project()
    pr3.wells["W3"].header["x"] = 300.0
    pr3.attach_faults([vertical_fault(zmax=60.0)])                                        # the fault does not reach Res
    assert collect(pr3, pr3.chart.match_name("Res"), "PHI").crossings == {}


def test_side_of_trace_and_crossing_primitives():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 5.0]])
    assert side_of_trace(line, (5, 3)) == pytest.approx(3.0) and side_of_trace(line, (5, -2)) == pytest.approx(-2.0)
    assert side_of_trace(line, (-1, 0)) is None and side_of_trace(line, (21, 6)) is None          # beside an end: past the tip
    assert side_of_trace(line[:1], (0, 0)) is None and side_of_trace(np.array([[1.0, 1.0], [1.0, 1.0]]), (0, 0)) is None
    assert side_of_trace(line, (10, 0.5)) is not None                                            # at a joint
    ws = crossing_warnings({"A": [Crossing("F", 1234.4, 1.0, 2.0), Crossing("G", 99.6, 0, 0)], "B": [], "C": [Crossing("F", 5.0, 0, 0)]}, wells={"A", "B"})
    assert len(ws) == 1 and ws[0].startswith("A crosses F at 1234 m TVDSS, G at 100 m TVDSS inside this interval")
    assert well_crossings({}, None, 0.0, 1.0) == []


def test_zero_length_segments_in_a_trace_are_harmless():
    line = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    assert side_of_trace(line, (0.0, 1.0)) == pytest.approx(1.0) and side_of_trace(line, (5.0, -2.0)) == pytest.approx(-2.0)
    assert side_of_trace(np.empty((0, 2)), (0.0, 0.0)) is None


def test_a_curved_fault_crossed_twice_gives_both_crossings():
    zs = np.arange(0.0, 300.1, 2.5)
    sticks = tuple(np.column_stack([400.0 + 20.0 * np.sin(2 * np.pi * (zs - 95.0) / 40.0), np.full(zs.size, y), zs]) for y in (-100.0, 300.0, 700.0, 1000.0))
    pr = synthetic(n=3, spacing=150.0, seed=5)
    pr.wells["W6"].header["x"] = 400.0
    pr.attach_faults([Fault("Wavy", sticks)])
    cs = collect(pr, pr.chart.match_name("Res"), "PHI").crossings["W6"]
    assert [round(c.tvdss) for c in cs] == [115, 135]
    assert "Wavy at 115 m TVDSS, Wavy at 135 m TVDSS" in layer_averages(pr, pr.chart.match_name("Res"), "PHI").warnings[0]


def test_well_crossings_clamps_to_the_surveyed_range():
    pr = crossing_project()
    pr.wells["W3"].header["x"] = 300.0
    pr.wells["W3"].survey = Survey(np.array([0.0, 120.0]), np.full(2, 30.0), np.full(2, 90.0), "grid")     # ends inside Res (100-150)
    tr = pr.wells["W3"].trajectory
    (c,) = well_crossings({"F1": vertical_fault(x=355.0)}, tr, 100.0, 150.0)                                 # asks for MD 100-150; the survey ends at 120
    assert c.tvdss == pytest.approx(55.0 / math.tan(math.radians(30)), abs=1.0)
    assert well_crossings({"F1": vertical_fault()}, tr, 100.0, 150.0) == []                                  # this one lies beyond the end of the survey
    assert well_crossings({"F1": vertical_fault(x=355.0)}, tr, 130.0, 100.0) == []


def test_a_crossing_well_that_is_not_used_is_not_warned_about():
    pr = crossing_project()
    pr.wells["W3"].header["x"] = 300.0
    pr.attach_faults([vertical_fault()])
    u = pr.chart.match_name("Res")
    assert set(layer_averages(pr, u, "PHI").crossings) == {"W3"}
    lg = pr.wells["W3"].logs["lg"]
    v = lg.curves["PHI"].copy()
    v[lg.md > 110.0] = np.nan                                                     # covers ~20 % of Res
    pr.wells["W3"].logs["lg"] = LogSet(lg.md, {"PHI": v})
    la = layer_averages(pr, u, "PHI")
    assert "W3" in la.skipped and la.crossings == {} and not any("crosses" in w for w in la.warnings)
