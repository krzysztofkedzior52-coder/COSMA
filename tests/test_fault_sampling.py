"""Side-aware sampling at faults. Truth: surfaces with a known step across a known line."""

import numpy as np
import pytest

from cosma.faults import Fault
from cosma.grid.boundary import grid_spec_from_cells
from cosma.grid.builder import Layering, build_grid
from cosma.grid.fault_sampling import (FaultSamplingError, classify_cells, corner_depths,
                                       fault_faces, local_trace, measure_throw)
from cosma.surfaces import Surface

NX, NY, D = 6, 4, 100.0


def vfault(name, p0, p1, z=(0.0, 3000.0)):
    """Vertical fault through map points p0 and p1 (two vertical sticks)."""
    return Fault(name, tuple(np.array([[x, y, zz] for zz in z]) for x, y in (p0, p1)))


def stepped_surface(z0, throw, line_p, line_n, slope=(0.1, -0.05), lo=-600.0, hi=1400.0, inc=25.0):
    """z = z0 + slope.(X,Y) + throw where n.(P - p) > 0. Nodes on the positive side carry the throw."""
    n = int((hi - lo) / inc) + 1
    X, Y = np.meshgrid(lo + inc * np.arange(n), lo + inc * np.arange(n), indexing="ij")
    pos = (line_n[0] * (X - line_p[0]) + line_n[1] * (Y - line_p[1])) > 0
    z = z0 + slope[0] * X + slope[1] * Y + throw * pos
    return Surface(lo, lo, inc, inc, 0.0, 1, z)


def plane(z0, X, Y, slope=(0.1, -0.05)):
    return z0 + slope[0] * X + slope[1] * Y


def aligned(throw, z0s=(1000.0, 1050.0, 1100.0)):
    spec = grid_spec_from_cells(0, 0, 0, NX, NY, D, D)
    fault = vfault("F1", (312.0, -100.0), (312.0, 500.0))
    surfs = [stepped_surface(z, throw, (312.0, 0.0), (1.0, 0.0)) for z in z0s]
    return spec, fault, surfs


def test_trace_direction_and_sides():
    spec, fault, _ = aligned(20.0)
    tr = local_trace(fault, 1500.0, spec.frame)
    side = classify_cells(spec, tr)
    # trace is directed +y, so its left is west: cells i<=2 are +1, i>=3 are -1
    assert (side[:3] == 1).all() and (side[3:] == -1).all()
    fi, fj = fault_faces(spec, side)
    assert fi.sum() == NY and fi[2].all() and not fj.any()


def test_measure_throw_recovers_the_step():
    spec, fault, surfs = aligned(20.0)
    tr = local_trace(fault, 1500.0, spec.frame)
    _, th, zl, zr = measure_throw(surfs[0], spec.frame, tr, 37.5, 25.0)
    assert np.allclose(th, -20.0, atol=1e-6)   # left (west) minus right (east): east is deeper


def test_corner_depths_follow_each_cells_own_side():
    spec, fault, surfs = aligned(20.0)
    out, (rep,) = corner_depths(spec, surfs, {"F1": fault}, {"F1": 1500.0})
    assert rep.has_throw and rep.n_faces == NY
    assert rep.snap_distance_max == pytest.approx(12.0)      # face at x=300, trace at x=312
    assert np.allclose(rep.median_throw, -20.0, atol=1e-6)
    cx = np.broadcast_to(spec.x_edges[np.arange(NX)[:, None] + np.arange(2)[None, :]][None, None], (NY, 2, NX, 2))
    cy = np.broadcast_to(spec.y_edges[np.arange(NY)[:, None] + np.arange(2)[None, :]][:, :, None, None], (NY, 2, NX, 2))
    east = (np.arange(NX) >= 3)[None, None, :, None]
    for h, z0 in enumerate((1000.0, 1050.0, 1100.0)):
        assert np.allclose(out[h], plane(z0, cx, cy) + 20.0 * east, atol=1e-6)


def test_built_grid_has_a_step_and_conserves_thickness():
    spec, fault, surfs = aligned(20.0)
    out, _ = corner_depths(spec, surfs, {"F1": fault}, {"F1": 1500.0})
    g, r = build_grid(spec, out, ["conformable"] * 2, [Layering("fixed_count", n=2)] * 2)
    assert g.actnum.all() and not r.warnings
    # across the fault face the same layer is 20 m deeper on the east side
    west = g.zcorn[0, 0, 0, 0, 2, 1]
    east = g.zcorn[0, 0, 0, 0, 3, 0]
    assert east - west == pytest.approx(20.0, abs=1e-6)
    # every interval keeps its 50 m thickness on both sides, so volume is exact
    vol = sum(g.cell_volume(i, j, k) for i in range(NX) for j in range(NY) for k in range(g.nz))
    assert vol == pytest.approx(NX * NY * D * D * 100.0)


def test_negligible_throw_leaves_geometry_alone():
    spec, fault, surfs = aligned(2.0)           # 2 m against 50 m thick intervals: 4 %
    out, (rep,) = corner_depths(spec, surfs, {"F1": fault}, {"F1": 1500.0})
    assert not rep.has_throw and all(rep.negligible)
    assert any("transmissibility" in w for w in rep.warnings)
    # geometry is the plain bilinear sample (smeared step and all), not side-corrected
    cx = spec.x_edges[np.arange(NX)[:, None] + np.arange(2)[None, :]]
    plain = surfs[0].sample(cx[None, None, :, :] + 0 * np.zeros((NY, 2, 1, 1)),
                            spec.y_edges[np.arange(NY)[:, None] + np.arange(2)[None, :]][:, :, None, None]
                            + 0 * np.zeros((1, 1, NX, 2)))
    assert np.allclose(out[0], plain)


def test_threshold_is_ten_percent_of_the_thinner_interval():
    spec, fault, surfs = aligned(4.0, z0s=(1000.0, 1040.0, 1100.0))   # thin interval 40 m: 10 % = 4 m
    _, (rep,) = corner_depths(spec, surfs, {"F1": fault}, {"F1": 1500.0})
    assert rep.median_ratio[1] == pytest.approx(0.1, abs=1e-6)
    _, (rep2,) = corner_depths(spec, surfs, {"F1": fault}, {"F1": 1500.0}, threshold=0.11)
    assert rep2.negligible[1] and not rep.negligible[1]


def test_rotated_frame_every_cell_follows_the_map_line():
    spec = grid_spec_from_cells(500.0, 700.0, 30.0, NX, NY, D, D)
    p = np.array(spec.frame.to_map(310.0, 150.0)).ravel()
    ang = np.radians(30.0)
    # fault runs along the grid's local +y direction (map direction of local +y)
    dirv = np.array([np.sin(ang), np.cos(ang)])
    nrm = np.array([dirv[1], -dirv[0]])            # right of the direction = local +x side (east)
    fault = vfault("F", p - 700 * dirv, p + 700 * dirv)
    surfs = [stepped_surface(z, 15.0, p, nrm, lo=-300.0, hi=1500.0) for z in (1000.0, 1060.0)]
    out, (rep,) = corner_depths(spec, surfs, {"F": fault}, {"F": 2000.0})
    assert rep.has_throw
    ci = np.arange(NX)[:, None] + np.arange(2)
    cj = np.arange(NY)[:, None] + np.arange(2)
    lx = np.broadcast_to(spec.x_edges[ci][None, None], (NY, 2, NX, 2))
    ly = np.broadcast_to(spec.y_edges[cj][:, :, None, None], (NY, 2, NX, 2))
    mx, my = spec.frame.to_map(lx, ly)
    cxm, cym = spec.frame.to_map(0.5 * (spec.x_edges[:-1] + spec.x_edges[1:])[None, :],
                                 0.5 * (spec.y_edges[:-1] + spec.y_edges[1:])[:, None])   # (ny, nx)
    pos = ((nrm[0] * (cxm - p[0]) + nrm[1] * (cym - p[1])) > 0)[:, None, :, None]
    for h, z0 in enumerate((1000.0, 1060.0)):
        assert np.allclose(out[h], plane(z0, mx, my) + 15.0 * pos, atol=1e-6)


def test_fault_tip_ends_the_faces():
    spec = grid_spec_from_cells(0, 0, 0, NX, NY, D, D)
    fault = vfault("T", (312.0, -100.0), (312.0, 200.0))     # tip at y=200: cells j=0,1 only
    side = classify_cells(spec, local_trace(fault, 100.0, spec.frame))
    fi, fj = fault_faces(spec, side)
    assert fi.sum() == 2 and fi[2, :2].all() and not fi[2, 2:].any()
    assert (side[:, 2:] == 0).all()


def test_two_close_faulted_faults_warn_about_intersection_handling():
    spec, f1, surfs = aligned(20.0)
    f2 = vfault("F2", (330.0, -100.0), (330.0, 500.0))
    _, reps = corner_depths(spec, surfs, {"F1": f1, "F2": f2}, {"F1": 1500.0, "F2": 1500.0})
    assert any("interaction tool" in w for r in reps for w in r.warnings)


def test_trace_depth_must_be_chosen_and_must_exist():
    spec, fault, surfs = aligned(20.0)
    with pytest.raises(FaultSamplingError, match="explicit"):
        corner_depths(spec, surfs, {"F1": fault}, {})
    with pytest.raises(FaultSamplingError, match="no trace"):
        corner_depths(spec, surfs, {"F1": fault}, {"F1": 5000.0})
