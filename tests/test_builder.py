"""Grid builder: layering, contacts, min thickness, sampling. Truth is analytic."""

import numpy as np
import pytest

from cosma.grid.boundary import grid_spec_from_cells
from cosma.grid.builder import (BuildError, Layering, apply_contacts, build_grid,
                                sample_horizons)
from cosma.surfaces import Surface

NX, NY = 6, 4
DX = DY = 100.0


def spec(**kw):
    return grid_spec_from_cells(0, 0, 0, NX, NY, DX, DY)


def flat(z):
    return np.full((NY + 1, NX + 1), float(z))


def ramp(z0, dzdx):
    x = np.arange(NX + 1) * DX
    return np.tile(z0 + dzdx * x, (NY + 1, 1))


def vol(grid):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    return sum(grid.cell_volume(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)
               if grid.actnum[i, j, k])


def layer_tops(grid, i=0, j=0):
    """Depth of every layer boundary at the SW pillar of cell (i, j)."""
    return np.append(grid.zcorn[:, 0, j, 0, i, 0], grid.zcorn[-1, 1, j, 0, i, 0])


FC = lambda n: Layering("fixed_count", n=n)  # noqa: E731


def test_flat_fixed_count_volume_and_shape():
    g, r = build_grid(spec(), [flat(1000), flat(1060)], ["conformable"], [FC(6)])
    assert (g.nx, g.ny, g.nz) == (NX, NY, 6)
    assert r.layers_per_interval == [6]
    assert np.allclose(layer_tops(g), 1000 + 10 * np.arange(7))
    assert vol(g) == pytest.approx(NX * NY * DX * DY * 60)
    assert g.actnum.all() and not r.warnings


def test_two_intervals_layer_interval_map():
    g, r = build_grid(spec(), [flat(1000), flat(1020), flat(1050)], ["conformable"] * 2, [FC(2), FC(3)])
    assert g.nz == 5
    assert list(r.layer_interval) == [0, 0, 1, 1, 1]
    assert np.allclose(layer_tops(g), [1000, 1010, 1020, 1030, 1040, 1050])


def test_fixed_thickness_sloping_base():
    # thickness 20 at west growing to 50 at east; dz=10 -> 5 layers; west column uses 2
    base = ramp(1020, 30.0 / (NX * DX))
    g, r = build_grid(spec(), [flat(1000), base], ["conformable"], [Layering("fixed_thickness", dz=10)],
                      min_thickness=0.1)
    assert g.nz == 5
    # column 0: t = 20..25 -> thick cells 4..5 m... each column's layers equal in thickness
    for i in range(NX):
        tops = layer_tops(g, i)
        assert tops[0] == pytest.approx(1000)
        assert tops[-1] == pytest.approx(base[0, i])  # pillar at i (west) ; base reached
        t = base[0, i] - 1000
        n_i = int(np.ceil(t / 10 - 1e-9))
        th = np.diff(tops)
        assert np.allclose(th[:n_i], t / n_i) and np.allclose(th[n_i:], 0.0)
        assert th.max() <= 10 + 1e-9
    # volume conserved exactly
    assert vol(g) == pytest.approx(NY * DY * DX * np.sum((base[0, :-1] + base[0, 1:]) / 2 - 1000))


def test_follow_top_and_follow_base_anchoring():
    base = flat(1025)
    gt, _ = build_grid(spec(), [flat(1000), base], ["conformable"], [Layering("follow_top", dz=10)])
    gb, _ = build_grid(spec(), [flat(1000), base], ["conformable"], [Layering("follow_base", dz=10)])
    assert gt.nz == gb.nz == 3
    assert np.allclose(layer_tops(gt), [1000, 1010, 1020, 1025])
    assert np.allclose(layer_tops(gb), [1000, 1005, 1015, 1025])


def test_follow_top_parallel_to_a_dipping_top():
    top = ramp(1000, 0.1)
    base = ramp(1050, 0.1)
    g, _ = build_grid(spec(), [top, base], ["conformable"], [Layering("follow_top", dz=10)])
    for i in range(NX):
        tops = layer_tops(g, i)
        assert np.allclose(np.diff(tops), 10.0)


def test_follow_top_pinch_makes_inactive_cells_at_base():
    # thickness varies 10 .. 40 m across; dz = 10 -> 4 layers; thin side keeps only 1 live layer
    base = ramp(1010, 30.0 / (NX * DX))
    g, r = build_grid(spec(), [flat(1000), base], ["conformable"], [Layering("follow_top", dz=10)])
    assert g.nz == 4
    assert g.actnum[0, 0, 0] == 1 and g.actnum[0, 0, 3] == 0
    assert g.actnum[NX - 1, 0, 3] == 1
    assert r.thin_cells == (g.actnum == 0).sum()
    # collapsed cells have no volume, so nothing real is lost and total volume is conserved
    assert r.lost_volume == pytest.approx(0.0, abs=1e-6)
    assert vol(g) == pytest.approx(r.total_volume)


def test_erosional_truncates_older_horizons():
    # horizon 1 (top of interval 1) is an erosional surface that cuts into interval 1's base (h2)
    h0, h1, h2 = flat(1000), flat(1030), ramp(1010, 0.1)  # h2 rises above h1 on the west side
    d, clipped, inv = apply_contacts([h0, h1, h2], ["conformable", "erosional"])
    assert np.all(d[2] >= d[1])
    assert clipped[2] > 0
    g, r = build_grid(spec(), [h0, h1, h2], ["conformable", "erosional"], [FC(3), FC(3)])
    # interval 1 has zero thickness where h2 was clipped -> its layers inactive there
    assert g.actnum[0, 0, 5] == 0
    assert g.actnum[NX - 1, 0, 5] == 1
    assert not any("repaired" in w for w in r.warnings)


def test_onlap_clips_younger_horizons():
    # interval 0 (younger) onlaps the surface h1: h0 cannot be deeper than h1
    h0 = ramp(1000, 0.2)    # dips to 1120 at east
    h1 = flat(1060)
    d, clipped, inv = apply_contacts([h0, h1, flat(1100)], ["conformable", "onlap"])
    assert np.all(d[0] <= d[1])
    assert d[0][0, -1] == pytest.approx(1060)
    assert clipped[0] > 0


def test_conformable_crossing_is_repaired_and_reported():
    h0, h1 = flat(1000), ramp(990, 0.1)
    g, r = build_grid(spec(), [h0, h1, flat(1100)], ["conformable"] * 2, [FC(2), FC(2)])
    assert r.inverted_corners[1] > 0
    assert any("repaired" in w for w in r.warnings)
    assert np.all(np.diff(layer_tops(g, 0)) >= 0)


def test_min_thickness_default_and_override():
    base = flat(1000.05)  # 0.05 m thick
    g, r = build_grid(spec(), [flat(1000), base], ["conformable"], [FC(1)])
    assert not g.actnum.any() and r.thin_cells == NX * NY
    assert r.lost_volume == pytest.approx(NX * NY * DX * DY * 0.05)
    g2, r2 = build_grid(spec(), [flat(1000), base], ["conformable"], [FC(1)], min_thickness=0.01)
    assert g2.actnum.all() and r2.thin_cells == 0


def test_inactive_columns_stay_inactive():
    s = spec()
    s.active_xy[2, 1] = False
    g, r = build_grid(s, [flat(1000), flat(1010)], ["conformable"], [FC(2)])
    assert g.actnum[2, 1].sum() == 0 and g.actnum[2, 2].sum() == 2


def test_undefined_pillar_deactivates_adjacent_columns():
    h1 = flat(1010)
    h1[1, 2] = np.nan  # pillar (j=1, i=2) touches columns i in {1,2}, j in {0,1}
    g, r = build_grid(spec(), [flat(1000), h1], ["conformable"], [FC(2)])
    off = {(i, j) for i in range(NX) for j in range(NY) if g.actnum[i, j].sum() == 0}
    assert off == {(1, 0), (2, 0), (1, 1), (2, 1)}
    assert r.undefined_columns == 4 and np.isfinite(g.zcorn).all()


@pytest.mark.parametrize("kw", [
    dict(mode="nope", n=1), dict(mode="fixed_count"), dict(mode="fixed_count", n=0),
    dict(mode="fixed_count", n=2, dz=1.0), dict(mode="follow_top"), dict(mode="follow_top", dz=-1),
    dict(mode="follow_base", dz=5, n=2), dict(mode="fixed_thickness", dz=float("nan")),
])
def test_layering_validation(kw):
    with pytest.raises(BuildError):
        Layering(**kw)


def test_every_interval_needs_a_layering_and_contact():
    with pytest.raises(BuildError, match="layering"):
        build_grid(spec(), [flat(0), flat(10), flat(20)], ["conformable"] * 2, [FC(1)])
    with pytest.raises(BuildError, match="contact"):
        build_grid(spec(), [flat(0), flat(10)], ["conformable", "conformable"], [FC(1)])
    with pytest.raises(BuildError, match="not one of"):
        build_grid(spec(), [flat(0), flat(10)], ["abrupt"], [FC(1)])
    with pytest.raises(BuildError, match="Layering objects"):
        build_grid(spec(), [flat(0), flat(10)], ["conformable"], [3])
    with pytest.raises(BuildError, match="shape"):
        build_grid(spec(), [flat(0), np.zeros((2, 2))], ["conformable"], [FC(1)])


def test_cell_limit():
    import cosma.grid.builder as b
    old = b.MAX_CELLS
    b.MAX_CELLS = 10
    try:
        with pytest.raises(BuildError, match="exceeds"):
            build_grid(spec(), [flat(0), flat(10)], ["conformable"], [FC(1)])
    finally:
        b.MAX_CELLS = old


def test_sample_horizons_on_a_rotated_frame():
    # plane z = 1000 + 0.5 * X - 0.2 * Y in map coordinates, rotated grid placed off-origin
    s = grid_spec_from_cells(5000, 7000, 30.0, NX, NY, DX, DY)
    ii, jj = np.meshgrid(np.arange(41), np.arange(41), indexing="ij")
    surf = Surface(4000.0, 6000.0, 50.0, 50.0, 0.0, 1, 1000 + 0.5 * (4000 + 50 * ii) - 0.2 * (6000 + 50 * jj))
    (h,) = sample_horizons(s, [surf])
    X, Y = s.frame.to_map(*np.meshgrid(s.x_edges, s.y_edges))
    assert np.allclose(h, 1000 + 0.5 * X - 0.2 * Y)


def test_sample_horizons_nan_outside_surface():
    surf = Surface(0.0, 0.0, 50.0, 50.0, 0.0, 1, np.full((5, 5), 1000.0))  # covers 0..200 only
    (h,) = sample_horizons(spec(), [surf])
    assert np.isnan(h[-1, -1]) and h[0, 0] == 1000.0


def test_grdecl_roundtrip_volume(tmp_path):
    from cosma.io.grdecl import write_grdecl
    g, _ = build_grid(spec(), [flat(1000), ramp(1050, 0.05)], ["conformable"], [FC(4)])
    p = write_grdecl(g, tmp_path / "g.grdecl")
    text = p.read_text()
    assert "COORD" in text and "ZCORN" in text and "ACTNUM" in text
