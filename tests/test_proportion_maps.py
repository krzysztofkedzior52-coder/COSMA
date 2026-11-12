import numpy as np
import pytest
from PIL import Image

from cosma.grid.boundary import grid_spec_from_cells
from cosma.proportion_maps import (NODATA, Legend, MapImportError, Raster, check_fractions, classify_colours, crop, flip,
                                   georeference, load_raster, mask_polygon, resample_to_grid, rotate90, scale,
                                   suggest_sigma, to_fractions)

RED, BLUE, GREEN = (255, 0, 0), (0, 0, 255), (0, 200, 0)


def raster(h=20, w=40):
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, : w // 2] = RED
    rgb[:, w // 2:] = BLUE
    return Raster(rgb, np.ones((h, w), bool))


def test_load_png_with_alpha_and_refuse_16bit(tmp_path):
    a = np.zeros((4, 5, 4), np.uint8)
    a[..., 0] = 255
    a[0, 0, 3] = 0
    a[1:, :, 3] = 255
    a[0, 1:, 3] = 255
    Image.fromarray(a, "RGBA").save(tmp_path / "a.png")
    r = load_raster(tmp_path / "a.png")
    assert r.shape == (4, 5) and not r.valid[0, 0] and r.valid.sum() == 19
    Image.fromarray(np.zeros((3, 3), np.uint16)).save(tmp_path / "b.tif")
    with pytest.raises(MapImportError, match="16-bit"):
        load_raster(tmp_path / "b.tif")
    (tmp_path / "c.png").write_bytes(b"nope")
    with pytest.raises(MapImportError, match="cannot read"):
        load_raster(tmp_path / "c.png")


def test_geotiff_tags_give_a_candidate_affine(tmp_path):
    import tifffile
    arr = np.zeros((6, 8, 3), np.uint8)
    tifffile.imwrite(tmp_path / "g.tif", arr, extratags=[(33550, "d", 3, (10.0, 10.0, 0.0), True),
                                                         (33922, "d", 6, (0, 0, 0, 1000.0, 5000.0, 0), True)])
    r = load_raster(tmp_path / "g.tif")
    assert r.geo_affine == (10.0, 0.0, 1000.0, 0.0, -10.0, 5000.0)


def test_crop_mask_flip_rotate_scale():
    r = raster(4, 6)
    c = crop(r, 1, 1, 5, 3)
    assert c.shape == (2, 4)
    for bad in ((-1, 0, 2, 2), (0, 0, 7, 2), (2, 0, 2, 2)):
        with pytest.raises(MapImportError):
            crop(r, *bad)
    m = mask_polygon(r, [(0, 0), (3, 0), (3, 4), (0, 4)], "inside")
    assert m.valid[:, :3].all() and not m.valid[:, 3:].any()
    assert mask_polygon(r, [(0, 0), (3, 0), (3, 4), (0, 4)], "outside").valid.sum() == 12
    f = flip(r, horizontal=True)
    assert tuple(f.rgb[0, 0]) == BLUE and tuple(f.rgb[0, -1]) == RED
    assert rotate90(r, 1).shape == (6, 4) and rotate90(r, 4).shape == (4, 6)
    s = scale(r, 2.0)
    assert s.shape == (8, 12) and {tuple(c) for c in s.rgb.reshape(-1, 3)} == {RED, BLUE}
    with pytest.raises(MapImportError):
        scale(r, 0)


def test_georeference_exact_affine_and_tolerance():
    # col -> X = 2*col + 100, row -> Y = -2*row + 500
    pts = [(0, 0, 100, 500), (10, 0, 120, 500), (0, 10, 100, 480)]
    g = georeference(pts, 1e-9)
    assert not g.redundant and g.max_residual < 1e-9
    np.testing.assert_allclose(g.to_map(5, 5), (110, 490))
    np.testing.assert_allclose(g.to_pixel(110, 490), (5, 5), atol=1e-9)
    # a fourth, wrong point
    bad = pts + [(10, 10, 120, 480 + 3)]
    with pytest.raises(MapImportError, match="control point"):
        georeference(bad, 0.5)
    g = georeference(bad, 5.0)
    assert g.redundant and max(g.residuals) == g.max_residual and g.max_residual > 0.5
    for p in (pts[:2], [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)], [(0, 0, 0, 0), (1, 0, 1, 0), (0, 1, float("nan"), 1)]):
        with pytest.raises(MapImportError):
            georeference(p, 1.0)
    with pytest.raises(MapImportError, match="tolerance"):
        georeference(pts, None)
    # rotated and sheared control points are fitted too
    th = np.radians(30)
    rot = [(c, r, 50 + 3 * (c * np.cos(th) - r * np.sin(th)), 70 + 3 * (c * np.sin(th) + r * np.cos(th)))
           for c, r in ((0, 0), (10, 0), (0, 10), (10, 10))]
    assert georeference(rot, 1e-6).max_residual < 1e-9


def test_legend_rules():
    Legend(((RED, 1), (BLUE, 2), (GREEN, 2)), 10).check_scheme((1, 2))
    with pytest.raises(MapImportError, match="two facies"):
        Legend(((RED, 1), (RED, 2)), 10)
    with pytest.raises(MapImportError, match="overlap"):
        Legend(((RED, 1), ((250, 0, 0), 2)), 10)
    Legend(((RED, 1), ((250, 0, 0), 1)), 10)                    # same facies may be close
    for bad in (-1, 500, float("nan"), None):
        with pytest.raises(MapImportError):
            Legend(((RED, 1),), bad)
    with pytest.raises(MapImportError, match="three whole numbers"):
        Legend((((300, 0, 0), 1),), 5)
    with pytest.raises(MapImportError):
        Legend((), 5)
    with pytest.raises(MapImportError, match="not in the project"):
        Legend(((RED, 9),), 5).check_scheme((1, 2))
    with pytest.raises(MapImportError, match="no colour on the legend"):
        Legend(((RED, 1),), 5).check_scheme((1, 2))
    Legend(((RED, 1),), 5, absent=(2,)).check_scheme((1, 2))
    with pytest.raises(MapImportError, match="also declared absent"):
        Legend(((RED, 1),), 5, absent=(1,)).check_scheme((1, 2))
    with pytest.raises(MapImportError, match="absent facies code 7"):
        Legend(((RED, 1),), 5, absent=(7,)).check_scheme((1,))


def test_classify_with_tolerance_and_unmatched():
    r = raster(2, 4)
    r.rgb[0, 0] = (250, 5, 3)                 # near red
    r.rgb[1, 3] = (0, 255, 0)                 # not on the legend
    r.valid[1, 0] = False
    cl, rep = classify_colours(r, Legend(((RED, 1), (BLUE, 2)), 20))
    assert cl[0, 0] == 1 and cl[1, 3] == NODATA and cl[1, 0] == NODATA
    assert rep["unmatched"] == 1 and rep["matched"] == 6 and rep["top_unmatched"][0][1] == 1
    cl2, _ = classify_colours(r, Legend(((RED, 1), (BLUE, 2)), 5))
    assert cl2[0, 0] == NODATA                # tolerance is applied


def test_fractions_sum_to_one_and_nodata_does_not_pull_to_zero():
    cl = np.full((9, 18), 1)
    cl[:, 9:] = 2
    cl[:, :2] = NODATA
    fr = to_fractions(cl, (1, 2), 0)
    assert np.isnan(fr[0, 0, 0]) and fr[0, 5, 0] == 1 and fr[0, 12, 1] == 1
    fr = to_fractions(cl, (1, 2), 2.0)
    check_fractions(fr)
    assert not np.isnan(fr[0, 0, 0])                 # edge pixels get a value from valid neighbours
    # across the 1|2 boundary the fraction of 1 falls monotonically 1 -> 0 and is 0.5 at the boundary
    prof = fr[4, 3:16, 0]
    assert np.all(np.diff(prof) <= 1e-12) and abs(fr[4, 8, 0] + fr[4, 9, 0] - 1) < 1e-3
    assert prof[0] > 0.99 and prof[-1] < 0.01
    # all-empty
    with pytest.raises(MapImportError, match="no pixel matched"):
        to_fractions(np.full((3, 3), NODATA), (1, 2), 1.0)
    for bad in (-1, None, float("nan")):
        with pytest.raises(MapImportError):
            to_fractions(cl, (1, 2), bad)


def test_check_fractions_refuses_bad_maps():
    ok = np.array([[[0.5, 0.5], [np.nan, np.nan]]])
    assert check_fractions(ok)["pixels"] == 1
    for bad in (np.array([[[0.5, 0.6]]]), np.array([[[1.2, -0.2]]])):
        with pytest.raises(MapImportError):
            check_fractions(bad)
    with pytest.raises(MapImportError):
        check_fractions(np.full((2, 2, 2), np.nan))


def test_suggested_sigma():
    assert suggest_sigma(10, 100) == 5.0 and suggest_sigma(10, 5) == 1.0
    with pytest.raises(MapImportError):
        suggest_sigma(0, 1)


def test_resample_matches_a_known_linear_field_and_reports_coverage():
    # fraction of class 1 rises linearly with col: f1 = col/(w-1) on pixel centres
    h, w = 30, 50
    f1 = np.tile(np.arange(w) / (w - 1), (h, 1))
    fr = np.stack([f1, 1 - f1], -1)
    # pixel size 10 m, origin (1000, 2000), north-up
    geo = georeference([(0, 0, 1000, 2000), (50, 0, 1500, 2000), (0, 30, 1000, 1700)], 1e-9)
    spec = grid_spec_from_cells(1100, 1700, 0, 4, 3, 100.0, 100.0)
    res = resample_to_grid(fr, (1, 2), geo, spec, 1.0)
    xc = 1100 + 50 + 100 * np.arange(4)
    exp = ((xc - 1000) / 10 - 0.5) / (w - 1)
    np.testing.assert_allclose(res.fractions[:, 1, 0], exp, atol=1e-9)
    np.testing.assert_allclose(res.fractions.sum(-1), 1.0, atol=1e-12)
    assert res.coverage == 1.0 and not res.uncovered.any()
    # a grid half outside the map: coverage reported and refused below the user's minimum
    spec2 = grid_spec_from_cells(1300, 1700, 0, 4, 3, 100.0, 100.0)
    with pytest.raises(MapImportError, match="covers 50.0%"):
        resample_to_grid(fr, (1, 2), geo, spec2, 0.9)
    res2 = resample_to_grid(fr, (1, 2), geo, spec2, 0.5)
    assert res2.coverage == 0.5 and res2.uncovered[2:].all() and np.isnan(res2.fractions[2:]).all()
    for bad in (0, 1.5, None):
        with pytest.raises(MapImportError):
            resample_to_grid(fr, (1, 2), geo, spec, bad)


def test_resample_respects_inactive_columns_and_rotation():
    h, w = 40, 40
    f1 = np.tile(np.arange(w) / (w - 1), (h, 1))
    fr = np.stack([f1, 1 - f1], -1)
    geo = georeference([(0, 0, 0, 400), (40, 0, 400, 400), (0, 40, 0, 0)], 1e-9)      # 10 m pixels, Y up
    spec = grid_spec_from_cells(100, 100, 0, 3, 3, 50.0, 50.0)
    spec.active_xy = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)
    res = resample_to_grid(fr, (1, 2), geo, spec, 1.0)
    assert res.active == 8 and np.isnan(res.fractions[1, 1]).all() and not np.isnan(res.fractions[0, 0]).any()
    # grid rotated 90 degrees: local x now runs along map -y... check the sample follows the frame
    spec = grid_spec_from_cells(200, 300, 90, 2, 2, 50.0, 50.0)
    X, Y = spec.frame.to_map(np.array([25.0]), np.array([25.0]))
    expected = (X[0] / 10 - 0.5) / (w - 1)
    got = resample_to_grid(fr, (1, 2), geo, spec, 1.0).fractions[0, 0, 0]
    assert abs(got - expected) < 1e-9
