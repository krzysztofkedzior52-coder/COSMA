"""M0 spike: synthetic faulted box -> GRDECL -> independent readers.

Every expected number below is derived analytically from the builder's
parameters, not from any reader's output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xtgeo

from cosma.grid.synthetic import make_faulted_box
from cosma.io.grdecl import scan_keywords, write_grdecl

DX, DY, DZ = 100.0, 100.0, 10.0
TOP, DIP, THROW = 1500.0, 0.02, 25.0
FAULT_I = 3

resdata_grid = pytest.importorskip("resdata.grid")


def _unique_cube(nx, ny, nz, scale=1.0):
    """Value encodes (i, j, k) uniquely, so any axis-order bug is visible."""
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return scale * (1000.0 * i + 10.0 * j + k)


def _faulted(j_north=True, **kw):
    nx, ny, nz = 6, 4, 5
    return make_faulted_box(
        nx=nx, ny=ny, nz=nz, dx=DX, dy=DY, dz=DZ, top=TOP, dip_x=DIP,
        fault_i=FAULT_I, throw=THROW, j_north=j_north,
        poro=_unique_cube(nx, ny, nz, 1e-4) + 0.05,
        kx=_unique_cube(nx, ny, nz, 1.0) + 1.0,
        ky=_unique_cube(nx, ny, nz, 2.0) + 1.0,
        kz=_unique_cube(nx, ny, nz, 0.5) + 1.0,
        **kw,
    )


def _analytic_top_west_z(i, k):
    return TOP + DIP * (i * DX) + k * DZ + (THROW if i >= FAULT_I else 0.0)


@pytest.fixture(params=[True, False], ids=["j_north", "j_south"])
def written(request, tmp_path):
    grid = _faulted(j_north=request.param)
    path = write_grdecl(grid, tmp_path / "box.grdecl")
    return grid, path


def test_keyword_counts(written):
    grid, path = written
    c = scan_keywords(path)
    assert c["SPECGRID"] == 5
    assert c["COORD"] == 6 * (grid.nx + 1) * (grid.ny + 1)
    assert c["ZCORN"] == 8 * grid.nx * grid.ny * grid.nz
    for k in ("ACTNUM", "PORO", "PERMX", "PERMY", "PERMZ"):
        assert c[k] == grid.nx * grid.ny * grid.nz


def test_our_geometry_matches_analytic(written):
    grid, _ = written
    for k in range(grid.nz):
        for i in range(grid.nx):
            c = grid.cell_corners(i, 0, k)
            assert c[0, 2] == pytest.approx(_analytic_top_west_z(i, k))  # top SW
            assert c[1, 2] == pytest.approx(_analytic_top_west_z(i, k) + DIP * DX)  # top SE
            assert c[4, 2] == pytest.approx(_analytic_top_west_z(i, k) + DZ)  # bottom SW


def test_xtgeo_corners_match_ours_everywhere(written):
    grid, path = written
    xg = xtgeo.grid_from_file(path, fformat="grdecl")
    assert tuple(xg.dimensions) == (grid.nx, grid.ny, grid.nz)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                theirs = np.array(xg.get_xyz_cell_corners(ijk=(i + 1, j + 1, k + 1))).reshape(8, 3)
                np.testing.assert_allclose(theirs, grid.cell_corners(i, j, k), atol=1e-6)


def test_handedness(written):
    grid, path = written
    xg = xtgeo.grid_from_file(path, fformat="grdecl")
    # i east, k down: j north is left-handed, j south is right-handed
    assert xg.ijk_handedness == ("left" if grid.j_north else "right")
    assert float(xg.get_bulk_volume().values.min()) > 0.0


def test_fault_throw_across_fault_xtgeo(written):
    grid, path = written
    xg = xtgeo.grid_from_file(path, fformat="grdecl")

    def top_z(i, j, k, corner):  # corner 0 = top SW, 1 = top SE
        t = xg.get_xyz_cell_corners(ijk=(i + 1, j + 1, k + 1))
        return t[3 * corner + 2], t[3 * corner], t[3 * corner + 1]

    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(1, grid.nx):
                z_w, x_w, y_w = top_z(i, j, k, 0)
                z_e, x_e, y_e = top_z(i - 1, j, k, 1)
                assert (x_w, y_w) == pytest.approx((x_e, y_e))  # same pillar: vertical fault
                expected = THROW if i == FAULT_I else 0.0
                assert z_w - z_e == pytest.approx(expected, abs=1e-6)


def test_fault_throw_across_fault_resdata(written):
    grid, path = written
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rg = resdata_grid.Grid.load_from_grdecl(str(path))
        for k in range(grid.nz):
            for j in range(grid.ny):
                for i in range(1, grid.nx):
                    w = rg.get_cell_corner(0, ijk=(i, j, k))  # top SW of cell i
                    e = rg.get_cell_corner(1, ijk=(i - 1, j, k))  # top SE of cell i-1
                    assert w[:2] == pytest.approx(e[:2])
                    expected = THROW if i == FAULT_I else 0.0
                    assert w[2] - e[2] == pytest.approx(expected, abs=1e-6)


def test_pore_volume_three_ways(written):
    grid, path = written
    # Analytical: every cell is DX*DY*DZ regardless of dip and throw.
    cell_v = DX * DY * DZ
    expected_bulk = grid.nx * grid.ny * grid.nz * cell_v
    expected_pv = float(grid.props["PORO"].sum()) * cell_v

    # 1. our own cell volumes
    assert grid.total_pore_volume() == pytest.approx(expected_pv, rel=1e-9)

    # 2. xtgeo: geometry + PORO read back from the file
    xg = xtgeo.grid_from_file(path, fformat="grdecl")
    bulk = xg.get_bulk_volume().values
    poro = xtgeo.gridproperty_from_file(path, fformat="grdecl", name="PORO", grid=xg).values
    assert float(bulk.sum()) == pytest.approx(expected_bulk, rel=1e-9)
    assert float((bulk * poro).sum()) == pytest.approx(expected_pv, rel=1e-6)

    # 3. resdata: geometry only (property check done via xtgeo above)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rg = resdata_grid.Grid.load_from_grdecl(str(path))
        assert rg.get_num_active() == grid.nx * grid.ny * grid.nz
        vols = np.array([rg.cell_volume(global_index=n) for n in range(rg.get_global_size())])
    assert vols.sum() == pytest.approx(expected_bulk, rel=1e-9)
    # resdata global index is I-fastest: compare against PORO in the same order
    pv_res = float((vols * grid.props["PORO"].ravel(order="F")).sum())
    assert pv_res == pytest.approx(expected_pv, rel=1e-6)


@pytest.mark.parametrize("name", ["PORO", "PERMX", "PERMY", "PERMZ"])
def test_property_axis_order_roundtrip(written, name):
    grid, path = written
    xg = xtgeo.grid_from_file(path, fformat="grdecl")
    got = xtgeo.gridproperty_from_file(path, fformat="grdecl", name=name, grid=xg).values
    np.testing.assert_allclose(np.ma.filled(got, np.nan), grid.props[name], rtol=1e-7)


def test_inactive_cells(tmp_path):
    grid = _faulted()
    grid.actnum[2, 1, 3] = 0
    grid.actnum[0, 0, 0] = 0
    path = write_grdecl(grid, tmp_path / "inactive.grdecl")
    xg = xtgeo.grid_from_file(path, fformat="grdecl")
    act = xg.get_actnum().values
    np.testing.assert_array_equal(act, grid.actnum)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rg = resdata_grid.Grid.load_from_grdecl(str(path))
        assert rg.get_num_active() == grid.nx * grid.ny * grid.nz - 2


def test_no_fault_zero_throw_is_regular(tmp_path):
    grid = make_faulted_box(fault_i=None, throw=0.0, dip_x=0.0)
    path = write_grdecl(grid, tmp_path / "plain.grdecl", properties=["PORO"])
    assert "PERMX" not in scan_keywords(path)
    xg = xtgeo.grid_from_file(path, fformat="grdecl")
    assert float(xg.get_bulk_volume().values.sum()) == pytest.approx(
        grid.nx * grid.ny * grid.nz * DX * DY * DZ
    )


def test_opm_parser_reads_identical_arrays(written):
    """The OPM deck parser (the one OPM Flow uses) must accept the file and
    return the same numbers we wrote. This validates keyword syntax, array
    lengths and ordering at the parser level; it does not build an OPM grid."""
    opm_parser = pytest.importorskip("opm.io.parser")
    grid, path = written
    ctx = opm_parser.ParseContext(
        [
            ("PARSE_RANDOM_SLASH", opm_parser.action.ignore),
            ("PARSE_UNKNOWN_KEYWORD", opm_parser.action.ignore),
        ]
    )
    deck = opm_parser.Parser().parse(str(path), ctx)
    assert [deck["SPECGRID"][0][i].get_int(0) for i in range(3)] == [grid.nx, grid.ny, grid.nz]

    def doubles(name):
        return np.array(deck[name][0][0].get_raw_data_list())

    np.testing.assert_array_equal(doubles("COORD"), grid.coord.ravel())
    np.testing.assert_array_equal(doubles("ZCORN"), grid.zcorn.ravel())
    np.testing.assert_array_equal(
        np.array(deck["ACTNUM"][0][0].get_data_list()), grid.actnum.ravel(order="F")
    )
    for name in ("PORO", "PERMX", "PERMY", "PERMZ"):
        np.testing.assert_allclose(doubles(name), grid.props[name].ravel(order="F"), rtol=1e-7)
