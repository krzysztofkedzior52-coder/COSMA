import numpy as np
import pytest

from cosma.project import Action, ProjectError
from cosma.variogram import VariogramError
from cosma.variogram_session import VariogramSession
from synthetic_field import TRUTH, synthetic


@pytest.fixture(scope="module")
def proj():
    return synthetic()


def sess(proj, **kw):
    return VariogramSession(proj, proj.chart.match_name("Res"), "PHI", **kw)


def test_defaults_come_from_data_and_are_labelled_starting_points(proj):
    s = sess(proj)
    assert s.default_lag("vertical") == 2.0                  # logs sampled every 2 m
    assert s.default_lag("major") == 150.0                   # median nearest-well distance
    assert s.variance == pytest.approx(np.var(s.data.values))
    assert s.nugget == 0.0 and len(s.missing()) == 4
    with pytest.raises(VariogramError, match="not set yet"):
        s.model()                                           # nothing is guessed


def test_azimuth_inferred_from_the_data(proj):
    s = sess(proj)
    assert s.suggest_azimuth() == 30.0
    scan = s.scan_azimuths()
    assert scan[30.0] == min(v for v in scan.values() if v is not None)


def test_no_azimuth_suggestion_without_anisotropy():
    from cosma.variogram import Structure, VariogramModel
    iso = VariogramModel(0.1, (Structure("spherical", 0.9, 300, 300, 20),), 0).check()
    s = sess(synthetic(iso, seed=5))
    assert s.suggest_azimuth() is None                     # no clear anisotropy: no made-up direction


def test_auto_fit_recovers_vertical_and_orders_horizontal(proj):
    s = sess(proj)
    s.set_azimuth(30)
    s.compute("vertical", lag=2.0, nlags=10)
    s.compute("major", lag=150.0, nlags=5, tolerance_deg=20)
    s.compute("minor", lag=150.0, nlags=5, tolerance_deg=20)
    msgs = s.auto_fit()
    st = s.structures[0]
    assert st["vertical"] == pytest.approx(20.0, rel=0.15)
    assert s.nugget == pytest.approx(0.1, abs=0.1)
    assert st["contribution"] + s.nugget == pytest.approx(1.0, abs=0.2)
    assert st["major"] is not None and (st["minor"] is None or st["major"] > st["minor"])
    assert len(msgs) >= 3 and msgs[0].startswith("vertical")
    m = s.model()
    assert m.azimuth == 30 and m.variable == "PHI" and m.transform == "none"


def test_auto_fit_needs_vertical_and_reports_unfittable_direction(proj):
    s = sess(proj)
    with pytest.raises(VariogramError, match="vertical"):
        s.auto_fit()
    s.compute("vertical", lag=2.0, nlags=10)
    s.set_azimuth(30)
    s.compute("major", lag=1000.0, nlags=3, tolerance_deg=20)        # lags far beyond the data: empty bins
    msgs = s.auto_fit()
    assert any(m.startswith("major: not fitted") for m in msgs) and s.structures[0]["major"] is None
    assert any("minor: not computed" in m for m in msgs)
    with pytest.raises(VariogramError, match="not set yet"):
        s.model()


def test_manual_editing_validation_and_curves(proj):
    s = sess(proj)
    s.set_nugget(0.1)
    s.set_structure(0, type="exponential", contribution=0.9, major=500, minor=250, vertical=10)
    s.set_azimuth(45)
    m = s.model()
    assert m.structures[0].type == "exponential" and m.sill == pytest.approx(1.0)
    h, g = s.curve("major", 1000, n=11)
    assert h[5] == 500 and g[0] == 0 and g[5] == pytest.approx(0.1 + 0.9 * (1 - np.exp(-3.0)))
    hv, gv = s.curve("vertical", 20, n=3)
    assert gv[2] == pytest.approx(0.1 + 0.9 * (1 - np.exp(-3 * 20 / 10)))
    with pytest.raises(VariogramError):
        s.set_structure(0, type="cubic")
    with pytest.raises(VariogramError, match="unknown structure"):
        s.set_structure(0, rangee=3)
    s.add_structure("gaussian")
    s.add_structure()
    with pytest.raises(VariogramError, match="three"):
        s.add_structure()
    s.remove_structure(2)
    s.remove_structure(1)
    with pytest.raises(VariogramError, match="at least one"):
        s.remove_structure(0)
    s.set_structure(0, minor=900)                                 # minor > major
    with pytest.raises(VariogramError, match="minor range"):
        s.model()


def test_transform_switches_data_and_clears_experimentals(proj):
    s = sess(proj)
    s.compute("vertical", lag=2.0, nlags=5)
    s.set_transform("normal_score")
    assert not s.exp and s.variance == pytest.approx(1.0, abs=0.05) 
    s.set_nugget(0.1); s.set_structure(0, contribution=0.9, major=100, minor=50, vertical=10)
    assert s.model().transform == "normal_score"
    with pytest.raises(VariogramError):
        s.set_transform("log")


def test_warnings(proj):
    s = sess(proj)
    s.set_nugget(0.0); s.set_structure(0, contribution=3.0, major=100, minor=50, vertical=10)
    assert any("variance" in w for w in s.warnings())
    s.set_structure(0, contribution=s.variance)
    assert not any("variance" in w for w in s.warnings())
    one = VariogramSession(proj, proj.chart.match_name("Res"), "PHI", wells=["W0"])
    assert any("Only one well" in w for w in one.warnings())
    with pytest.raises(VariogramError, match="only one well|cannot suggest"):
        one.compute("major")
    few = VariogramSession(proj, proj.chart.match_name("Res"), "PHI", wells=["W0", "W1", "W2"])
    assert any("Only 3 wells" in w for w in few.warnings())


def test_save_and_conflicts(proj):
    import copy
    pr = copy.deepcopy(proj)
    s = sess(pr)
    s.set_nugget(0.1); s.set_structure(0, contribution=0.9, major=400, minor=200, vertical=20)
    with pytest.raises(ProjectError, match="name"):
        s.save(" ")
    r = s.save("first")
    assert pr.variogram_for(s.uid, "PHI").structures[0].range_major == 400 and "first" in r.applied[0]
    assert s.conflicts("second")
    with pytest.raises(ProjectError):
        s.save("second")
    s.set_structure(0, major=500)
    s.save("second", {s.conflicts("second")[0].key: Action("replace")})
    assert pr.variogram_for(s.uid, "PHI").structures[0].range_major == 500


def test_horizontal_fit_holds_vertical_nugget_and_sill(proj):
    from cosma.variogram import fit_direction
    s = sess(proj)
    s.set_azimuth(30)
    s.compute("vertical", lag=2.0, nlags=10); s.compute("major", lag=150.0, nlags=5, tolerance_deg=20)
    s.auto_fit()
    st = s.structures[0]
    sill = s.nugget + st["contribution"]
    assert fit_direction(s.exp["major"], "spherical", nugget=s.nugget, sill=sill, min_pairs=5)["range"] == pytest.approx(st["major"])
    free = fit_direction(s.exp["major"], "spherical", min_pairs=5)["range"]
    assert abs(free - st["major"]) > 1e-3                          # holding them genuinely changes the answer


def test_minor_longer_than_major_is_not_accepted_silently(proj):
    s = sess(proj)
    s.set_azimuth(120)                                             # 90 degrees off: the 'major' axis is really the short one
    s.compute("vertical", lag=2.0, nlags=10)
    s.compute("major", lag=150.0, nlags=5, tolerance_deg=20); s.compute("minor", lag=150.0, nlags=5, tolerance_deg=20)
    msgs = s.auto_fit()
    st = s.structures[0]
    assert any("exceeds the major" in m for m in msgs) and st["minor"] is None


def test_experimentals_survive_edits_that_do_not_change_them(proj):
    s = sess(proj)
    s.set_azimuth(30)
    s.compute("vertical", lag=2.0, nlags=5); s.compute("major", lag=150.0, nlags=3, tolerance_deg=20)
    s.set_azimuth(30); s.set_nugget(0.2)
    assert set(s.exp) == {"vertical", "major"}                         # same azimuth, nugget edit: nothing recomputed
    s.set_azimuth(31)
    assert set(s.exp) == {"vertical"}                                  # a new azimuth invalidates the horizontal ones only
