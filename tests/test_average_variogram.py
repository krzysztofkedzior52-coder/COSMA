import copy

import numpy as np
import pytest

from cosma.maps import MapGrid, krige, krige_map, layer_averages
from cosma.variogram import VariogramError
from cosma.variogram_session import VariogramSession
from synthetic_field import facies_project, synthetic


@pytest.fixture(scope="module")
def proj():
    return synthetic()


def u(pr):
    return pr.chart.match_name("Res")


def avg_session(pr, **kw):
    return VariogramSession(pr, u(pr), "PHI", average=True, **kw)


def test_average_session_data_are_one_value_per_well(proj):
    s = avg_session(proj)
    ld = layer_averages(proj, u(proj), "PHI")
    assert s.variable == "PHI (layer average)" and s.average and len(s.data.values) == 49
    assert np.allclose(s.data.values, ld.value) and np.allclose(s.data.points[:, 2], 0)
    assert s.data.used == {w: 1 for w in ld.wells} and s.data.mean_thickness == pytest.approx(50)
    assert s.variance == pytest.approx(np.var(ld.value))


def test_average_session_has_no_vertical_variogram(proj):
    s = avg_session(proj)
    with pytest.raises(VariogramError, match="no vertical variogram"):
        s.compute("vertical")
    with pytest.raises(VariogramError, match="major"):
        s.auto_fit()


def test_average_fit_azimuth_ranges_and_vertical_convention(proj):
    s = avg_session(proj)
    assert s.suggest_azimuth() is not None
    s.set_azimuth(30)
    s.compute("major", lag=150.0, nlags=5, tolerance_deg=20)
    s.compute("minor", lag=150.0, nlags=5, tolerance_deg=20)
    assert s.exp["major"].settings["vertical_tolerance"] == 1.0
    msgs = s.auto_fit()
    st = s.structures[0]
    assert st["vertical"] == pytest.approx(50.0) and any("maps do not use it" in m for m in msgs)
    assert st["major"] is not None
    m = s.model()
    assert m.variable == "PHI (layer average)" and m.transform == "none"
    assert s.nugget + st["contribution"] == pytest.approx(s.variance, rel=0.6)


def test_average_indicator_is_continuous_proportion():
    fp = facies_project(n=3, spacing=100.0)
    s = VariogramSession(fp, u(fp), "FACIES", indicator=2, average=True)
    assert s.variable == "FACIES[2] (layer average)" and s.transform == "none"
    assert np.all((s.data.values >= 0) & (s.data.values <= 1)) and s.data.values.std() > 0
    s.set_transform("normal_score")                                           # allowed: the data are continuous proportions
    assert s.transform == "normal_score" and s.values.shape == s.data.values.shape
    with pytest.raises(VariogramError, match="do not apply|does not apply"):
        VariogramSession(fp, u(fp), "FACIES", indicator=2, average=True, transform="indicator")


def test_average_warnings_and_min_coverage(proj):
    s = avg_session(proj, wells=[f"W{i}" for i in range(3)])
    assert any("3 pairs" in w for w in s.warnings()) and any("Only 3 wells" in w for w in s.warnings())
    pr = copy.deepcopy(proj)
    lg = pr.wells["W0"].logs["lg"]
    lg.curves["PHI"][(lg.md > 110)] = np.nan                                   # W0 now covers ~20% of the interval
    assert "W0" in avg_session(pr).data.skipped and "W0" not in avg_session(pr).data.used
    assert "W0" in avg_session(pr, min_coverage=0.1).data.used


def test_save_records_average_in_meta_and_does_not_collide_with_point_variogram():
    pr = copy.deepcopy(synthetic(n=4))
    a = avg_session(pr, wells=None)
    a.set_azimuth(30)
    a.compute("major", lag=150.0, nlags=3, tolerance_deg=20); a.compute("minor", lag=150.0, nlags=3, tolerance_deg=20)
    a.set_nugget(0.01); a.set_structure(0, contribution=0.2, major=400, minor=200, vertical=50)
    a.save("avg")
    p = VariogramSession(pr, u(pr), "PHI")
    p.set_nugget(0.1); p.set_structure(0, contribution=0.9, major=400, minor=200, vertical=20)
    assert not p.conflicts("pt")                                               # different property key: no conflict
    p.save("pt")
    assert pr.variogram_for(u(pr), "PHI (layer average)").variable == "PHI (layer average)"
    meta = pr.var_meta[(u(pr), "PHI (layer average)", "avg")]
    assert meta["average"] is True and meta["curve"] == "PHI" and meta["min_coverage"] == 0.5
    assert pr.var_meta[(u(pr), "PHI", "pt")]["average"] is False


def test_kriging_layer_averages_predicts_held_out_wells_better_than_the_mean(proj):
    s = avg_session(proj)
    s.set_azimuth(30)
    s.compute("major", lag=150.0, nlags=5, tolerance_deg=20); s.compute("minor", lag=150.0, nlags=5, tolerance_deg=20)
    s.auto_fit()
    st = s.structures[0]
    if st["minor"] is None:
        st["minor"] = st["major"] / 2
    model = s.model()
    ld = layer_averages(proj, u(proj), "PHI")
    err_k, err_m = [], []
    for i in range(len(ld.value)):
        keep = np.arange(len(ld.value)) != i
        est, _ = krige(model, ld.x[keep], ld.y[keep], ld.value[keep], [[ld.x[i], ld.y[i]]])
        err_k.append(est[0] - ld.value[i])
        err_m.append(ld.value[keep].mean() - ld.value[i])
    rk, rm = np.sqrt(np.mean(np.square(err_k))), np.sqrt(np.mean(np.square(err_m)))
    assert rk < rm                                                              # the spatial model has genuine predictive skill
    assert abs(np.mean(err_k)) < 0.5 * rm                                       # and is not biased
