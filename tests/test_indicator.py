import copy

import numpy as np
import pytest

from cosma.project import ProjectError
from cosma.variogram import VariogramError, experimental_variogram, fit_direction
from cosma.variogram_data import collect
from cosma.variogram_session import VariogramSession
from synthetic_field import facies_project, synthetic


@pytest.fixture(scope="module")
def fp():
    return facies_project()


def u(pr):
    return pr.chart.match_name("Res")


def test_facies_codes_detected_only_for_whole_number_curves(fp):
    d = collect(fp, u(fp), "FACIES")
    codes = d.facies_codes()
    assert set(codes) == {1, 2, 3} and sum(codes.values()) == pytest.approx(1.0)
    assert codes[1] == pytest.approx(np.mean(d.values == 1))
    assert collect(fp, u(fp), "PHI").facies_codes() is None
    assert d.as_indicator(2).facies_codes() is None                          # an indicator is not a category curve


def test_as_indicator_values_and_refusals(fp):
    d = collect(fp, u(fp), "FACIES")
    i = d.as_indicator(2)
    assert set(np.unique(i.values)) == {0.0, 1.0} and i.indicator == 2 and i.variable == "FACIES[2]"
    assert np.array_equal(i.values, (d.values == 2).astype(float))
    assert np.array_equal(i.points, d.points) and d.indicator is None             # original untouched
    with pytest.raises(ProjectError, match="does not occur"):
        d.as_indicator(9)
    with pytest.raises(ProjectError, match="does not look like a facies"):
        collect(fp, u(fp), "PHI").as_indicator(1)
    one = copy.deepcopy(d)
    one.values[:] = 4.0
    with pytest.raises(ProjectError, match="without variation"):
        one.as_indicator(4)
    assert i.transformed("indicator") is i.values and i.transformed("none") is i.values
    with pytest.raises(ProjectError, match="normal-score"):
        i.transformed("normal_score")
    with pytest.raises(ProjectError, match="needs a facies code"):
        d.transformed("indicator")


def test_indicator_variance_is_p_one_minus_p():
    pr = facies_project(n=2, spacing=100.0)
    s = VariogramSession(pr, u(pr), "FACIES", indicator=1)
    p = np.mean(s.data.values)
    assert s.variance == pytest.approx(p * (1 - p))                          # variance of 0/1 data is p(1-p)


def test_indicator_variogram_sill_is_p_one_minus_p(fp):
    s = VariogramSession(fp, u(fp), "FACIES", indicator=2)
    p = float(np.mean(s.data.values))
    s.compute("vertical", lag=2.0, nlags=10)
    f = fit_direction(s.exp["vertical"], "spherical", min_pairs=5)
    assert f["nugget"] + f["contribution"] == pytest.approx(p * (1 - p), rel=0.25)
    assert f["range"] > 4                                                      # facies bodies are continuous at the log scale


def test_session_for_indicator(fp):
    s = VariogramSession(fp, u(fp), "FACIES", indicator=3)
    assert s.transform == "indicator" and s.variable == "FACIES[3]" and s.codes.keys() == {1, 2, 3}
    with pytest.raises(VariogramError, match="indicator variogram"):
        s.set_transform("normal_score")
    with pytest.raises(VariogramError, match="needs a facies code"):
        VariogramSession(fp, u(fp), "FACIES", transform="indicator")
    with pytest.raises(ProjectError):
        VariogramSession(fp, u(fp), "PHI", indicator=1)
    s.compute("vertical", lag=2.0, nlags=10)
    s.auto_fit()
    s.set_azimuth(0)
    st = s.structures[0]
    st["major"], st["minor"] = 400.0, 200.0
    m = s.model()
    assert m.transform == "indicator" and m.variable == "FACIES[3]"


def test_indicator_variograms_stored_per_facies_independently():
    pr = copy.deepcopy(facies_project())
    outs = {}
    for c in (1, 2):
        s = VariogramSession(pr, u(pr), "FACIES", indicator=c)
        s.compute("vertical", lag=2.0, nlags=10)
        s.auto_fit()
        s.set_structure(0, major=400.0, minor=200.0)
        s.save(f"ind{c}")
        outs[c] = pr.variogram_for(u(pr), f"FACIES[{c}]")
    assert outs[1] is not outs[2] and outs[1].variable == "FACIES[1]" and outs[2].variable == "FACIES[2]"
    assert pr.variogram_for(u(pr), "FACIES[3]") is None
    assert not VariogramSession(pr, u(pr), "FACIES", indicator=3).conflicts("x")      # a different facies never conflicts
    assert s.conflicts("x")                                                         # the same facies does


def test_rare_facies_warning(fp):
    pr = copy.deepcopy(fp)
    for rec in pr.wells.values():
        c = rec.logs["lg"].curves["FACIES"]
        c[(c == 3)] = 2.0
    first = next(iter(pr.wells.values()))
    first.logs["lg"].curves["FACIES"][55:57] = 3.0                          # a handful of facies-3 samples
    s = VariogramSession(pr, u(pr), "FACIES", indicator=3)
    assert any("very noisy" in w for w in s.warnings())
    s2 = VariogramSession(fp, u(fp), "FACIES", indicator=2)
    assert not any("very noisy" in w for w in s2.warnings())


def test_load_existing_model_requires_same_data(fp):
    s = VariogramSession(fp, u(fp), "FACIES", indicator=2)
    s.compute("vertical", lag=2.0, nlags=10); s.auto_fit(); s.set_structure(0, major=400.0, minor=200.0)
    s.save("m")
    saved = s.project.variogram_for(u(fp), "FACIES[2]")
    t = VariogramSession(fp, u(fp), "FACIES", indicator=2)
    t.load(saved)
    assert t.model() == saved and not t.exp
    with pytest.raises(VariogramError, match="computed on"):
        VariogramSession(fp, u(fp), "PHI").load(saved)


def test_many_whole_number_values_are_not_a_facies_curve(fp):
    d = collect(fp, u(fp), "FACIES")
    d.values = np.arange(d.values.size, dtype=float)                    # 1274 distinct integers: a depth-like curve, not categories
    assert d.facies_codes() is None
    d.values = np.arange(d.values.size, dtype=float) % 50
    assert len(d.facies_codes()) == 50                                  # the limit is inclusive
    assert d.facies_codes(max_codes=49) is None


def test_available_properties_lists_sources_for_logs_and_core(fp):
    import copy
    from cosma.variogram_data import available_properties
    from cosma.wells.model import RawCore
    pr = copy.deepcopy(fp)
    assert available_properties(pr) == {"FACIES": ["lg"], "PHI": ["lg"]}
    pr.wells["W0"].core["c1"] = RawCore(np.array([160.0]), "md", {"POR": np.array([0.2]), "KH": np.array([5.0])})
    pr.wells["W1"].core["c2"] = RawCore(np.array([160.0]), "md", {"POR": np.array([0.2])})
    assert available_properties(pr, "core") == {"KH": ["c1"], "POR": ["c1", "c2"]}
    with pytest.raises(ProjectError):
        available_properties(pr, "seismic")


def _count3(pr, k):
    """Make exactly k facies-3 samples inside the interval (all others become 2) and one facies 1 so 2 stays partial."""
    pr = copy.deepcopy(pr)
    n = 0
    for rec in pr.wells.values():
        c = rec.logs["lg"].curves["FACIES"]
        inside = np.where(np.isfinite(c))[0]
        c[inside] = 2.0
        for i in inside:
            if n < k:
                c[i] = 3.0
                n += 1
    return pr


def test_rare_facies_thresholds_are_exact():
    base = facies_project(n=2, spacing=100.0)
    warn = lambda pr, c=3: any("very noisy" in w for w in VariogramSession(pr, u(pr), "FACIES", indicator=c).warnings())
    assert warn(_count3(base, 29)) and not warn(_count3(base, 30))              # 104 samples: count rule (k < 30)
    big = facies_project(n=6, spacing=100.0, dz=1.0)                              # 1836 samples
    assert warn(_count3(big, 33)) and not warn(_count3(big, 40))                  # k >= 30 both; 33/1836 = 1.8% < 2% rare, 40/1836 = 2.2% fine
    nearly_all = _count3(base, 1)                                                 # facies 2 everywhere but one sample
    assert warn(nearly_all, 2)                                                    # 103/104 = 99% > 98%: indicator of the dominant code
