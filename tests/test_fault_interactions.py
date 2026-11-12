import numpy as np
import pytest

from cosma.fault_interactions import (SAME_AGE, UNRELATED, InteractionError, age_order, carry_answers,
                                      find_interactions, from_state, set_answer, to_state, unconfirmed)


def line(*pts):
    return np.array(pts, float)


def test_crossing_has_no_suggestion():
    ints = find_interactions({"A": line((0, 0), (100, 100)), "B": line((0, 100), (100, 0))}, tol=5)
    (i,) = ints
    assert i.kind == "crossing" and i.suggestion is None and not i.confirmed
    assert (i.x, i.y) == pytest.approx((50, 50))


def test_t_junction_suggests_the_ender_is_younger():
    ints = find_interactions({"A": line((0, 50), (100, 50)), "B": line((50, 50), (50, 150))}, tol=5)
    (i,) = ints
    assert i.kind == "abutting" and i.ends == "B" and i.suggestion == "A"
    assert "B ends against A" in i.describe()


def test_both_ends_at_the_crossing_is_not_a_t():
    # X where both traces have an end at the meeting point: cannot tell, no suggestion
    ints = find_interactions({"A": line((0, 0), (50, 50)), "B": line((50, 50), (100, 0))}, tol=5)
    (i,) = ints
    assert i.kind == "crossing" and i.suggestion is None


def test_near_miss_within_reach_is_reported_and_far_is_not():
    a, b = line((0, 50), (100, 50)), line((50, 60), (50, 150))
    (i,) = find_interactions({"A": a, "B": b}, tol=15)
    assert i.kind == "near" and i.ends == "B" and i.suggestion == "A" and "10 m short" in i.note
    assert find_interactions({"A": a, "B": b}, tol=5) == []


def test_parallel_faults_do_not_interact():
    assert find_interactions({"A": line((0, 0), (0, 100)), "B": line((200, 0), (200, 100))}, tol=10) == []


def test_tolerance_must_be_explicit_and_positive():
    with pytest.raises(InteractionError):
        find_interactions({}, tol=0)


def test_answers_and_validation():
    ints = find_interactions({"A": line((0, 0), (100, 100)), "B": line((0, 100), (100, 0))}, tol=5)
    assert len(unconfirmed(ints)) == 1
    ints = set_answer(ints, ("A", "B"), "B")
    assert not unconfirmed(ints) and ints[0].answer == "B"
    with pytest.raises(InteractionError, match="not one of"):
        set_answer(ints, ("A", "B"), "C")
    with pytest.raises(InteractionError, match="no interaction"):
        set_answer(ints, ("A", "Z"), "A")


def test_age_order_and_cycles():
    ints = [find_interactions({"A": line((0, 0), (10, 10)), "B": line((0, 10), (10, 0))}, tol=1)[0]]
    assert age_order(["A", "B"], set_answer(ints, ("A", "B"), "B")) == ["B", "A"]      # B older
    assert age_order(["A", "B"], set_answer(ints, ("A", "B"), "A")) == ["A", "B"]
    assert age_order(["B", "A"], set_answer(ints, ("A", "B"), SAME_AGE)) == ["B", "A"]  # no ordering from same age
    assert age_order(["B", "A"], set_answer(ints, ("A", "B"), UNRELATED)) == ["B", "A"]
    from cosma.fault_interactions import Interaction
    cyc = [Interaction("A", "B", "crossing", 0, 0, answer="A"), Interaction("B", "C", "crossing", 0, 0, answer="B"),
           Interaction("A", "C", "crossing", 0, 0, answer="C")]
    with pytest.raises(InteractionError, match="contradict"):
        age_order(["A", "B", "C"], cyc)


def test_transitive_order():
    from cosma.fault_interactions import Interaction
    rows = [Interaction("A", "B", "crossing", 0, 0, answer="B"), Interaction("B", "C", "crossing", 0, 0, answer="C")]
    assert age_order(["A", "B", "C"], rows) == ["C", "B", "A"]


def test_carry_answers_keeps_only_matching_geometry():
    t1 = {"A": line((0, 0), (100, 100)), "B": line((0, 100), (100, 0))}
    old = set_answer(find_interactions(t1, 5), ("A", "B"), "A")
    same, dropped = carry_answers(find_interactions(t1, 5), old, 5)
    assert same[0].answer == "A" and not dropped
    moved = {"A": line((0, 0), (100, 100)), "B": line((0, 140), (140, 0))}   # meets at (70, 70)
    new, dropped = carry_answers(find_interactions(moved, 5), old, 5)
    assert new[0].answer is None and len(dropped) == 1
    gone, dropped = carry_answers([], old, 5)
    assert gone == [] and len(dropped) == 1


def test_state_roundtrip_and_refusals():
    ints = set_answer(find_interactions({"A": line((0, 0), (100, 100)), "B": line((0, 100), (100, 0))}, 5), ("A", "B"), "B")
    assert from_state(to_state(ints)) == ints
    bad = to_state(ints)
    bad[0]["answer"] = "Q"
    with pytest.raises(InteractionError):
        from_state(bad)
    bad = to_state(ints)
    bad[0]["kind"] = "weird"
    with pytest.raises(InteractionError):
        from_state(bad)
    with pytest.raises(InteractionError):
        from_state([{"a": "A"}])


def test_age_order_changes_who_claims_crowded_corners():
    from cosma.faults import Fault
    from cosma.grid.boundary import grid_spec_from_cells
    from cosma.grid.fault_sampling import FaultSamplingError, corner_depths
    from cosma.surfaces import Surface

    spec = grid_spec_from_cells(0, 0, 0, 60, 4, 10.0, 10.0)
    n = 161
    X, Y = np.meshgrid(-200.0 + 12.5 * np.arange(n), -200.0 + 12.5 * np.arange(n), indexing="ij")
    surfs = [Surface(-200.0, -200.0, 12.5, 12.5, 0.0, 1, z0 + 0.0 * X + 20.0 * (X > 312.0) + 10.0 * (X > 330.0))
             for z0 in (1000.0, 1050.0)]

    def vf(name, x):
        return Fault(name, tuple(np.array([[x, y, z] for z in (0.0, 3000.0)]) for y in (-100.0, 500.0)))

    faults = {"F1": vf("F1", 312.0), "F2": vf("F2", 330.0)}
    depths = {"F1": 1500.0, "F2": 1500.0}
    a, ra = corner_depths(spec, surfs, faults, depths, age_order=["F1", "F2"])    # F2 younger
    b, rb = corner_depths(spec, surfs, faults, depths, age_order=["F2", "F1"])    # F1 younger
    assert not np.allclose(a[0], b[0])
    assert any("younger fault" in w for r in ra for w in r.warnings)
    with pytest.raises(FaultSamplingError, match="lacks"):
        corner_depths(spec, surfs, faults, depths, age_order=["F1"])


def test_t_junction_when_the_first_named_fault_is_the_one_that_ends():
    ints = find_interactions({"A": line((50, 50), (50, 150)), "B": line((0, 50), (100, 50))}, tol=5)
    (i,) = ints
    assert i.kind == "abutting" and i.ends == "A" and i.suggestion == "B"


def test_an_answer_is_dropped_when_the_kind_of_meeting_changes_at_the_same_place():
    crossing = {"A": line((0, 0), (100, 100)), "B": line((0, 100), (100, 0))}
    old = set_answer(find_interactions(crossing, 5), ("A", "B"), "A")
    # now B stops at the meeting point: same place, but an abutting relation, which must be decided afresh
    abutting = {"A": line((0, 0), (100, 100)), "B": line((0, 100), (50, 50))}
    new, dropped = carry_answers(find_interactions(abutting, 5), old, 5)
    assert new[0].kind == "abutting" and new[0].answer is None and len(dropped) == 1
