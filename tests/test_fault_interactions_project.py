import numpy as np
import pytest

from cosma.faults import Fault
from cosma.project import Project, ProjectError


def vf(name, p0, p1):
    return Fault(name, tuple(np.array([[x, y, z] for z in (0.0, 3000.0)]) for x, y in (p0, p1)))


def proj():
    pr = Project("p")
    pr.attach_faults([vf("A", (0, 0), (100, 100)), vf("B", (0, 100), (100, 0)), vf("C", (500, 0), (500, 100))])
    return pr


def test_nothing_is_guessed_depth_then_reach_then_answers():
    pr = proj()
    with pytest.raises(ProjectError, match="choose a trace depth"):
        pr.fault_age_order()
    for n in pr.faults:
        pr.set_fault_depth(n, 1500.0)
    with pytest.raises(ProjectError, match="not been checked"):
        pr.fault_age_order()
    ints, dropped, problems = pr.refresh_fault_interactions(5.0)
    assert [i.key for i in ints] == [("A", "B")] and not dropped and not problems
    with pytest.raises(ProjectError, match="confirm how these faults relate"):
        pr.fault_age_order()
    pr.confirm_fault_interaction(("A", "B"), "A")
    order = pr.fault_age_order()
    assert sorted(order) == ["A", "B", "C"] and order.index("A") < order.index("B")


def test_missing_depth_is_a_problem_not_a_silent_skip():
    pr = proj()
    pr.set_fault_depth("A", 1500.0)
    pr.set_fault_depth("B", 1500.0)
    _, _, problems = pr.refresh_fault_interactions(5.0)
    assert problems == ["C: no trace depth chosen"]


def test_changing_a_depth_discards_that_faults_answers():
    pr = proj()
    for n in pr.faults:
        pr.set_fault_depth(n, 1500.0)
    pr.refresh_fault_interactions(5.0)
    pr.confirm_fault_interaction(("A", "B"), "B")
    pr.set_fault_depth("C", 1500.0)             # unchanged value for C, unrelated: answer stays
    assert pr.fault_interactions[0].confirmed
    pr.set_fault_depth("A", 1600.0)
    assert pr.fault_interactions == []


def test_refresh_keeps_answers_when_nothing_moved():
    pr = proj()
    for n in pr.faults:
        pr.set_fault_depth(n, 1500.0)
    pr.refresh_fault_interactions(5.0)
    pr.confirm_fault_interaction(("A", "B"), "A")
    ints, dropped, _ = pr.refresh_fault_interactions(5.0)
    assert ints[0].answer == "A" and not dropped


def test_removing_or_replacing_a_fault_clears_its_depth_and_interactions():
    pr = proj()
    for n in pr.faults:
        pr.set_fault_depth(n, 1500.0)
    pr.refresh_fault_interactions(5.0)
    pr.remove_fault("B")
    assert "B" not in pr.fault_depths and pr.fault_interactions == []
    pr.set_fault_depth("A", 1500.0)
    from cosma.project import Action
    pr.attach_faults([vf("A", (0, 0), (50, 50))], {"fault|A": Action("replace")})
    assert "A" not in pr.fault_depths


def test_bad_depth_and_unknown_fault():
    pr = proj()
    with pytest.raises(ProjectError):
        pr.set_fault_depth("A", float("nan"))
    with pytest.raises(ProjectError):
        pr.set_fault_depth("Z", 1.0)
