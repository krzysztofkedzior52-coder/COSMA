import os

import numpy as np
import pytest

from cosma.progress import Cancelled, Tracker, format_duration, report


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_format_duration():
    assert format_duration(0) == "0 s" and format_duration(59.4) == "59 s"
    assert format_duration(61) == "1 min 01 s" and format_duration(3599) == "59 min 59 s"
    assert format_duration(3600) == "1 h 00 min" and format_duration(7384) == "2 h 03 min"
    assert format_duration(float("nan")) == "unknown" and format_duration(-1) == "unknown"


def test_tracker_extrapolates_linearly_and_never_goes_back():
    c = Clock()
    t = Tracker(4, clock=c)
    assert t.eta() is None and "estimating" in t.text()                 # nothing to extrapolate from
    c.t += 10
    t.item_progress(0.5)                                               # 0.5 of 4 items in 10 s -> 70 s left
    assert t.fraction == pytest.approx(0.125) and t.eta() == pytest.approx(70.0)
    t.item_progress(0.2)                                               # a redraw restarting its stages must not move the bar back
    assert t.fraction == pytest.approx(0.125)
    t.item_progress(5.0)
    assert t.current < 1.0 and t.fraction < 0.25                       # an item is never complete before item_done
    c.t += 30
    t.item_done()
    assert t.fraction == pytest.approx(0.25) and t.eta() == pytest.approx(120.0)
    assert "25% done" in t.text() and "2 min 00 s left" in t.text()
    for _ in range(3):
        t.item_done()
    assert t.fraction == 1.0 and t.eta() == 0.0


def test_cached_items_take_no_part_in_the_estimate():
    c = Clock()
    t = Tracker(10, skipped=8, clock=c)
    c.t += 5
    t.item_done()                                                       # one of the two computed items took 5 s
    assert t.fraction == pytest.approx(0.5) and t.eta() == pytest.approx(5.0)
    t.set_skipped(10)
    assert t.fraction == 1.0 and t.eta() == 0.0
    with pytest.raises(ValueError):
        t.set_skipped(11)
    with pytest.raises(ValueError):
        Tracker(3, skipped=4)
    allc = Tracker(5, skipped=5, clock=c)
    assert allc.fraction == 1.0 and allc.eta() == 0.0


def test_report_is_a_noop_without_callback_and_cancel_propagates():
    report(None, "x", 1, 2)
    seen = []
    report(lambda *a: seen.append(a), "x", 1, 2)
    assert seen == [("x", 1.0, 2.0)]

    def stop(*a):
        raise Cancelled()
    with pytest.raises(Cancelled):
        report(stop, "x", 0, 1)
