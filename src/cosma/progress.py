"""Progress reporting for long runs: a cancel exception, a progress callback convention and an estimated-time tracker.

A progress callback is ``progress(label, done, total)``: ``done`` and ``total`` count work units (cells for the simulation engines),
``label`` says what is being worked on. A callback may raise ``Cancelled`` to stop the run; the engines call it between their
numerical kernels, so a stop takes effect after the interval or domain being simulated, not in the middle of it.

The estimated time is a straight-line extrapolation of the time spent so far over the fraction done. It does not know about work
that has not started (redrawn realizations, a slower interval later on), so it is an estimate only."""

from __future__ import annotations

import time


class Cancelled(Exception):
    """Raised by a progress callback to stop a run at the next safe point."""


def report(progress, label: str, done: float, total: float) -> None:
    if progress is not None:
        progress(label, float(done), float(total))


def format_duration(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "unknown"
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


class Tracker:
    """Overall progress of ``total_items`` items (realizations) whose individual progress is a fraction 0..1.

    ``skip`` items (for example realizations found in the cache) are finished at once and take no part in the estimate."""

    def __init__(self, total_items: int, skipped: int = 0, clock=time.monotonic):
        if total_items < 0 or not 0 <= skipped <= total_items:
            raise ValueError("bad item counts")
        self.total, self.skipped, self.clock = total_items, skipped, clock
        self.work = total_items - skipped                 # items that really have to be computed
        self.t0 = clock()
        self.finished = 0                                 # computed items finished
        self._hi = 0.0
        self.current = 0.0                                # fraction of the running item

    def set_skipped(self, skipped: int) -> None:
        if not 0 <= skipped <= self.total:
            raise ValueError("bad item counts")
        self.skipped, self.work = skipped, self.total - skipped

    def item_progress(self, fraction: float) -> None:
        self.current = min(max(self.current, float(fraction)), 0.999)     # never goes back (redraws restart the stages)

    def item_done(self) -> None:
        self.finished += 1
        self.current = 0.0

    def update_running(self, partial_sum: float) -> None:
        """Parallel runs: the summed fractions (each 0..1) of all items that are running at the moment."""
        self.current = float(partial_sum)

    @property
    def fraction(self) -> float:
        if self.work == 0:
            return 1.0
        f = min((self.finished + self.current) / self.work, 1.0)
        self._hi = max(self._hi, f)                       # the bar never moves back when an item leaves the running set
        return self._hi

    @property
    def elapsed(self) -> float:
        return self.clock() - self.t0

    def eta(self) -> float | None:
        """Seconds left, or None while there is nothing to extrapolate from."""
        f = self.fraction
        if self.work == 0 or f >= 1.0:
            return 0.0
        if f <= 0.0 or self.elapsed <= 0:
            return None
        return self.elapsed * (1.0 - f) / f

    def text(self) -> str:
        e = self.eta()
        done = f"{self.fraction * 100:.0f}% done, elapsed {format_duration(self.elapsed)}"
        if e is None:
            return done + ", estimating the remaining time..."
        return done + (", about " + format_duration(e) + " left" if e > 0 else "")
