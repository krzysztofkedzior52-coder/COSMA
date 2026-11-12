"""A progress bar with a stage line, elapsed and estimated remaining time, and a Stop button, for runs in a worker thread.

The job function runs in a QThread and receives a ``ctx``: ``ctx.progress(label, fraction)`` (fraction 0..1 of the item being
worked on; it raises ``Cancelled`` after Stop is pressed) and ``ctx.item_done()`` when an item (a realization) is finished. The window
stays responsive; the bar is refreshed by a timer, not by every callback. Stop takes effect at the next safe point (the engines
check between intervals and domains), so the last interval being simulated is finished first."""

from __future__ import annotations

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from ..progress import Cancelled, Tracker


class Ctx:
    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.label = ""
        self.cancel_requested = False

    def progress(self, label: str, fraction: float = 0.0) -> None:
        if self.cancel_requested:
            raise Cancelled()
        self.label = label
        self.tracker.item_progress(fraction)

    def item_done(self) -> None:
        self.tracker.item_done()


class Worker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn, ctx: Ctx):
        super().__init__()
        self.fn, self.ctx = fn, ctx

    def run(self):
        try:
            out = self.fn(self.ctx)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:                       # reported in the dialog, never lost in the thread
            self.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.succeeded.emit(out)


class ProgressPanel(QWidget):
    """Hidden until a job runs. ``running`` is True from start() until the worker has finished."""

    finished = Signal()

    def __init__(self, parent=None, interval_ms: int = 250):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        row.addWidget(self.bar, 1)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        self.stage = QLabel("")
        self.time = QLabel("")
        lay.addWidget(self.stage)
        lay.addWidget(self.time)
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.refresh)
        self.worker = None
        self.ctx = None
        self.outcome = None            # ('ok', result) | ('failed', message) | ('cancelled', None)
        self.setVisible(False)

    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def start(self, fn, items: int = 1, skipped: int = 0, clock=None):
        if self.running:
            raise RuntimeError("a run is already in progress")
        self.tracker = Tracker(items, skipped, *([clock] if clock else []))
        self.ctx = Ctx(self.tracker)
        self.outcome = None
        self.worker = Worker(fn, self.ctx)
        self.worker.succeeded.connect(lambda r: self._end(("ok", r)))
        self.worker.failed.connect(lambda m: self._end(("failed", m)))
        self.worker.cancelled.connect(lambda: self._end(("cancelled", None)))
        self.stop_btn.setEnabled(True)
        self.setVisible(True)
        self.refresh()
        self.timer.start()
        self.worker.start()

    def stop(self):
        if self.ctx is not None and self.running:
            self.ctx.cancel_requested = True
            self.stop_btn.setEnabled(False)
            self.stage.setText("Stopping at the next safe point (the interval being simulated is finished first)...")

    def refresh(self):
        if self.ctx is None:
            return
        self.bar.setValue(int(round(self.tracker.fraction * 1000)))
        if not self.ctx.cancel_requested:
            self.stage.setText(self.ctx.label)
        self.time.setText(self.tracker.text())

    def _end(self, outcome):
        self.outcome = outcome
        self.timer.stop()
        self.refresh()
        if outcome[0] == "ok":
            self.bar.setValue(1000)
            self.time.setText(f"finished in {self._fmt(self.tracker.elapsed)}")
        elif outcome[0] == "cancelled":
            self.time.setText(f"stopped after {self._fmt(self.tracker.elapsed)}")
        self.stop_btn.setEnabled(False)
        self.finished.emit()

    @staticmethod
    def _fmt(s):
        from ..progress import format_duration
        return format_duration(s)

    def wait(self, timeout_ms: int = 600000) -> None:
        """Blocks (processing events) until the worker is done; for tests and scripted use."""
        from PySide6.QtWidgets import QApplication
        if self.worker is not None:
            self.worker.wait(timeout_ms)
        QApplication.processEvents()
