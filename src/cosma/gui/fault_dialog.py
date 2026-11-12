"""View a fault: sticks in plan, the plane fit, and the plan-view trace at a chosen depth."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ..faults import Fault, FaultError, VERTICAL_WARN_BELOW


def fit_text(f: Fault) -> str:
    fit = f.fit()
    lines = [f"{f.n_sticks} stick(s), {f.n_points} points"]
    if fit.dip_deg is None:
        lines.append(f"Dip: NOT DETERMINED - {fit.reason}")
    else:
        lines.append(f"Dip {fit.dip_deg:.1f} deg"
                     + (f" towards {fit.dip_direction_deg:.0f} deg" if fit.dip_direction_deg is not None else " (vertical: no dip direction)")
                     + (f"; strike line {fit.axis_azimuth_deg:.0f}/{(fit.axis_azimuth_deg + 180) % 360:.0f} deg" if fit.axis_azimuth_deg is not None else ""))
        lines.append(f"Plane misfit: RMS {fit.rms:.3g} m, largest {fit.max_residual:.3g} m")
        lines += [f"Warning: {w}" for w in fit.warnings]
        if fit.dip_deg < VERTICAL_WARN_BELOW:
            lines.append("Warning: COSMA v1 treats faults as vertical, so the trace is taken at ONE depth. "
                         "Choose a depth that represents the reservoir.")
    z0, z1 = f.bounds()[4:]
    lines.append(f"Depths of the sticks: {z0:.1f} to {z1:.1f} m")
    return "\n".join(lines)


def draw_fault(ax, f: Fault, trace=None):
    ax.clear()
    for s in f.sticks:
        ax.plot(s[:, 0], s[:, 1], "-", color="#9aa0a6", lw=0.8)
    if trace is not None and len(trace.xy):
        ax.plot(trace.xy[:, 0], trace.xy[:, 1], "o-", color="#c5221f", lw=1.8, ms=4)
        for k in trace.skipped:
            s = f.sticks[k]
            ax.plot(s[:, 0], s[:, 1], "-", color="#e8a900", lw=1.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f.name + (f" - trace at {trace.depth:g} m" if trace is not None else ""), fontsize=9, loc="left")
    ax.tick_params(labelsize=7)


class FaultViewDialog(QDialog):
    def __init__(self, fault: Fault, parent: QWidget | None = None):
        super().__init__(parent)
        self.fault = fault
        self.trace = None
        self.setWindowTitle(f"Fault - {fault.name}")
        self.setMinimumSize(900, 520)
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 0)
        self.summary = QLabel(fit_text(fault))
        self.summary.setWordWrap(True)
        left.addWidget(self.summary)
        row = QHBoxLayout()
        row.addWidget(QLabel("Trace at depth (m, positive down)"))
        z0, z1 = fault.bounds()[4:]
        self.depth = QLineEdit(f"{(z0 + z1) / 2:.1f}")
        row.addWidget(self.depth)
        b = QPushButton("Show trace")
        b.clicked.connect(self.show_trace)
        row.addWidget(b)
        left.addLayout(row)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        left.addWidget(self.problem)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        left.addWidget(self.notes, 1)
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        root.addWidget(self.canvas, 1)
        self.show_trace()

    def show_trace(self):
        self.problem.setText("")
        try:
            d = float(self.depth.text())
            self.trace = self.fault.trace_at(d)
        except (ValueError, FaultError) as exc:
            self.problem.setText("The depth must be a number." if isinstance(exc, ValueError) and not isinstance(exc, FaultError) else str(exc))
            return
        tr = self.trace
        lines = [f"{len(tr.used)} of {self.fault.n_sticks} stick(s) give a point at {d:g} m; trace length {tr.length:.1f} m"]
        lines += [f"stick {k + 1}: {why}" for k, why in tr.skipped.items()]
        lines += [f"Warning: {w}" for w in tr.warnings]
        self.notes.setPlainText("\n".join(lines))
        draw_fault(self.ax, self.fault, tr)
        self.canvas.draw_idle()
