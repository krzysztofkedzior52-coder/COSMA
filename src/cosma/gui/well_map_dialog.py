"""Well map: the wells of the project in plan view, with faults, boundary, grid outline and an optional surface behind."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from ..project import Project
from ..wellmap import MapData, collect

COLOURS = {"core and logs": "#c5221f", "core": "#e8710a", "logs": "#1a73e8", "no logs or core": "#80868b"}


def draw_map(ax, data: MapData, *, names=True, paths=True, sticks=True, traces=True, boundary=True, grid=True):
    """Draw `data` on a matplotlib axes (map coordinates, equal aspect). Returns the artists that need a legend."""
    ax.clear()
    if data.surface is not None:
        s = data.surface
        cs = ax.contourf(s["X"], s["Y"], np.ma.masked_invalid(s["Z"]), levels=14, cmap="viridis_r", alpha=0.55)
        cb = ax.figure.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(f"depth of {s['label']} (m)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if boundary and data.boundary is not None:
        ax.plot(data.boundary[:, 0], data.boundary[:, 1], "-", color="#188038", lw=1.6, label="boundary")
    if grid and data.grid_outline is not None:
        ax.plot(data.grid_outline[:, 0], data.grid_outline[:, 1], "--", color="#5f6368", lw=1.2, label="grid outline")
    if sticks:
        first = True
        for arrs in data.fault_sticks.values():
            for a in arrs:
                ax.plot(a[:, 0], a[:, 1], "-", color="#9aa0a6", lw=0.7, label="fault sticks" if first else None)
                first = False
    if traces:
        first = True
        for n, t in data.fault_traces.items():
            ax.plot(t[:, 0], t[:, 1], "-", color="#a50e0e", lw=2.0, label="fault trace" if first else None)
            first = False
            mid = len(t) // 2
            ax.annotate(f"{n} ({data.fault_depths[n]:g} m)", (t[mid, 0], t[mid, 1]), fontsize=7, color="#a50e0e",
                        xytext=(4, 4), textcoords="offset points")
    seen = set()
    for w in data.wells:
        c = COLOURS[w.data]
        if paths and w.path is not None and w.deviated:
            ax.plot(w.path[:, 0], w.path[:, 1], "-", color=c, lw=1.4)
            ax.plot([w.bottom[0]], [w.bottom[1]], "s", color=c, ms=3.5)
        ax.plot([w.x], [w.y], "*" if w.is_type_well else "o", color=c, ms=11 if w.is_type_well else 6.5, mec="black", mew=0.6,
                label=w.data if w.data not in seen else None, zorder=5)
        seen.add(w.data)
        if names:
            ax.annotate(w.name, (w.x, w.y), fontsize=7, xytext=(5, 4), textcoords="offset points", zorder=6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(fontsize=7, loc="best", framealpha=0.85)


class WellMapDialog(QDialog):
    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Well map")
        self.resize(980, 720)
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.boxes = {}
        for key, text in (("names", "Well names"), ("paths", "Deviated paths"), ("boundary", "Boundary"), ("grid", "Grid outline"),
                          ("sticks", "Fault sticks"), ("traces", "Fault traces")):
            b = QCheckBox(text)
            b.setChecked(True)
            b.toggled.connect(self.redraw)
            bar.addWidget(b)
            self.boxes[key] = b
        bar.addWidget(QLabel("Background:"))
        self.background = QComboBox()
        self.background.addItem("none", None)
        ch = project.chart
        if ch is not None:
            for h in ch.horizons():
                if project.surfaces.get(h.uid):
                    self.background.addItem(f"depth of {h.label}", h.uid)
        self.background.currentIndexChanged.connect(self.redraw)
        bar.addWidget(self.background)
        bar.addStretch(1)
        save = QPushButton("Save image...")
        save.clicked.connect(self.save_image)
        bar.addWidget(save)
        lay.addLayout(bar)
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)
        self.info = QLabel("Click a well to see its details.")
        self.info.setWordWrap(True)
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #b36b00;")
        lay.addWidget(self.info)
        lay.addWidget(self.notes)
        self.canvas.mpl_connect("button_press_event", self._click)
        self.data = None
        self.redraw()

    def options(self) -> dict:
        return {k: b.isChecked() for k, b in self.boxes.items()}

    def redraw(self, *_):
        self.data = collect(self.project, horizon_uid=self.background.currentData())
        draw_map(self.ax, self.data, **self.options())
        self.notes.setText("<br>".join(self.data.notes))
        if not self.data.wells:
            self.info.setText("No well has a position yet: import well headers (Import > Well headers...).")
        self.canvas.draw_idle()

    def nearest_well(self, x, y, tol_fraction=0.03):
        """The well whose wellhead is closest to (x, y) in map units, if within a few percent of the view width."""
        if not self.data or not self.data.wells:
            return None
        x0, x1 = self.ax.get_xlim()
        tol = abs(x1 - x0) * tol_fraction
        d = [np.hypot(w.x - x, w.y - y) for w in self.data.wells]
        k = int(np.argmin(d))
        return self.data.wells[k] if d[k] <= tol else None

    def _click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or self.canvas.toolbar is not None and getattr(self.canvas.toolbar, "mode", ""):
            return
        w = self.nearest_well(ev.xdata, ev.ydata)
        if w is not None:
            self.info.setText(w.describe())

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save well map", "well_map.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.fig.savefig(path, dpi=200)
