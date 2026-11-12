"""Simulate facies (SIS, or truncated Gaussian for ordered facies), look at one realization, and store the settings.
Nothing is preset except that the method list starts on SIS (the project's default method). Save is possible only after a
simulation with exactly the entries being saved."""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..facies_sim import FaciesSimError, FaciesSimSetup, run, to_state
from ..sis import SisSettings
from ..tgs import TgsSettings
from .progress_widget import ProgressPanel
from .proportions_dialog import _default_build

SIS, TGS = "SIS - sequential indicator simulation (default)", "Truncated Gaussian (ordered facies)"
CHOOSE = "- choose -"
KEEP, STOP = "keep the curve alone", "stop with an error"


class FaciesSimDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, build=None, runner=None):
        super().__init__(parent)
        self.project = project
        self.build = build or _default_build
        self._native = runner is None
        self.runner = runner or run
        self.setWindowTitle("Simulate facies")
        self.setMinimumSize(1000, 900)
        self.scheme = tuple(project.facies_setup.scheme) if project.facies_setup is not None else ()
        lay = QVBoxLayout(self)
        note = QLabel("Facies are simulated per interval from the vertical proportion curves (combined with proportion maps where "
                      "they exist) and the hard facies at the wells. SIS needs an indicator variogram named after each facies on "
                      "every interval. Truncated Gaussian needs one Gaussian (normal-score) variogram and an order of the facies. "
                      "One realization is shown; it is not forced to its target proportions.")
        note.setWordWrap(True)
        lay.addWidget(note)
        g = QGridLayout()
        self.method = QComboBox()
        self.method.addItems([SIS, TGS])
        self.nn, self.search, self.levels, self.seed = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        rows = (("Method:", self.method), ("Neighbours used per node:", self.nn),
                ("Search radius (multiple of the largest variogram range):", self.search),
                ("Multigrid levels (1 = none):", self.levels), ("Seed (whole number):", self.seed))
        for r, (t, w) in enumerate(rows):
            g.addWidget(QLabel(t), r, 0)
            g.addWidget(w, r, 1)
        lay.addLayout(g)
        self.tgs_box = QWidget()
        tg = QGridLayout(self.tgs_box)
        tg.addWidget(QLabel("Order of the facies (one end to the other):"), 0, 0)
        self.order = QListWidget()
        self.order.addItems([f.name for f in self.scheme])
        self.order.setMaximumHeight(90)
        tg.addWidget(self.order, 0, 1, 2, 1)
        up, dn = QPushButton("Move up"), QPushButton("Move down")
        up.clicked.connect(lambda: self._move(-1))
        dn.clicked.connect(lambda: self._move(1))
        tg.addWidget(up, 0, 2)
        tg.addWidget(dn, 1, 2)
        self.order_ok = QCheckBox("This is the facies order")
        tg.addWidget(self.order_ok, 2, 1)
        self.gname, self.gibbs = QLineEdit(), QLineEdit()
        tg.addWidget(QLabel("Name of the Gaussian variogram on each interval:"), 3, 0)
        tg.addWidget(self.gname, 3, 1)
        tg.addWidget(QLabel("Gibbs sweeps for hard data (0 only without hard data):"), 4, 0)
        tg.addWidget(self.gibbs, 4, 1)
        lay.addWidget(self.tgs_box)
        self.map_box = QWidget()
        mg = QGridLayout(self.map_box)
        mg.addWidget(QLabel("Proportion maps exist. Where a map has no data:"), 0, 0)
        self.outside = QComboBox()
        self.outside.addItems([CHOOSE, KEEP, STOP])
        mg.addWidget(self.outside, 0, 1)
        mg.addWidget(QLabel("Largest share (0-1) of cells curve and map may leave empty:"), 1, 0)
        self.incompat = QLineEdit()
        mg.addWidget(self.incompat, 1, 1)
        lay.addWidget(self.map_box)
        self.panel = ProgressPanel()
        self.panel.finished.connect(self._finished)
        lay.addWidget(self.panel)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(130)
        lay.addWidget(self.report)
        view = QHBoxLayout()
        view.addWidget(QLabel("Layer:"))
        self.layer = QSpinBox()
        view.addWidget(self.layer)
        view.addWidget(QLabel("Row (j) for the section:"))
        self.row = QSpinBox()
        view.addWidget(self.row)
        view.addStretch(1)
        lay.addLayout(view)
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(9, 3.6))
            self.canvas = FigureCanvasQTAgg(self.fig)
            lay.addWidget(self.canvas, 1)
        except Exception:
            self.fig = self.canvas = None
        row = QHBoxLayout()
        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.simulate)
        self.save_btn = QPushButton("Save settings")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for w in (self.sim_btn, self.save_btn, close):
            row.addWidget(w)
        lay.addLayout(row)
        self.result = None
        self._ran = None
        self.saved = False
        self._load(project.facies_sim)
        self.tgs_box.setVisible(self.method.currentText() == TGS)
        self.map_box.setVisible(bool(project.proportion_maps))
        self.method.currentTextChanged.connect(lambda t: self.tgs_box.setVisible(t == TGS))
        for e in (self.nn, self.search, self.levels, self.seed, self.gname, self.gibbs, self.incompat):
            e.textChanged.connect(self._edited)
        for c in (self.method, self.outside):
            c.currentTextChanged.connect(self._edited)
        self.order_ok.toggled.connect(self._edited)
        self.layer.valueChanged.connect(self._draw)
        self.row.valueChanged.connect(self._draw)

    # -- form ---------------------------------------------------------------------------------------------------------
    def _edited(self, *_):
        self._ran = None
        self.save_btn.setEnabled(False)

    def _move(self, d):
        r = self.order.currentRow()
        if r < 0 or not 0 <= r + d < self.order.count():
            return
        it = self.order.takeItem(r)
        self.order.insertItem(r + d, it)
        self.order.setCurrentRow(r + d)
        self.order_ok.setChecked(False)
        self._edited()

    def _load(self, s):
        if s is None:
            return
        self.seed.setText(str(s.seed))
        if s.outside_map:
            self.outside.setCurrentText(KEEP if s.outside_map == "curve" else STOP)
        if s.max_incompatible is not None:
            self.incompat.setText(f"{s.max_incompatible:g}")
        if s.method == "sis":
            self.method.setCurrentText(SIS)
            t = s.sis
        else:
            self.method.setCurrentText(TGS)
            t = s.tgs
            names = {f.code: f.name for f in self.scheme}
            self.order.clear()
            self.order.addItems([names[c] for c in t.order])
            self.order_ok.setChecked(True)
            self.gname.setText(t.variogram)
            self.gibbs.setText(str(t.gibbs_sweeps))
        self.nn.setText(str(t.max_neighbours))
        self.search.setText(f"{t.search_factor:g}")
        self.levels.setText(str(t.levels))

    @staticmethod
    def _num(text, what, integer=False):
        try:
            v = float(text)
            if integer:
                if v != int(v):
                    raise ValueError
                return int(v)
            return v
        except ValueError:
            raise FaciesSimError(f"{what}: enter {'a whole number' if integer else 'a number'}") from None

    def read_setup(self) -> FaciesSimSetup:
        try:
            nn = self._num(self.nn.text(), "neighbours", True)
            sf = self._num(self.search.text(), "search radius")
            lv = self._num(self.levels.text(), "multigrid levels", True)
            seed = self._num(self.seed.text(), "seed", True)
            outside = mx = None
            if self.project.proportion_maps:
                o = self.outside.currentText()
                if o == CHOOSE:
                    raise FaciesSimError("choose what to do where a proportion map has no data")
                outside = "curve" if o == KEEP else "stop"
                mx = self._num(self.incompat.text(), "largest share of empty cells")
            if self.method.currentText() == SIS:
                return FaciesSimSetup("sis", SisSettings(nn, sf, lv), None, seed, outside, mx)
            if not self.order_ok.isChecked():
                raise FaciesSimError("confirm the facies order")
            codes = {f.name: f.code for f in self.scheme}
            order = tuple(codes[self.order.item(i).text()] for i in range(self.order.count()))
            gs = self._num(self.gibbs.text(), "Gibbs sweeps", True)
            return FaciesSimSetup("tgs", None, TgsSettings(order, self.gname.text().strip(), nn, sf, lv, gs), seed, outside, mx)
        except FaciesSimError:
            raise
        except ValueError as exc:                    # SisError / TgsError are ValueErrors
            raise FaciesSimError(str(exc)) from None

    # -- actions ------------------------------------------------------------------------------------------------------
    def simulate(self):
        """Builds the grid and simulates in a worker thread (the window stays usable; Stop works between intervals)."""
        self.problem.setText("")
        self._ran = None
        self.save_btn.setEnabled(False)
        try:
            s = self.read_setup()
        except FaciesSimError as exc:
            self.problem.setText(str(exc))
            return
        self._setup = s

        def job(ctx):
            ctx.progress("building the grid", 0.0)
            model = self.build(self.project)
            if self._native:
                res, notes = self.runner(self.project, model, s, progress=lambda l, d, t: ctx.progress(l, d / t if t else 0.0))
            else:
                res, notes = self.runner(self.project, model, s)
            return res, model, notes
        self.sim_btn.setEnabled(False)
        self.panel.start(job, 1, 0)

    def _finished(self):
        self.sim_btn.setEnabled(True)
        kind, val = self.panel.outcome
        if kind == "failed":
            self.problem.setText(val)
            return
        if kind == "cancelled":
            self.problem.setText("Stopped; nothing was simulated.")
            return
        res, model, notes = val
        s = self._setup
        self.result, self.model = res, model
        self.report.setPlainText("\n".join([*notes, res.report]))
        self._ran = to_state(s)
        self.save_btn.setEnabled(True)
        nz, ny = res.facies.shape[2], res.facies.shape[1]
        self.layer.setRange(0, nz - 1)
        self.row.setRange(0, ny - 1)
        self._draw()

    def done(self, code):
        if self.panel.running:
            self.panel.stop()
            self.panel.wait()
        super().done(code)

    def _draw(self, *_):
        if self.fig is None or self.result is None:
            return
        import numpy as np
        from matplotlib.colors import ListedColormap
        f = self.result.facies
        codes = [x.code for x in self.scheme]
        cmap = ListedColormap(["#dddddd"] + [f"C{i}" for i in range(len(codes))])
        idx = np.zeros(f.shape, int)
        for q, c in enumerate(codes):
            idx[f == c] = q + 1
        self.fig.clear()
        a1, a2 = self.fig.subplots(1, 2)
        a1.imshow(idx[:, :, self.layer.value()].T, origin="lower", cmap=cmap, vmin=0, vmax=len(codes), interpolation="nearest")
        a1.set_title(f"layer {self.layer.value()}", fontsize=8)
        a2.imshow(idx[:, self.row.value(), :].T, origin="upper", cmap=cmap, vmin=0, vmax=len(codes), interpolation="nearest", aspect="auto")
        a2.set_title(f"section at j = {self.row.value()}", fontsize=8)
        for a in (a1, a2):
            a.tick_params(labelsize=6)
        from matplotlib.patches import Patch
        a1.legend(handles=[Patch(color=f"C{i}", label=x.name) for i, x in enumerate(self.scheme)], fontsize=6, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save(self) -> bool:
        try:
            s = self.read_setup()
        except FaciesSimError as exc:
            self.problem.setText(str(exc))
            return False
        if self._ran is None or to_state(s) != self._ran:
            self.problem.setText("Simulate with these entries first: what is saved must be what was run.")
            self.save_btn.setEnabled(False)
            return False
        self.project.set_facies_sim(s)
        self.saved = True
        self.problem.setText("")
        return True
