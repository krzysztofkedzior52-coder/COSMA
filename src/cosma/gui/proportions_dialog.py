"""Vertical proportion curves per interval. Nothing is preselected: the position mode, bin size, smoothing and the
minimum amount of data are typed by the user, and every interval gets an explicit source (from the wells, or a curve
typed here). Save is possible only after Compute has run on exactly the current entries."""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..proportions import Curve, ProportionError, ProportionSetup, Settings, build_curves, to_state

CHOOSE = "- choose -"
M_REL, M_DEPTH = "relative position (0 at the top of the interval, 1 at its base)", "true depth from the top (m)"
S_WELLS, S_USER = "from the wells", "my curve"


def _default_build(project):
    from ..grid_definition import GridDefinitionError, build_model, required_items
    gd = project.grid_def
    if gd is None:
        raise ProportionError("define the grid first (Grid menu)")
    missing = required_items(project, gd)
    if missing:
        raise ProportionError("the grid cannot be built yet (" + missing[0] + (", ..." if len(missing) > 1 else "") + ")")
    try:
        return build_model(project, gd)
    except (GridDefinitionError, ValueError) as exc:
        raise ProportionError(f"the grid could not be built: {exc}") from None


class ProportionsDialog(QDialog):
    def __init__(self, project, parent: QWidget | None = None, build=None):
        super().__init__(parent)
        self.project = project
        self.build = build or _default_build
        self.setWindowTitle("Vertical proportion curves")
        self.setMinimumSize(980, 900)
        lay = QVBoxLayout(self)
        note = QLabel("A vertical proportion curve gives the share of each facies at each position inside an interval. "
                      "Say how positions are binned, how much data a bin needs, and where each interval's curve comes "
                      "from, then press Compute. An interval without enough data is reported and stops the build; nothing "
                      "is borrowed from another interval. Typed curves must sum to 1 in every bin (they are not rescaled).")
        note.setWordWrap(True)
        lay.addWidget(note)
        grid = QGridLayout()
        self.mode = QComboBox()
        self.mode.addItems([CHOOSE, M_REL, M_DEPTH])
        self.nbins, self.bin_size, self.window = QLineEdit(), QLineEdit(), QLineEdit()
        self.min_wells, self.min_cells = QLineEdit(), QLineEdit()
        for r, (label, w) in enumerate((("Position:", self.mode), ("Number of bins (relative position):", self.nbins),
                                        ("Bin size in m (true depth):", self.bin_size),
                                        ("Smoothing window in bins (odd; 1 = none):", self.window),
                                        ("Minimum wells with hard cells in an interval:", self.min_wells),
                                        ("Minimum hard cells inside every bin's smoothing window:", self.min_cells))):
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(w, r, 1)
        lay.addLayout(grid)
        lay.addWidget(QLabel("Source of the curve of each interval:"))
        names = [iv.name for iv in project.chart.intervals] if project.chart is not None else []
        self.names = names
        self.table = QTableWidget(len(names), 2)
        self.table.setHorizontalHeaderLabels(["Interval", "Source"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(160)
        self.sources = {}
        for r, n in enumerate(names):
            self.table.setItem(r, 0, QTableWidgetItem(n))
            self.table.item(r, 0).setFlags(self.table.item(r, 0).flags() & ~self.table.item(r, 0).flags().ItemIsEditable)
            c = QComboBox()
            c.addItems([CHOOSE, S_WELLS, S_USER])
            c.currentTextChanged.connect(self._edited)
            self.table.setCellWidget(r, 1, c)
            self.sources[n] = c
        lay.addWidget(self.table)
        row = QHBoxLayout()
        row.addWidget(QLabel("Typed curve of interval:"))
        self.curve_for = QComboBox()
        row.addWidget(self.curve_for, 1)
        add, rem = QPushButton("Add bin"), QPushButton("Remove last bin")
        add.clicked.connect(self._add_bin)
        rem.clicked.connect(self._remove_bin)
        row.addWidget(add)
        row.addWidget(rem)
        lay.addLayout(row)
        scheme = project.facies_setup.scheme if project.facies_setup is not None else ()
        self.codes = tuple(f.code for f in scheme)
        self.curve_table = QTableWidget(0, len(scheme))
        self.curve_table.setHorizontalHeaderLabels([f.name for f in scheme])
        self.curve_table.setMaximumHeight(170)
        lay.addWidget(self.curve_table)
        self.typed: dict[str, list] = {}
        self._showing = None
        self.curve_for.currentTextChanged.connect(self._show_curve)
        self.curve_table.itemChanged.connect(self._curve_edited)
        self.problem = QLabel("")
        self.problem.setStyleSheet("color: #b00020")
        self.problem.setWordWrap(True)
        lay.addWidget(self.problem)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(210)
        lay.addWidget(self.report)
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(8, 3))
            self.canvas = FigureCanvasQTAgg(self.fig)
            lay.addWidget(self.canvas, 1)
        except Exception:                      # plotting is a convenience; the report carries every number
            self.fig = self.canvas = None
        row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self.compute)
        row.addWidget(self.compute_btn)
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        row.addWidget(self.save_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        self._ran = None
        self.saved = False
        self._load(project.proportions)
        for e in (self.nbins, self.bin_size, self.window, self.min_wells, self.min_cells):
            e.textChanged.connect(self._edited)
        self.mode.currentTextChanged.connect(self._edited)
        self._refresh_curve_choices()

    # -- editing -----------------------------------------------------------------------------------------------------
    def _edited(self, *_):
        self._ran = None
        self.save_btn.setEnabled(False)
        self._refresh_curve_choices()

    def _refresh_curve_choices(self):
        want = [n for n in self.names if self.sources[n].currentText() == S_USER]
        have = [self.curve_for.itemText(i) for i in range(self.curve_for.count())]
        if want != have:
            self.curve_for.blockSignals(True)
            self.curve_for.clear()
            self.curve_for.addItems(want)
            self.curve_for.blockSignals(False)
            self._show_curve(self.curve_for.currentText())

    def _show_curve(self, name):
        self._showing = None
        rows = self.typed.get(name, []) if name else []
        self.curve_table.blockSignals(True)
        self.curve_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.curve_table.setItem(r, c, QTableWidgetItem(v))
        self.curve_table.blockSignals(False)
        self._showing = name or None

    def _curve_edited(self, *_):
        if self._showing is None:
            return
        self.typed[self._showing] = [[(self.curve_table.item(r, c).text() if self.curve_table.item(r, c) else "")
                                      for c in range(self.curve_table.columnCount())] for r in range(self.curve_table.rowCount())]
        self._edited()

    def _add_bin(self):
        if self._showing is not None:
            self.curve_table.setRowCount(self.curve_table.rowCount() + 1)
            self._curve_edited()

    def _remove_bin(self):
        if self._showing is not None and self.curve_table.rowCount():
            self.curve_table.setRowCount(self.curve_table.rowCount() - 1)
            self._curve_edited()

    # -- stored setup -> form ----------------------------------------------------------------------------------------
    def _load(self, s):
        if s is None:
            return
        st = s.settings
        self.mode.setCurrentText(M_REL if st.mode == "relative" else M_DEPTH)
        self.nbins.setText("" if st.nbins is None else str(st.nbins))
        self.bin_size.setText("" if st.bin_size is None else f"{st.bin_size:g}")
        self.window.setText(str(st.window))
        self.min_wells.setText(str(st.min_wells))
        self.min_cells.setText(str(st.min_cells))
        for iv, src in s.choices:
            if iv in self.sources:
                self.sources[iv].setCurrentText(S_WELLS if src == "wells" else S_USER)
        for c in s.user_curves:
            self.typed[c.interval] = [[f"{v:g}" for v in row] for row in c.values]

    # -- form -> setup -----------------------------------------------------------------------------------------------
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
            raise ProportionError(f"{what}: enter {'a whole number' if integer else 'a number'}") from None

    def read_setup(self) -> ProportionSetup:
        m = self.mode.currentText()
        if m == CHOOSE:
            raise ProportionError("choose how positions are measured")
        mode = "relative" if m == M_REL else "depth"
        if mode == "relative":
            st = Settings(mode, nbins=self._num(self.nbins.text(), "number of bins", True),
                          window=self._num(self.window.text(), "smoothing window", True),
                          min_wells=self._num(self.min_wells.text(), "minimum wells", True),
                          min_cells=self._num(self.min_cells.text(), "minimum hard cells", True))
        else:
            st = Settings(mode, bin_size=self._num(self.bin_size.text(), "bin size"),
                          window=self._num(self.window.text(), "smoothing window", True),
                          min_wells=self._num(self.min_wells.text(), "minimum wells", True),
                          min_cells=self._num(self.min_cells.text(), "minimum hard cells", True))
        if not self.names:
            raise ProportionError("define the stratigraphic chart first")
        if not self.codes:
            raise ProportionError("define the facies first (Facies menu)")
        choices, curves = [], []
        for n in self.names:
            s = self.sources[n].currentText()
            if s == CHOOSE:
                raise ProportionError(f"choose a source for interval {n!r}")
            if s == S_WELLS:
                choices.append((n, "wells"))
                continue
            choices.append((n, "user"))
            rows = self.typed.get(n, [])
            if not rows:
                raise ProportionError(f"interval {n!r}: type at least one bin of the curve")
            vals = []
            for r, row in enumerate(rows):
                if any(not x.strip() for x in row):
                    raise ProportionError(f"interval {n!r}, bin {r + 1}: fill in a proportion for every facies")
                vals.append(tuple(self._num(x, f"interval {n!r}, bin {r + 1}") for x in row))
            size = (1.0 / len(rows)) if mode == "relative" else st.bin_size
            curves.append(Curve(n, mode, size, self.codes, tuple(vals), "user"))
        return ProportionSetup(st, tuple(choices), tuple(curves))

    # -- compute and save --------------------------------------------------------------------------------------------
    def compute(self):
        self.problem.setText("")
        self._ran = None
        self.save_btn.setEnabled(False)
        try:
            s = self.read_setup()
            if self.project.facies_setup is None:
                raise ProportionError("define the facies first (Facies menu)")
            model = self.build(self.project)
            res = build_curves(self.project, model, s)
        except ProportionError as exc:
            self.problem.setText(str(exc))
            self.report.setPlainText("")
            return
        self.report.setPlainText(res.summary())
        self._plot(res)
        self._ran = to_state(s)
        self.save_btn.setEnabled(True)
        self.result = res

    def _plot(self, res):
        if self.fig is None:
            return
        import numpy as np
        self.fig.clear()
        ivs = [r for r in res.intervals if r.curve is not None]
        if not ivs:
            self.canvas.draw_idle()
            return
        axs = self.fig.subplots(1, len(ivs), squeeze=False)[0]
        names = {f.code: f.name for f in res.scheme}
        for ax, r in zip(axs, ivs):
            v, c = r.curve.array(), r.curve
            nb = c.nbins
            edges = np.arange(nb + 1) * c.bin_size
            bottom = np.zeros(nb)
            for f, code in enumerate(c.codes):
                ax.bar(edges[:-1], v[:, f], width=c.bin_size, bottom=bottom, align="edge", label=names[code], edgecolor="white",
                       linewidth=0.3)
                bottom += v[:, f]
            ax.set_title(f"{r.interval} ({r.source})", fontsize=9)
            ax.set_xlabel("relative position" if c.mode == "relative" else "depth from top (m)", fontsize=8)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
        axs[0].legend(fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save(self) -> bool:
        try:
            s = self.read_setup()
        except ProportionError as exc:
            self.problem.setText(str(exc))
            return False
        if self._ran is None or to_state(s) != self._ran:
            self.problem.setText("Compute on these entries first: what is saved must be what was computed.")
            self.save_btn.setEnabled(False)
            return False
        self.project.set_proportions(s)
        self.saved = True
        self.problem.setText("")
        return True
