"""The process must end cleanly after Qt dialogs were used (it used to crash at exit with all tests passed)."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6.QtWidgets")

CODE = """
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, {tests!r})
from PySide6.QtWidgets import QApplication
from cosma.gui.datacheck_dialog import DataCheckDialog
from test_datacheck import make_project, make_model


def use_a_dialog():                     # as in the application: nothing keeps the QApplication or the dialog afterwards
    app = QApplication.instance() or QApplication([])
    d = DataCheckDialog(make_project(), build=lambda p: make_model())
    d.edits["min_hard_cells"].setText("1")


use_a_dialog()
{release}
print("done")
"""


def run(release):
    tests = os.path.dirname(__file__)
    return subprocess.run([sys.executable, "-c", CODE.format(tests=tests, release=release)], capture_output=True, text=True, timeout=120)


def test_process_exits_cleanly_after_release_widgets():
    r = run("from cosma.gui.shutdown import release_widgets; release_widgets()")
    assert "done" in r.stdout and r.returncode == 0, (r.returncode, r.stderr[-500:])
