"""Test session setup: offscreen Qt, and an orderly end of Qt so the run does not crash at exit (see cosma.gui.shutdown)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_sessionfinish(session, exitstatus):
    try:
        from cosma.gui.shutdown import release_widgets
    except ImportError:
        return
    release_widgets()
