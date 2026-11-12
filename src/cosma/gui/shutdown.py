"""Orderly end of the Qt side of the program.

Dialogs are only released by Python's cyclic garbage collector (their signal connections hold references to them). If that is left
to interpreter exit it happens after Qt has been torn down, and the process can end with a crash on quit. ``release_widgets``
collects and deletes every widget while the QApplication is still alive; the application calls it after the event loop ends, the
test suite at the end of the session."""

from __future__ import annotations

import gc


def release_widgets(app=None) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    app = app or QApplication.instance()
    if app is None:
        return
    for _ in range(3):
        gc.collect()
        for w in app.topLevelWidgets():
            w.hide()                      # not close(): a close event may ask a question and wait for an answer
            w.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
