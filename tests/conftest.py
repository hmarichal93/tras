"""Shared pytest setup."""

import os

# Never open the online update dialog during tests (it is modal and blocks headless runs).
os.environ.setdefault("TRAS_NO_UPDATE_CHECK", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
