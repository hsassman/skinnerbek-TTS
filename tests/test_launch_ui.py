"""Regression checks for the UI launcher."""
import sys
import types

import launch_ui


def test_launcher_forwards_dtype_to_app(monkeypatch):
    captured = []
    monkeypatch.setitem(sys.modules, 'app', types.SimpleNamespace(
        main=lambda: captured.extend(sys.argv) or 0))
    monkeypatch.setattr(sys, 'argv', ['launch_ui.py', '--dtype', 'bfloat16'])
    assert launch_ui.main() == 0
    assert captured[captured.index('--dtype') + 1] == 'bfloat16'
