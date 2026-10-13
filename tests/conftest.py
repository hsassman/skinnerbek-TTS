"""Shared fixtures. `pythonpath = ["scripts"]` in pyproject.toml puts the
project modules on sys.path, so tests can `import textprep, common, ...`.
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
