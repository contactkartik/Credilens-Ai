"""
Pytest configuration for backend API tests.

Ensures the ``backend`` directory (for the ``app`` package) and the
repository root (for the ``ml`` package) are importable.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

for path in (str(BACKEND_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
