import os
import sys

# conftest.py -> tests -> slime_mopd -> mopd (package root, holds the slime_mopd package)
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# The MOPD hooks import slime; fail loudly instead of silently skipping when
# slime is not installed in the active environment.
try:
    import slime  # noqa: F401
except ModuleNotFoundError as error:  # pragma: no cover - environment guard
    raise SystemExit("these tests require the slime package (e.g. `pip install -e . --no-deps`)") from error
