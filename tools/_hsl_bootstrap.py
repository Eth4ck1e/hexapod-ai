"""tools/_hsl_bootstrap.py

Sets up the Windows DLL search path so CasADi's bundled IPOPT can load
the HSL solvers (currently MA27 — that's all the Coin-HSL Archive
2023.11.17 actually exports despite the header advertising MA57/86/97).

Import this BEFORE `import casadi`. No-op on non-Windows platforms.

After import, pass `linear_solver="ma27"` to IPOPT options to use HSL.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HSL_BIN     = _PROJECT_ROOT / "vendor" / "coinhsl" / "bin"


def setup_hsl_path() -> bool:
    """Returns True if HSL is available and the path was set up."""
    if sys.platform != "win32":
        return False
    if not _HSL_BIN.is_dir():
        return False
    hsl_str = str(_HSL_BIN)
    # PATH so LoadLibraryEx finds dependent DLLs (libgfortran, libopenblas, ...)
    if hsl_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hsl_str + os.pathsep + os.environ.get("PATH", "")
    # Also add to the Python-level DLL search path (Python 3.8+ on Windows).
    try:
        os.add_dll_directory(hsl_str)
    except (AttributeError, FileNotFoundError):
        pass
    return True


# Set up immediately on import — IPOPT loads HSL lazily on first
# `linear_solver=ma27` solve, so this just needs to happen before then.
HSL_AVAILABLE = setup_hsl_path()
