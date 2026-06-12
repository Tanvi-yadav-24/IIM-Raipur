#!/usr/bin/env python3
"""
run_project.py
==============
Convenience entry-point for the Carhart Four-Factor Fund Analysis Pipeline.

This script is a thin wrapper around main.py that:
  1. Checks the Python version (≥ 3.9 required).
  2. Verifies all required data files exist.
  3. Verifies required packages are importable.
  4. Prints a pre-flight summary.
  5. Calls main.main() with any forwarded CLI arguments.

Usage
-----
    python run_project.py                  # full pipeline
    python run_project.py --no-plots       # skip charts (faster)
    python run_project.py --stage 6        # fast-resume from Stage 6
    python run_project.py --dry-run        # preflight only

This is the recommended single command to run the entire project.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_PYTHON = (3, 9)

_REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("numpy",               "numpy"),
    ("pandas",              "pandas"),
    ("scipy",               "scipy"),
    ("statsmodels",         "statsmodels"),
    ("matplotlib",          "matplotlib"),
    ("sklearn",             "scikit-learn"),
    ("openpyxl",            "openpyxl"),
    ("thefuzz",             "thefuzz"),
]

_REQUIRED_DATA: list[str] = [
    "data/monthly_nav_with_returns.csv",
    "data/all_active_large_cap_funds_with_returns.csv",
    "data/fund_expense_ratios.xlsx",
    "data/factor_data.csv",
]


def _check_python() -> bool:
    ok = sys.version_info >= _REQUIRED_PYTHON
    status = "✅" if ok else "❌"
    print(
        f"  {status} Python {sys.version_info.major}.{sys.version_info.minor}"
        f"  (required ≥ {_REQUIRED_PYTHON[0]}.{_REQUIRED_PYTHON[1]})"
    )
    return ok


def _check_packages() -> bool:
    all_ok = True
    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name}  →  pip install {pip_name}")
            all_ok = False
    return all_ok


def _check_data() -> bool:
    all_ok = True
    for rel_path in _REQUIRED_DATA:
        full = _PROJECT_ROOT / rel_path
        ok   = full.exists()
        size = f"{full.stat().st_size / 1024:.0f} KB" if ok else "MISSING"
        status = "✅" if ok else "❌"
        print(f"  {status} {rel_path:<55}  {size}")
        if not ok:
            all_ok = False
    return all_ok


def preflight() -> bool:
    """Run all pre-flight checks. Returns True if everything is ready."""
    print("\n" + "━" * 68)
    print("  CARHART FOUR-FACTOR FUND ANALYSIS — PRE-FLIGHT CHECK")
    print("━" * 68)

    print("\n  Python version:")
    py_ok = _check_python()

    print("\n  Required packages:")
    pkg_ok = _check_packages()

    print("\n  Required data files:")
    data_ok = _check_data()

    print("\n" + "━" * 68)
    all_ok = py_ok and pkg_ok and data_ok
    if all_ok:
        print("  ✅  All checks passed.  Starting pipeline …\n")
    else:
        print(
            "  ❌  Pre-flight failed.  Resolve the issues above before running.\n"
        )
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.perf_counter()

    if not preflight():
        sys.exit(1)

    from main import main
    main()
