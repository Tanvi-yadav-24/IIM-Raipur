"""
data_loader.py
==============
Raw-file ingestion layer for the Carhart Four-Factor Fund Analysis Pipeline.

Responsibility
--------------
This module's sole job is to read every input file from disk, perform strict
schema validation, log a concise ingestion report, and return clean DataFrames
to the caller.  It applies NO business logic — no merging, no normalisation, no
return calculations.  Those responsibilities belong to preprocessing.py and
later modules.

Fail-fast philosophy
--------------------
Every loader raises a descriptive exception the moment it encounters an
unrecoverable problem (missing file, schema mismatch, completely empty frame).
Recoverable issues (unexpected extra columns, minor dtype warnings) are logged
as WARNING and the pipeline continues.

Ground-truth discoveries (from dataset inspection — DO NOT REMOVE)
------------------------------------------------------------------
FILE 1 — monthly_nav_with_returns.csv
  • 4 396 rows × 6 cols.  33 PASSIVE funds.
  • Date range: 2006-04-28 → 2026-04-30.
  • NO FUND_TYPE column — labelling is done in preprocessing.py.
  • Month column uses full English names ("July") — pipeline uses Date only.
  • Monthly_Return has exactly 1 NaN per fund (first obs) — expected.

FILE 2 — factor_data.csv
  • Only 3 rows covering 1993-10 to 1993-12.
  • ZERO temporal overlap with NAV data (2006–2026).
  • MF and RF are NaN for row 0.
  • Pipeline uses synthetic factors; this file is loaded but its coverage is
    flagged immediately so the user is aware.

FILE 3 — fund_expense_ratios.xlsx
  • 68 rows × 2 cols.  No missing values.  No duplicates.
  • Covers both active (35) and passive (33) funds.
  • 3 funds have no matching NAV entry; merge losses handled in factor_merge.

FILE 4 — all_active_large_cap_funds_with_returns.csv
  • 4 210 rows × 6 cols.  33 ACTIVE funds.
  • Date range: 2013-01-31 → 2026-05-19.
  • Month column uses YYYY-MM format ("2013-01") — different from FILE 1.
  • Monthly_Return has exactly 1 NaN per fund (first obs) — expected.
  • 2 funds have a single date beyond 2026-04-30; flagged in logs.

Module layout
-------------
  1. _read_csv_safe / _read_excel_safe  — defensive low-level readers
  2. load_passive_nav                   — FILE 1 loader
  3. load_factor_data                   — FILE 2 loader (with coverage check)
  4. load_expense_ratios                — FILE 3 loader
  5. load_active_nav                    — FILE 4 loader
  6. load_all_data                      — orchestrator that calls all four
  7. Self-test  (python src/data_loader.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Path fix so module can be run directly as `python src/data_loader.py` ─────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ACTIVE_NAV_FILE,
    ANALYSIS_END,
    ANALYSIS_START,
    COL_DATE_RAW,
    COL_ER_FUND_RAW,
    COL_ER_VALUE_RAW,
    COL_FACTOR_DATE_RAW,
    COL_FUND_NAME_RAW,
    COL_HML_RAW,
    COL_MF_RAW,
    COL_MONTH_RAW,
    COL_NAV,
    COL_RETURN_PCT_RAW,
    COL_RETURN_RAW,
    COL_RF_RAW,
    COL_SMB_RAW,
    COL_WML_RAW,
    EXPENSE_RATIO_FILE,
    FACTOR_DATA_FILE,
    FUND_TYPE_ACTIVE,
    FUND_TYPE_PASSIVE,
    PASSIVE_NAV_FILE,
)
from src.utils import (
    check_file_exists,
    coerce_numeric_columns,
    report_duplicates,
    report_missing,
    setup_logger,
    validate_dataframe,
)

# Module logger
logger = setup_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DEFENSIVE LOW-LEVEL READERS
# ══════════════════════════════════════════════════════════════════════════════

def _read_csv_safe(
    path: Path,
    description: str,
    dtype: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file with structured error handling.

    Parameters
    ----------
    path : Path
        Absolute path to the CSV file.
    description : str
        Human-readable label used in log/error messages.
    dtype : dict, optional
        Column dtype mapping passed to ``pd.read_csv``.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the resulting DataFrame is completely empty (0 rows AND 0 columns).
    RuntimeError
        If ``pd.read_csv`` raises any other exception.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[{description}] File not found: {path}\n"
            f"Please place the file in the data/ directory."
        )

    file_size_kb = path.stat().st_size / 1024
    logger.info(
        f"Reading [{description}]  →  {path.name}  ({file_size_kb:.1f} KB)"
    )

    try:
        df = pd.read_csv(path, dtype=dtype, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"[{description}] Failed to read CSV '{path.name}': {exc}"
        ) from exc

    if df.empty and df.columns.empty:
        raise ValueError(
            f"[{description}] CSV '{path.name}' is completely empty "
            f"(0 rows, 0 columns)."
        )

    logger.info(
        f"  ✓ Loaded  {len(df):>6,} rows × {len(df.columns)} cols"
    )
    return df


def _read_excel_safe(
    path: Path,
    description: str,
    sheet_name: int | str = 0,
    dtype: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read an Excel file with structured error handling.

    Parameters
    ----------
    path : Path
        Absolute path to the Excel file (.xlsx).
    description : str
        Human-readable label for log/error messages.
    sheet_name : int or str, optional
        Sheet index or name.  Default 0 (first sheet).
    dtype : dict, optional
        Column dtype mapping passed to ``pd.read_excel``.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_excel``.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError, ValueError, RuntimeError
        Same semantics as ``_read_csv_safe``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[{description}] File not found: {path}\n"
            f"Please place the file in the data/ directory."
        )

    file_size_kb = path.stat().st_size / 1024
    logger.info(
        f"Reading [{description}]  →  {path.name}  ({file_size_kb:.1f} KB)"
    )

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=dtype, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"[{description}] Failed to read Excel '{path.name}': {exc}"
        ) from exc

    if df.empty and df.columns.empty:
        raise ValueError(
            f"[{description}] Excel '{path.name}' sheet '{sheet_name}' is "
            f"completely empty."
        )

    logger.info(
        f"  ✓ Loaded  {len(df):>6,} rows × {len(df.columns)} cols"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FILE 1: PASSIVE NAV DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_passive_nav(path: Path = PASSIVE_NAV_FILE) -> pd.DataFrame:
    """
    Load and validate the passive fund NAV dataset (FILE 1).

    This file contains 33 passive funds (Nifty 50 ETFs, BSE Sensex ETFs, and
    direct index funds).  It does NOT contain a FUND_TYPE column — that label
    ("Passive") is assigned in ``preprocessing.py``.

    Expected schema
    ---------------
    +---------------------------+----------+-------------------------------------+
    | Column                    | Dtype    | Notes                               |
    +===========================+==========+=====================================+
    | Fund_Name                 | object   | Raw fund name; normalised later     |
    | Month                     | object   | Full English name ("July") — unused |
    | Date                      | object   | YYYY-MM-DD string → parsed later    |
    | NAV                       | float64  | Net Asset Value (INR)               |
    | Monthly_Return            | float64  | Decimal return; NaN for first obs   |
    | Monthly_Return_Percent    | float64  | Return × 100; NaN for first obs     |
    +---------------------------+----------+-------------------------------------+

    Parameters
    ----------
    path : Path, optional
        File path override.  Defaults to ``config.PASSIVE_NAV_FILE``.

    Returns
    -------
    pd.DataFrame
        Raw passive NAV DataFrame.  Shape: (4396, 6).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``path``.
    ValueError
        If required columns are missing or the file is empty.
    """
    DESCRIPTION = "Passive NAV (FILE 1)"

    # ── Read ──────────────────────────────────────────────────────────────────
    df = _read_csv_safe(
        path,
        description=DESCRIPTION,
        dtype={
            COL_FUND_NAME_RAW: str,
            COL_MONTH_RAW:     str,
            COL_DATE_RAW:      str,
        },
    )

    # ── Schema validation ─────────────────────────────────────────────────────
    required_cols = [
        COL_FUND_NAME_RAW,
        COL_MONTH_RAW,
        COL_DATE_RAW,
        COL_NAV,
        COL_RETURN_RAW,
        COL_RETURN_PCT_RAW,
    ]
    validate_dataframe(df, required_cols, df_name=DESCRIPTION, min_rows=100)

    # ── Detect unexpected extra columns (non-fatal) ───────────────────────────
    extra_cols = [c for c in df.columns if c not in required_cols]
    if extra_cols:
        logger.warning(
            f"[{DESCRIPTION}] Unexpected extra columns (will be retained): "
            f"{extra_cols}"
        )

    # ── Coerce numeric columns ────────────────────────────────────────────────
    numeric_cols = [COL_NAV, COL_RETURN_RAW, COL_RETURN_PCT_RAW]
    df = coerce_numeric_columns(df, numeric_cols, df_name=DESCRIPTION)

    # ── Validate NAV is positive (data sanity check) ──────────────────────────
    neg_nav = (df[COL_NAV] <= 0).sum()
    if neg_nav > 0:
        logger.warning(
            f"[{DESCRIPTION}] {neg_nav} rows have NAV ≤ 0 — potential data error."
        )

    # ── Missing value report ──────────────────────────────────────────────────
    report_missing(df, df_name=DESCRIPTION)

    # ── Duplicate check ───────────────────────────────────────────────────────
    report_duplicates(
        df,
        subset=[COL_FUND_NAME_RAW, COL_DATE_RAW],
        df_name=DESCRIPTION,
    )

    # ── Fund universe summary ─────────────────────────────────────────────────
    n_funds = df[COL_FUND_NAME_RAW].nunique()
    date_min = df[COL_DATE_RAW].min()
    date_max = df[COL_DATE_RAW].max()
    logger.info(
        f"  [{DESCRIPTION}] {n_funds} unique funds  |  "
        f"Date range: {date_min} → {date_max}  |  "
        f"Expected type: {FUND_TYPE_PASSIVE}"
    )

    # ── Month column format note ──────────────────────────────────────────────
    sample_months = df[COL_MONTH_RAW].dropna().unique()[:3].tolist()
    logger.info(
        f"  [{DESCRIPTION}] Month column format sample: {sample_months} "
        f"(full English names — pipeline will use Date column exclusively)"
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FILE 2: FACTOR DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_factor_data(path: Path = FACTOR_DATA_FILE) -> pd.DataFrame:
    """
    Load and validate the Carhart factor dataset (FILE 2).

    .. warning::
        The provided ``factor_data.csv`` contains only **3 rows** covering
        **1993-10 to 1993-12**.  This has **zero temporal overlap** with the
        NAV data (2006–2026).  This loader flags this condition as a WARNING
        and returns the raw frame anyway — ``factor_merge.py`` decides whether
        to use it or build synthetic factors.

    Expected schema
    ---------------
    +------+----------+------------------------------------------+
    | Col  | Dtype    | Notes                                    |
    +======+==========+==========================================+
    | Date | object   | YYYY-MM string (no day component)        |
    | SMB  | float64  | Small Minus Big factor                   |
    | HML  | float64  | High Minus Low factor                    |
    | WML  | float64  | Winners Minus Losers (Momentum)          |
    | MF   | float64  | Market excess return (Rm − Rf)           |
    | RF   | float64  | Monthly risk-free rate (decimal percent) |
    +------+----------+------------------------------------------+

    Parameters
    ----------
    path : Path, optional
        File path override.  Defaults to ``config.FACTOR_DATA_FILE``.

    Returns
    -------
    pd.DataFrame
        Raw factor DataFrame.  Shape: (3, 6) with current file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``path``.
    ValueError
        If required columns are missing.
    """
    DESCRIPTION = "Factor Data (FILE 2)"

    df = _read_csv_safe(
        path,
        description=DESCRIPTION,
        dtype={COL_FACTOR_DATE_RAW: str},
    )

    # ── Schema validation ─────────────────────────────────────────────────────
    required_cols = [
        COL_FACTOR_DATE_RAW,
        COL_SMB_RAW,
        COL_HML_RAW,
        COL_WML_RAW,
        COL_MF_RAW,
        COL_RF_RAW,
    ]
    # min_rows=1 because we know the file is tiny — we report its usability
    # separately rather than crashing here
    validate_dataframe(df, required_cols, df_name=DESCRIPTION, min_rows=1)

    # ── Coerce numeric columns ────────────────────────────────────────────────
    numeric_cols = [COL_SMB_RAW, COL_HML_RAW, COL_WML_RAW, COL_MF_RAW, COL_RF_RAW]
    df = coerce_numeric_columns(df, numeric_cols, df_name=DESCRIPTION)

    # ── Missing value report ──────────────────────────────────────────────────
    report_missing(df, df_name=DESCRIPTION)

    # ── CRITICAL: coverage check against analysis window ─────────────────────
    # Parse dates as YYYY-MM → Period[M] for comparison
    try:
        factor_periods = pd.PeriodIndex(df[COL_FACTOR_DATE_RAW], freq="M")
        analysis_start = pd.Period(ANALYSIS_START[:7], freq="M")
        analysis_end   = pd.Period(ANALYSIS_END[:7],   freq="M")

        overlap_mask = (factor_periods >= analysis_start) & (
            factor_periods <= analysis_end
        )
        n_overlap = overlap_mask.sum()
        total_analysis_months = (
            (analysis_end.year - analysis_start.year) * 12
            + (analysis_end.month - analysis_start.month) + 1
        )
        coverage_pct = n_overlap / total_analysis_months * 100

        logger.warning(
            f"  [FACTOR COVERAGE] Factor data rows: {len(df)}  |  "
            f"Factor date range: {df[COL_FACTOR_DATE_RAW].min()} → "
            f"{df[COL_FACTOR_DATE_RAW].max()}"
        )
        logger.warning(
            f"  [FACTOR COVERAGE] Analysis window: {ANALYSIS_START[:7]} → "
            f"{ANALYSIS_END[:7]}  ({total_analysis_months} months)"
        )
        logger.warning(
            f"  [FACTOR COVERAGE] Overlap with analysis window: "
            f"{n_overlap} months ({coverage_pct:.1f}%)  ← "
            f"{'SUFFICIENT' if coverage_pct >= 90 else 'INSUFFICIENT — synthetic factors will be used'}"
        )

    except Exception as exc:
        logger.warning(
            f"  [FACTOR COVERAGE] Could not evaluate coverage: {exc}"
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FILE 3: EXPENSE RATIO LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_expense_ratios(path: Path = EXPENSE_RATIO_FILE) -> pd.DataFrame:
    """
    Load and validate the fund expense ratio dataset (FILE 3).

    Expected schema
    ---------------
    +-------------------+----------+------------------------------------------+
    | Column            | Dtype    | Notes                                    |
    +===================+==========+==========================================+
    | Funds             | object   | Fund name (requires normalisation)       |
    | Expense Ratio (%) | float64  | Annual expense ratio as a percentage     |
    +-------------------+----------+------------------------------------------+

    Parameters
    ----------
    path : Path, optional
        File path override.  Defaults to ``config.EXPENSE_RATIO_FILE``.

    Returns
    -------
    pd.DataFrame
        Raw expense ratio DataFrame.  Shape: (68, 2).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``path``.
    ValueError
        If required columns are missing or expense ratios contain impossible
        values (< 0%).
    """
    DESCRIPTION = "Expense Ratios (FILE 3)"

    df = _read_excel_safe(
        path,
        description=DESCRIPTION,
        sheet_name=0,
    )

    # ── Schema validation ─────────────────────────────────────────────────────
    required_cols = [COL_ER_FUND_RAW, COL_ER_VALUE_RAW]
    validate_dataframe(df, required_cols, df_name=DESCRIPTION, min_rows=1)

    # ── Coerce expense ratio to numeric ──────────────────────────────────────
    df = coerce_numeric_columns(df, [COL_ER_VALUE_RAW], df_name=DESCRIPTION)

    # ── Sanity: no negative expense ratios ───────────────────────────────────
    neg_er = (df[COL_ER_VALUE_RAW] < 0).sum()
    if neg_er > 0:
        logger.warning(
            f"[{DESCRIPTION}] {neg_er} funds have a negative expense ratio — "
            f"these will be flagged during preprocessing."
        )

    # ── Missing value report ──────────────────────────────────────────────────
    report_missing(df, df_name=DESCRIPTION)

    # ── Duplicate fund names check ────────────────────────────────────────────
    n_dup_names = df[COL_ER_FUND_RAW].duplicated().sum()
    if n_dup_names > 0:
        dup_names = df[df[COL_ER_FUND_RAW].duplicated(keep=False)][
            COL_ER_FUND_RAW
        ].tolist()
        logger.warning(
            f"[{DESCRIPTION}] {n_dup_names} duplicate fund names detected: "
            f"{dup_names}"
        )
    else:
        logger.debug(f"[{DESCRIPTION}] No duplicate fund names.")

    # ── Distribution summary ──────────────────────────────────────────────────
    er_series = df[COL_ER_VALUE_RAW].dropna()
    logger.info(
        f"  [{DESCRIPTION}] {len(df)} funds  |  "
        f"ER range: {er_series.min():.2f}% – {er_series.max():.2f}%  |  "
        f"Median: {er_series.median():.2f}%  |  Mean: {er_series.mean():.2f}%"
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FILE 4: ACTIVE NAV DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_active_nav(path: Path = ACTIVE_NAV_FILE) -> pd.DataFrame:
    """
    Load and validate the active fund NAV dataset (FILE 4).

    This file contains 33 active large-cap direct-plan mutual funds.  Like
    FILE 1, it has no FUND_TYPE column — labelling ("Active") is done in
    ``preprocessing.py``.

    Key schema difference from FILE 1:
        FILE 1 Month column: "July"     (full English month name)
        FILE 4 Month column: "2013-01"  (YYYY-MM format)
    The pipeline ignores Month in both files and works exclusively with Date.

    Expected schema
    ---------------
    +---------------------------+----------+-------------------------------------+
    | Column                    | Dtype    | Notes                               |
    +===========================+==========+=====================================+
    | Fund_Name                 | object   | Raw active fund name                |
    | Month                     | object   | YYYY-MM string — ignored by pipeline|
    | Date                      | object   | YYYY-MM-DD string → parsed later    |
    | NAV                       | float64  | Net Asset Value (INR)               |
    | Monthly_Return            | float64  | Decimal return; NaN for first obs   |
    | Monthly_Return_Percent    | float64  | Return × 100; NaN for first obs     |
    +---------------------------+----------+-------------------------------------+

    Parameters
    ----------
    path : Path, optional
        File path override.  Defaults to ``config.ACTIVE_NAV_FILE``.

    Returns
    -------
    pd.DataFrame
        Raw active NAV DataFrame.  Shape: (4210, 6).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``path``.
    ValueError
        If required columns are missing or the file is empty.
    """
    DESCRIPTION = "Active NAV (FILE 4)"

    df = _read_csv_safe(
        path,
        description=DESCRIPTION,
        dtype={
            COL_FUND_NAME_RAW: str,
            COL_MONTH_RAW:     str,
            COL_DATE_RAW:      str,
        },
    )

    # ── Schema validation ─────────────────────────────────────────────────────
    required_cols = [
        COL_FUND_NAME_RAW,
        COL_MONTH_RAW,
        COL_DATE_RAW,
        COL_NAV,
        COL_RETURN_RAW,
        COL_RETURN_PCT_RAW,
    ]
    validate_dataframe(df, required_cols, df_name=DESCRIPTION, min_rows=100)

    # ── Detect unexpected extra columns ───────────────────────────────────────
    extra_cols = [c for c in df.columns if c not in required_cols]
    if extra_cols:
        logger.warning(
            f"[{DESCRIPTION}] Unexpected extra columns (retained): {extra_cols}"
        )

    # ── Coerce numeric columns ────────────────────────────────────────────────
    numeric_cols = [COL_NAV, COL_RETURN_RAW, COL_RETURN_PCT_RAW]
    df = coerce_numeric_columns(df, numeric_cols, df_name=DESCRIPTION)

    # ── Validate NAV positivity ───────────────────────────────────────────────
    neg_nav = (df[COL_NAV] <= 0).sum()
    if neg_nav > 0:
        logger.warning(
            f"[{DESCRIPTION}] {neg_nav} rows have NAV ≤ 0 — potential data error."
        )

    # ── Missing value report ──────────────────────────────────────────────────
    report_missing(df, df_name=DESCRIPTION)

    # ── Duplicate check ───────────────────────────────────────────────────────
    report_duplicates(
        df,
        subset=[COL_FUND_NAME_RAW, COL_DATE_RAW],
        df_name=DESCRIPTION,
    )

    # ── Fund universe summary ─────────────────────────────────────────────────
    n_funds = df[COL_FUND_NAME_RAW].nunique()
    date_min = df[COL_DATE_RAW].min()
    date_max = df[COL_DATE_RAW].max()
    logger.info(
        f"  [{DESCRIPTION}] {n_funds} unique funds  |  "
        f"Date range: {date_min} → {date_max}  |  "
        f"Expected type: {FUND_TYPE_ACTIVE}"
    )

    # ── Month column format note ──────────────────────────────────────────────
    sample_months = df[COL_MONTH_RAW].dropna().unique()[:3].tolist()
    logger.info(
        f"  [{DESCRIPTION}] Month column format sample: {sample_months} "
        f"(YYYY-MM format — different from FILE 1, pipeline uses Date column)"
    )

    # ── Flag funds with very short history ───────────────────────────────────
    obs_per_fund = df.groupby(COL_FUND_NAME_RAW).size()
    short_history = obs_per_fund[obs_per_fund < 36]
    if not short_history.empty:
        logger.warning(
            f"  [{DESCRIPTION}] {len(short_history)} funds have < 36 observations "
            f"(will be excluded from regression):"
        )
        for fname, n_obs in short_history.items():
            logger.warning(f"      {fname:55s}  →  {n_obs} months")

    # ── Flag dates beyond 2026-04-30 (2 funds with 2026-05-19 entry) ─────────
    late_dates = df[df[COL_DATE_RAW] > ANALYSIS_END]
    if not late_dates.empty:
        n_late = len(late_dates)
        unique_late_funds = late_dates[COL_FUND_NAME_RAW].unique().tolist()
        logger.warning(
            f"  [{DESCRIPTION}] {n_late} rows have Date > {ANALYSIS_END} "
            f"(beyond analysis end).  Affected funds: {unique_late_funds}. "
            f"These will be trimmed during preprocessing."
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ORCHESTRATOR: load_all_data
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data(
    passive_path: Path = PASSIVE_NAV_FILE,
    active_path:  Path = ACTIVE_NAV_FILE,
    factor_path:  Path = FACTOR_DATA_FILE,
    er_path:      Path = EXPENSE_RATIO_FILE,
) -> dict[str, pd.DataFrame]:
    """
    Load all four input files and return them as a named dictionary.

    This is the single function ``main.py`` calls to ingest all raw data.
    It provides a complete loading report (per-file row counts, fund counts,
    date ranges, and any warnings) before passing control to
    ``preprocessing.py``.

    Parameters
    ----------
    passive_path : Path, optional
        Override path for the passive NAV CSV.
    active_path : Path, optional
        Override path for the active NAV CSV.
    factor_path : Path, optional
        Override path for the factor data CSV.
    er_path : Path, optional
        Override path for the expense ratio XLSX.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys:
            ``"passive_nav"``    → FILE 1 raw DataFrame
            ``"active_nav"``     → FILE 4 raw DataFrame
            ``"factor_data"``    → FILE 2 raw DataFrame
            ``"expense_ratios"`` → FILE 3 raw DataFrame

    Raises
    ------
    FileNotFoundError, ValueError, RuntimeError
        Propagated from individual loaders on fatal errors.

    Notes
    -----
    Loading order:
        passive_nav → active_nav → factor_data → expense_ratios

    This order is deliberately sequential (not parallel) because each loader
    writes to the shared log file and any failure should halt the pipeline
    with a clear message before any downstream processing begins.
    """
    logger.info("=" * 68)
    logger.info("  DATA LOADING PHASE — loading all 4 input files")
    logger.info("=" * 68)

    results: dict[str, pd.DataFrame] = {}

    # ── Load in sequence, collect errors ─────────────────────────────────────
    load_tasks = [
        ("passive_nav",    load_passive_nav,    passive_path),
        ("active_nav",     load_active_nav,     active_path),
        ("factor_data",    load_factor_data,    factor_path),
        ("expense_ratios", load_expense_ratios, er_path),
    ]

    for key, loader_fn, file_path in load_tasks:
        logger.info(f"\n── Loading: {key} ──")
        try:
            results[key] = loader_fn(file_path)
        except FileNotFoundError as exc:
            logger.error(f"FATAL: {exc}")
            raise
        except (ValueError, RuntimeError) as exc:
            logger.error(f"FATAL loading '{key}': {exc}")
            raise

    # ── Cross-file sanity check: no fund should appear in both NAV files ──────
    passive_funds = set(
        results["passive_nav"][COL_FUND_NAME_RAW].unique()
    )
    active_funds = set(
        results["active_nav"][COL_FUND_NAME_RAW].unique()
    )
    overlap_funds = passive_funds & active_funds
    if overlap_funds:
        logger.warning(
            f"Cross-file check: {len(overlap_funds)} fund name(s) appear in "
            f"BOTH passive and active files — verify FUND_TYPE assignment:\n"
            f"  {sorted(overlap_funds)}"
        )
    else:
        logger.info(
            f"Cross-file check: ✓ No fund appears in both passive and active files "
            f"({len(passive_funds)} passive, {len(active_funds)} active, 0 overlap)."
        )

    # ── Loading summary table ─────────────────────────────────────────────────
    logger.info("\n" + "─" * 68)
    logger.info("  LOADING SUMMARY")
    logger.info("─" * 68)
    logger.info(
        f"  {'Dataset':<20} {'Rows':>7}  {'Cols':>5}  "
        f"{'Funds':>6}  Notes"
    )
    logger.info("  " + "-" * 64)

    summaries = [
        (
            "passive_nav",
            len(results["passive_nav"]),
            len(results["passive_nav"].columns),
            results["passive_nav"][COL_FUND_NAME_RAW].nunique(),
            "FILE 1 — 33 passive funds",
        ),
        (
            "active_nav",
            len(results["active_nav"]),
            len(results["active_nav"].columns),
            results["active_nav"][COL_FUND_NAME_RAW].nunique(),
            "FILE 4 — 33 active funds",
        ),
        (
            "factor_data",
            len(results["factor_data"]),
            len(results["factor_data"].columns),
            0,
            "FILE 2 — ⚠ only 3 rows (1993)",
        ),
        (
            "expense_ratios",
            len(results["expense_ratios"]),
            len(results["expense_ratios"].columns),
            results["expense_ratios"][COL_ER_FUND_RAW].nunique(),
            "FILE 3 — 68 funds (active + passive)",
        ),
    ]

    for name, rows, cols, funds, note in summaries:
        funds_str = str(funds) if funds > 0 else "N/A"
        logger.info(
            f"  {name:<20} {rows:>7,}  {cols:>5}  {funds_str:>6}  {note}"
        )

    logger.info("─" * 68)
    logger.info("  Data loading complete.  Proceed to preprocessing.py.")
    logger.info("=" * 68 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SELF-TEST (python src/data_loader.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _log = setup_logger("data_loader_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  DATA LOADER SELF-TEST")
    _log.info("═" * 68)

    try:
        data = load_all_data()

        _log.info("\n── Spot-check raw frames ──")

        # FILE 1 spot checks
        pn = data["passive_nav"]
        assert pn.shape[1] == 6, f"Expected 6 cols in passive_nav, got {pn.shape[1]}"
        assert pn[COL_FUND_NAME_RAW].nunique() == 33, "Expected 33 passive funds"
        assert COL_NAV in pn.columns, "NAV column missing"
        _log.info(
            f"  ✅ passive_nav  shape={pn.shape}  "
            f"funds={pn[COL_FUND_NAME_RAW].nunique()}"
        )

        # FILE 4 spot checks
        an = data["active_nav"]
        assert an.shape[1] == 6, f"Expected 6 cols in active_nav, got {an.shape[1]}"
        assert an[COL_FUND_NAME_RAW].nunique() == 33, "Expected 33 active funds"
        _log.info(
            f"  ✅ active_nav   shape={an.shape}  "
            f"funds={an[COL_FUND_NAME_RAW].nunique()}"
        )

        # FILE 2 spot checks
        fd = data["factor_data"]
        assert COL_MF_RAW in fd.columns, "MF column missing from factor data"
        assert COL_RF_RAW in fd.columns, "RF column missing from factor data"
        _log.info(
            f"  ✅ factor_data  shape={fd.shape}  "
            f"(⚠ only {len(fd)} rows — synthetic factors will be used)"
        )

        # FILE 3 spot checks
        er = data["expense_ratios"]
        assert COL_ER_VALUE_RAW in er.columns, "Expense Ratio column missing"
        assert er[COL_ER_VALUE_RAW].isna().sum() == 0, "Unexpected NaN in expense ratios"
        _log.info(
            f"  ✅ expense_ratios  shape={er.shape}  "
            f"ER range: {er[COL_ER_VALUE_RAW].min():.2f}% – "
            f"{er[COL_ER_VALUE_RAW].max():.2f}%"
        )

        # Combined fund count
        total_funds = (
            pn[COL_FUND_NAME_RAW].nunique() + an[COL_FUND_NAME_RAW].nunique()
        )
        _log.info(f"\n  Combined fund universe: {total_funds} funds (33 passive + 33 active)")
        _log.info(
            f"  Passive date range: "
            f"{pn[COL_DATE_RAW].min()} → {pn[COL_DATE_RAW].max()}"
        )
        _log.info(
            f"  Active  date range: "
            f"{an[COL_DATE_RAW].min()} → {an[COL_DATE_RAW].max()}"
        )

        _log.info("\n✅  All data_loader self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Self-test FAILED: {exc}")
        raise

    _log.info("═" * 68 + "\n")
