"""
config.py
=========
Central configuration module for the Carhart Four-Factor Fund Analysis Pipeline.

All hard-coded constants, file paths, column names, analysis parameters, and
output settings live here.  Every other module imports exclusively from this
file — never from one another for configuration values.  This means you can
change an analysis window, a column name, or an output path in exactly one
place and the change propagates throughout the entire pipeline automatically.

Project Context
---------------
This project replicates and extends the analysis in:
    "The Historical Record on Active vs Passive Mutual Fund Performance"
    David Nanigian (2019)

The central hypothesis tested is:
    Passive funds may outperform active funds primarily because they charge
    lower expense ratios, rather than because active management lacks skill.

Data Reality Notes (discovered during inspection — DO NOT REMOVE)
-----------------------------------------------------------------
* monthly_nav_with_returns.csv  → 33 PASSIVE funds (ETFs / index funds)
                                   NO FUND_TYPE column — assigned here
* all_active_large_cap_funds_with_returns.csv → 33 ACTIVE large-cap funds
* factor_data.csv               → ONLY 3 rows (1993-10 to 1993-12)
                                   NO overlap with NAV data (2006–2026)
                                   → Synthetic factors are built in factor_merge.py
* fund_expense_ratios.xlsx      → 68 funds; 65/68 match after normalization

Usage
-----
    from src.config import (
        DATA_DIR, PASSIVE_NAV_FILE, ACTIVE_NAV_FILE,
        ANALYSIS_START, ANALYSIS_END, MIN_OBS, ...
    )
"""

from __future__ import annotations

import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ROOT & DIRECTORY PATHS
# ══════════════════════════════════════════════════════════════════════════════

# Resolve the project root relative to this file's location so the project
# can be cloned anywhere without breaking paths.
#   config.py lives at:  <project_root>/src/config.py
#   So project root  is:  <project_root>/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Input data directory
DATA_DIR: Path = PROJECT_ROOT / "data"

# Output subdirectories
OUTPUTS_DIR: Path          = PROJECT_ROOT / "outputs"
CLEANED_DATA_DIR: Path     = OUTPUTS_DIR / "cleaned_data"
REGRESSION_RESULTS_DIR: Path = OUTPUTS_DIR / "regression_results"
PLOTS_DIR: Path            = OUTPUTS_DIR / "plots"
FIGURES_DIR: Path          = OUTPUTS_DIR / "figures"   # visualization.py output
REPORTS_DIR: Path          = OUTPUTS_DIR / "reports"

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — INPUT FILE PATHS
# ──────────────────────────────────────────────────────────────────────────────

# FILE 1: 33 passive funds (Nifty 50 ETFs + BSE Sensex ETFs + index funds)
#         Columns: Fund_Name, Month (full name), Date (YYYY-MM-DD), NAV,
#                  Monthly_Return, Monthly_Return_Percent
#         Date range: 2006-04-28 → 2026-04-30
PASSIVE_NAV_FILE: Path = DATA_DIR / "monthly_nav_with_returns.csv"

# FILE 2: Fama-French-style Indian market factor data
#         WARNING: Only covers 1993-10 to 1993-12 (3 rows).
#                  No temporal overlap with NAV data.
#                  Synthetic factors will be constructed instead.
#         Columns: Date (YYYY-MM), SMB, HML, WML, MF, RF
FACTOR_DATA_FILE: Path = DATA_DIR / "factor_data.csv"

# FILE 3: Annual expense ratios for all 68 funds (active + passive)
#         Columns: Funds, Expense Ratio (%)
EXPENSE_RATIO_FILE: Path = DATA_DIR / "fund_expense_ratios.xlsx"

# FILE 4: 33 active large-cap direct-plan mutual funds
#         Columns: Fund_Name, Month (YYYY-MM), Date (YYYY-MM-DD), NAV,
#                  Monthly_Return, Monthly_Return_Percent
#         Date range: 2013-01-31 → 2026-04-30
ACTIVE_NAV_FILE: Path = DATA_DIR / "all_active_large_cap_funds_with_returns.csv"

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — OUTPUT FILE PATHS
# ──────────────────────────────────────────────────────────────────────────────

# Cleaned / merged intermediate datasets
MASTER_DATA_FILE: Path          = CLEANED_DATA_DIR / "master_fund_data.csv"
PASSIVE_CLEAN_FILE: Path        = CLEANED_DATA_DIR / "passive_funds_clean.csv"
ACTIVE_CLEAN_FILE: Path         = CLEANED_DATA_DIR / "active_funds_clean.csv"
FACTOR_SYNTHETIC_FILE: Path     = CLEANED_DATA_DIR / "synthetic_factors.csv"

# Regression & metrics results
REGRESSION_SUMMARY_FILE: Path   = REGRESSION_RESULTS_DIR / "carhart_regression_summary.csv"
PERFORMANCE_METRICS_FILE: Path  = REGRESSION_RESULTS_DIR / "performance_metrics.csv"
STATISTICAL_TESTS_FILE: Path    = REGRESSION_RESULTS_DIR / "statistical_tests.csv"

# Report
SUMMARY_REPORT_FILE: Path       = REPORTS_DIR / "summary_report.txt"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COLUMN NAME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# Centralise every column name so that downstream modules never use raw
# strings and a column rename only requires editing this section.

# ── Raw input columns ─────────────────────────────────────────────────────────
COL_FUND_NAME_RAW: str       = "Fund_Name"       # original column in both NAV files
COL_MONTH_RAW: str           = "Month"           # "July" in passive; "2013-01" in active
COL_DATE_RAW: str            = "Date"            # YYYY-MM-DD string in both NAV files
COL_NAV: str                 = "NAV"
COL_RETURN_RAW: str          = "Monthly_Return"          # decimal (0.077)
COL_RETURN_PCT_RAW: str      = "Monthly_Return_Percent"  # percentage (7.7)

COL_ER_FUND_RAW: str         = "Funds"                   # expense ratio file
COL_ER_VALUE_RAW: str        = "Expense Ratio (%)"       # expense ratio file

COL_FACTOR_DATE_RAW: str     = "Date"    # YYYY-MM string in factor file
COL_SMB_RAW: str             = "SMB"
COL_HML_RAW: str             = "HML"
COL_WML_RAW: str             = "WML"
COL_MF_RAW: str              = "MF"     # market excess return in factor file
COL_RF_RAW: str              = "RF"     # risk-free rate in factor file

# ── Standardised / pipeline columns (used after preprocessing) ───────────────
COL_FUND_NAME: str           = "Fund_Name"         # normalised display name
COL_FUND_NAME_NORM: str      = "Fund_Name_Norm"    # lowercase normalised key for merges
COL_FUND_TYPE: str           = "Fund_Type"         # "Active" | "Passive"
COL_DATE: str                = "Date"              # pd.Timestamp after parsing
COL_YEAR_MONTH: str          = "Year_Month"        # Period[M] for factor merge
COL_RETURN: str              = "Monthly_Return"    # decimal monthly return
COL_EXCESS_RETURN: str       = "Excess_Return"     # Return - RF

COL_MF: str                  = "MF"               # market factor (synthetic or real)
COL_SMB: str                 = "SMB"
COL_HML: str                 = "HML"
COL_WML: str                 = "WML"
COL_RF: str                  = "RF"               # risk-free rate (monthly, decimal)
COL_ER: str                  = "Expense_Ratio"    # annual expense ratio (%)

# ── Regression output columns ─────────────────────────────────────────────────
COL_ALPHA: str               = "Alpha"
COL_T_ALPHA: str             = "T_Alpha"
COL_P_ALPHA: str             = "P_Alpha"
COL_BETA_MF: str             = "Beta_MF"
COL_BETA_SMB: str            = "Beta_SMB"
COL_BETA_HML: str            = "Beta_HML"
COL_BETA_WML: str            = "Beta_WML"
COL_R2: str                  = "R_Squared"
COL_ADJ_R2: str              = "Adj_R_Squared"
COL_N_OBS: str               = "N_Obs"
COL_REG_STATUS: str          = "Regression_Status"  # "OK" | "SKIPPED" | "ERROR"

# ── Return calculation columns (added by return_calculations.py) ─────────────
COL_LOG_RETURN: str          = "Log_Return"          # ln(NAV_t / NAV_{t-1})
COL_CUMULATIVE_RETURN: str   = "Cumulative_Return"   # wealth index starting at 1.0
COL_ROLLING_3M: str          = "Rolling_Return_3M"   # 3-month trailing mean return
COL_ROLLING_6M: str          = "Rolling_Return_6M"   # 6-month trailing mean return
COL_ROLLING_12M: str         = "Rolling_Return_12M"  # 12-month trailing mean return
COL_RF_SOURCE: str           = "RF_Source"           # "synthetic" | "provided"

# ── Performance metric columns ────────────────────────────────────────────────
COL_CAGR: str                = "CAGR"
COL_ANN_RETURN: str          = "Annualized_Return"
COL_ANN_VOL: str             = "Annualized_Volatility"
COL_SHARPE: str              = "Sharpe_Ratio"
COL_MAX_DRAWDOWN: str        = "Max_Drawdown"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FUND TYPE LABELS
# ══════════════════════════════════════════════════════════════════════════════

FUND_TYPE_ACTIVE: str   = "Active"
FUND_TYPE_PASSIVE: str  = "Passive"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ANALYSIS PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Common analysis window for active vs passive comparison.
# Rationale: Active fund data begins 2013-01; passive data begins 2006-04.
#            Restricting to 2013-01 → 2026-04 gives a fair, balanced comparison
#            (~160 months).  Passive funds with earlier history are retained for
#            standalone fund-level analysis but flagged in the master dataset.
ANALYSIS_START: str     = "2013-01-01"   # inclusive lower bound (YYYY-MM-DD)
ANALYSIS_END: str       = "2026-04-30"   # inclusive upper bound (YYYY-MM-DD)

# Minimum number of monthly observations required for a fund to be included in
# the Carhart regression.  Funds with fewer observations are logged and skipped.
# 36 months (3 years) is the standard academic minimum for factor regressions.
MIN_OBS: int            = 36

# Annualisation factor for return and volatility calculations
# (number of months in a year)
MONTHS_PER_YEAR: int    = 12

# Significance level for all hypothesis tests (two-tailed)
ALPHA_SIGNIFICANCE: float = 0.05

# Winsorisation bounds for monthly returns — values outside this range are
# capped and logged as outliers. (Monthly return of -50% or +100% is extreme.)
RETURN_LOWER_BOUND: float = -0.50    # -50% monthly
RETURN_UPPER_BOUND: float =  1.00    # +100% monthly

# Minimum expense ratio sanity check (anything below this is flagged)
MIN_EXPENSE_RATIO: float  = 0.01    # 0.01%
MAX_EXPENSE_RATIO: float  = 5.00    # 5.00%

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SYNTHETIC FACTOR PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
# These are used by factor_merge.py when the provided factor_data.csv does not
# cover the analysis window (which is the case with the current 3-row file).

# Annual RBI repo rate proxy for the risk-free rate.
# Approximate average over the 2013–2026 period in India.
# Converted to monthly: RF_MONTHLY = RF_ANNUAL / MONTHS_PER_YEAR
SYNTHETIC_RF_ANNUAL: float  = 0.06       # 6.0% per annum (decimal)
SYNTHETIC_RF_MONTHLY: float = SYNTHETIC_RF_ANNUAL / MONTHS_PER_YEAR  # ~0.5% per month

# When the provided factor file covers the full analysis window, it is used
# directly.  When it does not (current situation), the pipeline falls back to
# synthetic factors.  Set this flag to False to force synthetic factors even if
# a complete factor file exists (useful for reproducibility testing).
USE_PROVIDED_FACTORS_IF_AVAILABLE: bool = True

# Factor coverage threshold:  if the provided factor file covers at least this
# fraction of the analysis window, it is used; otherwise synthetic fallback.
MIN_FACTOR_COVERAGE_FRACTION: float = 0.90   # 90 % of analysis months required

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FUND NAME NORMALISATION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
# Words stripped during name normalisation to improve cross-file merge accuracy.
# These are common suffixes/qualifiers that appear inconsistently across files.

NAME_STOP_WORDS: frozenset[str] = frozenset({
    "direct",
    "growth",
    "regular",
    "plan",
    "idcw",
    "option",
    "cumulative",
    "annual",
    "fund",      # note: keep "fund" in the set — it adds no discriminatory value
    "the",
    "a",
    "an",
})

# Punctuation characters to collapse to a space during normalisation
NAME_PUNCTUATION_CHARS: str = r"[^a-z0-9\s]"   # regex: keep only alphanumeric + space

# Minimum token length to retain after stop-word removal
NAME_MIN_TOKEN_LENGTH: int = 2

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — LOGGING SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL: str          = "INFO"           # "DEBUG" | "INFO" | "WARNING" | "ERROR"
LOG_FORMAT: str         = (
    "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
)
LOG_DATE_FORMAT: str    = "%Y-%m-%d %H:%M:%S"
LOG_FILE: Path          = REPORTS_DIR / "pipeline.log"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — VISUALISATION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Matplotlib / Seaborn style
PLOT_STYLE: str         = "seaborn-v0_8-whitegrid"   # fallback: "whitegrid"
PLOT_CONTEXT: str       = "paper"                     # "paper" | "talk" | "poster"
PLOT_FONT_FAMILY: str   = "DejaVu Sans"               # safe cross-platform font
PLOT_DPI: int           = 150          # resolution for saved figures
PLOT_FIGSIZE_WIDE: tuple[int, int]   = (14, 6)
PLOT_FIGSIZE_SQUARE: tuple[int, int] = (10, 8)
PLOT_FIGSIZE_TALL: tuple[int, int]   = (10, 12)

# Colour palette for Active vs Passive
COLOUR_ACTIVE: str      = "#E63946"    # vivid red
COLOUR_PASSIVE: str     = "#457B9D"    # steel blue
COLOUR_NEUTRAL: str     = "#6c757d"    # grey
COLOUR_HIGHLIGHT: str   = "#F4A261"    # amber (for annotations)

PALETTE_FUND_TYPE: dict[str, str] = {
    FUND_TYPE_ACTIVE: COLOUR_ACTIVE,
    FUND_TYPE_PASSIVE: COLOUR_PASSIVE,
}

# File format for saved plots
PLOT_FORMAT: str        = "png"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — EXPORT SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

CSV_INDEX: bool         = False     # never write the row index to CSVs
CSV_ENCODING: str       = "utf-8-sig"   # BOM prefix — opens cleanly in Excel
CSV_DATE_FORMAT: str    = "%Y-%m-%d"

# Number of decimal places for round() calls in exported tables
EXPORT_DECIMAL_PLACES: int = 6      # keep full precision in CSV
REPORT_DECIMAL_PLACES: int = 4      # rounded for human-readable reports

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — RUNTIME VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_paths() -> None:
    """
    Verify that all required input files exist and all output directories can
    be created.  Call this once at pipeline startup (inside main.py) to surface
    missing-file errors immediately rather than mid-pipeline.

    Raises
    ------
    FileNotFoundError
        If any required input data file is missing from DATA_DIR.
    """
    required_inputs: list[Path] = [
        PASSIVE_NAV_FILE,
        FACTOR_DATA_FILE,
        EXPENSE_RATIO_FILE,
        ACTIVE_NAV_FILE,
    ]

    output_dirs: list[Path] = [
        CLEANED_DATA_DIR,
        REGRESSION_RESULTS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
    ]

    # ── Check input files ────────────────────────────────────────────────────
    missing: list[Path] = [p for p in required_inputs if not p.exists()]
    if missing:
        missing_str = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"The following required input files are missing:\n  {missing_str}\n"
            f"Please place them in:  {DATA_DIR}"
        )

    # ── Ensure output directories exist (create if needed) ──────────────────
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_project_info() -> dict[str, str]:
    """
    Return a human-readable summary of the active configuration.  Useful for
    logging at pipeline startup so every run is fully reproducible.

    Returns
    -------
    dict[str, str]
        Key-value pairs describing the current configuration.
    """
    return {
        "Project Root":       str(PROJECT_ROOT),
        "Data Directory":     str(DATA_DIR),
        "Analysis Start":     ANALYSIS_START,
        "Analysis End":       ANALYSIS_END,
        "Min Observations":   str(MIN_OBS),
        "Significance Level": str(ALPHA_SIGNIFICANCE),
        "Synthetic RF (ann)": f"{SYNTHETIC_RF_ANNUAL:.1%}",
        "Synthetic RF (mo)":  f"{SYNTHETIC_RF_MONTHLY:.4%}",
        "Return Lower Bound": f"{RETURN_LOWER_BOUND:.0%}",
        "Return Upper Bound": f"{RETURN_UPPER_BOUND:.0%}",
        "Plot DPI":           str(PLOT_DPI),
        "Log Level":          LOG_LEVEL,
        "Passive NAV File":   PASSIVE_NAV_FILE.name,
        "Active NAV File":    ACTIVE_NAV_FILE.name,
        "Factor Data File":   FACTOR_DATA_FILE.name,
        "Expense Ratio File": EXPENSE_RATIO_FILE.name,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — SELF-TEST (run directly: python src/config.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  CONFIG SELF-TEST — Carhart Fund Analysis Pipeline")
    print("═" * 65)

    try:
        validate_paths()
        print("✅  All input files found.")
        print("✅  All output directories created / verified.\n")
    except FileNotFoundError as exc:
        print(f"❌  {exc}")

    info = get_project_info()
    max_key = max(len(k) for k in info)
    for k, v in info.items():
        print(f"  {k:<{max_key}}  →  {v}")

    print("\n" + "═" * 65)
    print("  Key column constants (sample)")
    print("═" * 65)
    sample_cols = {
        "COL_FUND_NAME":      COL_FUND_NAME,
        "COL_FUND_TYPE":      COL_FUND_TYPE,
        "COL_DATE":           COL_DATE,
        "COL_RETURN":         COL_RETURN,
        "COL_EXCESS_RETURN":  COL_EXCESS_RETURN,
        "COL_ALPHA":          COL_ALPHA,
        "COL_SHARPE":         COL_SHARPE,
        "COL_ER":             COL_ER,
    }
    max_k = max(len(k) for k in sample_cols)
    for k, v in sample_cols.items():
        print(f"  {k:<{max_k}}  =  '{v}'")

    print("\n✅  config.py loaded successfully.\n")
