"""
utils.py
========
Shared utility functions for the Carhart Four-Factor Fund Analysis Pipeline.

This module is the foundational toolkit imported by every other module in the
pipeline.  It contains no business logic — only pure, reusable helper functions
and the logging infrastructure.

Design principles
-----------------
* Every function is fully type-annotated.
* Every function has a NumPy-style docstring with Parameters, Returns,
  Raises, Notes, and Examples sections where relevant.
* All functions are exception-safe and emit structured log messages.
* No circular imports: utils.py imports ONLY from config.py and the
  Python standard library / third-party packages.

Module layout
-------------
  1. Logging setup
  2. Directory / path utilities
  3. Fund name normalisation
  4. DataFrame validation
  5. Numeric / type coercion helpers
  6. Date utilities
  7. Return / financial calculation helpers
  8. Outlier detection & winsorisation
  9. Export helpers
 10. Self-test (run directly: python src/utils.py)
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Local import — config is the only sibling allowed here
from src.config import (
    COL_DATE,
    COL_FUND_NAME_NORM,
    CSV_DATE_FORMAT,
    CSV_ENCODING,
    CSV_INDEX,
    LOG_DATE_FORMAT,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    NAME_MIN_TOKEN_LENGTH,
    NAME_PUNCTUATION_CHARS,
    NAME_STOP_WORDS,
    REPORTS_DIR,
    RETURN_LOWER_BOUND,
    RETURN_UPPER_BOUND,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(
    name: str,
    level: str = LOG_LEVEL,
    log_file: Optional[Path] = LOG_FILE,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create and configure a named logger with a console handler and an optional
    rotating file handler.

    This is the single entry-point for logger creation across the entire
    pipeline.  Call once per module at module level::

        logger = setup_logger(__name__)

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : str, optional
        Logging level string — "DEBUG", "INFO", "WARNING", "ERROR".
        Defaults to ``config.LOG_LEVEL``.
    log_file : Path or None, optional
        Absolute path to the log file.  Pass ``None`` to disable file logging.
        Defaults to ``config.LOG_FILE``.
    propagate : bool, optional
        Whether to propagate messages to the root logger.  Defaults to False
        to prevent duplicate console output when multiple modules log.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    * If the logger ``name`` already exists (i.e. the module is imported
      twice), its handlers are cleared before re-adding to avoid duplication.
    * The log directory is created automatically if it does not exist.
    """
    numeric_level: int = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = propagate

    # Clear stale handlers (avoids duplicate lines on re-import in notebooks)
    if logger.handlers:
        logger.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (optional) ───────────────────────────────────────────────
    if log_file is not None:
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# NOTE: Module-level logger is declared AFTER ensure_dir (Section 2) because
# setup_logger calls ensure_dir(log_file.parent) to create the log directory.
# The actual declaration appears after ensure_dir's definition below.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DIRECTORY / PATH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory path to create.

    Returns
    -------
    Path
        The same ``path`` object, so calls can be chained::

            df.to_csv(ensure_dir(OUTPUT_DIR) / "result.csv")

    Notes
    -----
    Uses ``exist_ok=True`` — calling this function on an existing directory
    is safe and produces no warning.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Module-level logger (declared here, after ensure_dir is defined) ──────────
logger: logging.Logger = setup_logger(__name__)


def check_file_exists(path: Path, description: str = "File") -> bool:
    """
    Check whether a file exists and log a clear message either way.

    Parameters
    ----------
    path : Path
        Path to the file being checked.
    description : str, optional
        Human-readable label used in log messages (e.g. "Passive NAV file").

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    if path.exists():
        size_kb = path.stat().st_size / 1024
        logger.debug(f"{description} found: {path.name}  ({size_kb:.1f} KB)")
        return True
    logger.warning(f"{description} NOT FOUND: {path}")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FUND NAME NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_fund_name(name: str) -> str:
    """
    Produce a canonical, lowercase, punctuation-free token string from a raw
    mutual fund name.  This normalised key is used for cross-file merges where
    the same fund appears under slightly different spellings.

    The algorithm:
      1. Decode any Unicode characters to closest ASCII equivalent.
      2. Lowercase the string.
      3. Replace all non-alphanumeric characters (hyphens, dots, brackets …)
         with a single space.
      4. Split into tokens and remove stop-words (defined in config.py).
      5. Remove tokens shorter than ``config.NAME_MIN_TOKEN_LENGTH``.
      6. Re-join with a single space and strip leading/trailing whitespace.

    Parameters
    ----------
    name : str
        Raw fund name as it appears in the source file.

    Returns
    -------
    str
        Normalised, merge-ready fund name key.

    Examples
    --------
    >>> normalize_fund_name("HDFC NIFTY 50 ETF - Growth Plan")
    'hdfc nifty 50 etf'

    >>> normalize_fund_name("Kotak Large Cap Fund - Growth - Direct")
    'kotak large cap'

    >>> normalize_fund_name("  ICICI Prudential BSE Sensex ETF  ")
    'icici prudential bse sensex etf'

    Notes
    -----
    * "fund" is included in ``NAME_STOP_WORDS`` because it appears in virtually
      every fund name and carries no discriminatory information for matching.
    * Numeric tokens (e.g. "50" in "Nifty 50") are intentionally retained —
      they are part of the index name and are discriminatory.
    """
    if not isinstance(name, str):
        name = str(name)

    # Step 1: Normalise Unicode → closest ASCII (handles accented chars etc.)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", errors="ignore").decode("ascii")

    # Step 2: Lowercase
    name = name.lower().strip()

    # Step 3: Replace all non-alphanumeric chars with a space
    name = re.sub(NAME_PUNCTUATION_CHARS, " ", name)

    # Step 4: Tokenise → remove stop-words → remove short tokens
    tokens: list[str] = [
        tok
        for tok in name.split()
        if tok not in NAME_STOP_WORDS and len(tok) >= NAME_MIN_TOKEN_LENGTH
    ]

    return " ".join(tokens)


def build_norm_column(
    df: pd.DataFrame,
    raw_col: str,
    norm_col: str = COL_FUND_NAME_NORM,
) -> pd.DataFrame:
    """
    Add a normalised fund name column to a DataFrame in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the raw fund name column.
    raw_col : str
        Name of the column holding raw fund names.
    norm_col : str, optional
        Name for the new normalised column.
        Defaults to ``config.COL_FUND_NAME_NORM``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with ``norm_col`` appended.

    Raises
    ------
    KeyError
        If ``raw_col`` is not present in ``df``.
    """
    if raw_col not in df.columns:
        raise KeyError(
            f"build_norm_column: column '{raw_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    df[norm_col] = df[raw_col].apply(normalize_fund_name)
    logger.debug(
        f"Normalised column '{raw_col}' → '{norm_col}'  "
        f"({df[norm_col].nunique()} unique keys from {df[raw_col].nunique()} raw names)"
    )
    return df


def find_best_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.50,
) -> Optional[str]:
    """
    Find the best fuzzy match for ``query`` among ``candidates`` using a simple
    Jaccard token similarity.  Falls back to None if no candidate meets
    ``threshold``.

    This is used as a last resort when exact normalised key matching fails
    (e.g. "Franklin India NSE Nifty 50 Index" vs
    "Franklin India INDEX FUND NSE NIFTY 50 INDEX FUND").

    Parameters
    ----------
    query : str
        Already-normalised query string.
    candidates : list[str]
        Already-normalised candidate strings to search.
    threshold : float, optional
        Minimum Jaccard similarity score to accept a match.  Default 0.50.

    Returns
    -------
    str or None
        The best-matching candidate string, or None if nothing meets the
        threshold.

    Notes
    -----
    Jaccard similarity for token sets A and B:
        J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not candidates:
        return None

    q_tokens: set[str] = set(query.split())
    best_score: float = -1.0
    best_match: Optional[str] = None

    for candidate in candidates:
        c_tokens: set[str] = set(candidate.split())
        if not q_tokens and not c_tokens:
            continue
        intersection = q_tokens & c_tokens
        union = q_tokens | c_tokens
        score = len(intersection) / len(union) if union else 0.0
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATAFRAME VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str = "DataFrame",
    min_rows: int = 1,
) -> None:
    """
    Assert that a DataFrame has required columns and a minimum number of rows.
    Raises a descriptive exception immediately — fail-fast, fail-loud.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required_columns : list[str]
        Column names that MUST be present.
    df_name : str, optional
        Human-readable name for the DataFrame, used in error messages.
    min_rows : int, optional
        Minimum acceptable number of rows.  Default 1.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If required columns are missing or the row count is below ``min_rows``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"validate_dataframe: '{df_name}' must be a pandas DataFrame, "
            f"got {type(df).__name__}."
        )

    missing_cols: list[str] = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"validate_dataframe: '{df_name}' is missing required columns: "
            f"{missing_cols}.\nActual columns: {list(df.columns)}"
        )

    if len(df) < min_rows:
        raise ValueError(
            f"validate_dataframe: '{df_name}' has {len(df):,} rows, "
            f"but at least {min_rows:,} are required."
        )

    logger.debug(
        f"✓ '{df_name}' validated — shape: {df.shape}, "
        f"required columns present: {required_columns}"
    )


def report_missing(df: pd.DataFrame, df_name: str = "DataFrame") -> pd.Series:
    """
    Log a structured missing-value report for a DataFrame and return the
    missing-count Series for programmatic use.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    df_name : str, optional
        Human-readable label for log messages.

    Returns
    -------
    pd.Series
        Series of missing counts, indexed by column name, containing only
        columns with at least one missing value.
    """
    missing: pd.Series = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        logger.info(f"'{df_name}': no missing values detected.")
    else:
        pct = (missing / len(df) * 100).round(2)
        report_df = pd.DataFrame({"Missing": missing, "Pct": pct})
        logger.warning(
            f"'{df_name}' — missing values detected:\n{report_df.to_string()}"
        )
    return missing


def report_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    df_name: str = "DataFrame",
) -> int:
    """
    Log the number of duplicate rows in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    subset : list[str] or None, optional
        Columns to consider for duplicate detection.
        None means all columns.
    df_name : str, optional
        Human-readable label for log messages.

    Returns
    -------
    int
        Number of duplicate rows found.
    """
    n_dups: int = df.duplicated(subset=subset).sum()
    if n_dups == 0:
        logger.debug(f"'{df_name}': no duplicates found.")
    else:
        context = f"(subset={subset})" if subset else "(all columns)"
        logger.warning(
            f"'{df_name}': {n_dups:,} duplicate rows detected {context}."
        )
    return n_dups


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — NUMERIC / TYPE COERCION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_to_numeric(
    series: pd.Series,
    col_name: str = "column",
    coerce_errors: bool = True,
) -> pd.Series:
    """
    Convert a pandas Series to numeric dtype, with structured logging of any
    values that fail conversion.

    Parameters
    ----------
    series : pd.Series
        Input Series to convert.
    col_name : str, optional
        Column name used in log messages.
    coerce_errors : bool, optional
        If True, non-convertible values become NaN (``errors='coerce'``).
        If False, raises on the first bad value.

    Returns
    -------
    pd.Series
        Numeric Series (float64).

    Notes
    -----
    A before/after NaN count comparison identifies how many values were
    coerced to NaN during conversion and logs them as warnings.
    """
    before_nulls: int = series.isna().sum()
    errors_mode = "coerce" if coerce_errors else "raise"
    converted = pd.to_numeric(series, errors=errors_mode)
    after_nulls: int = converted.isna().sum()
    newly_coerced: int = after_nulls - before_nulls

    if newly_coerced > 0:
        logger.warning(
            f"safe_to_numeric: '{col_name}' — {newly_coerced} values could not "
            f"be converted to numeric and were set to NaN."
        )
    else:
        logger.debug(f"safe_to_numeric: '{col_name}' converted cleanly.")

    return converted


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    df_name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Apply ``safe_to_numeric`` to a list of columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify (a copy is NOT made — operates in-place).
    columns : list[str]
        Column names to coerce.
    df_name : str, optional
        Label for log messages.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame.
    """
    for col in columns:
        if col in df.columns:
            df[col] = safe_to_numeric(df[col], col_name=f"{df_name}.{col}")
        else:
            logger.warning(
                f"coerce_numeric_columns: column '{col}' not found in '{df_name}' — skipped."
            )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def parse_date_column(
    df: pd.DataFrame,
    col: str = COL_DATE,
    df_name: str = "DataFrame",
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Parse a string date column to ``pd.Timestamp`` (datetime64[ns]) in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the date column.
    col : str, optional
        Name of the date column to parse.
    df_name : str, optional
        Label for log messages.
    dayfirst : bool, optional
        Whether to interpret the first token of the date string as the day.
        False (default) for YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with the column converted to datetime64[ns].

    Raises
    ------
    KeyError
        If ``col`` is not in ``df``.
    """
    if col not in df.columns:
        raise KeyError(
            f"parse_date_column: column '{col}' not found in '{df_name}'. "
            f"Available: {list(df.columns)}"
        )

    before_nulls = df[col].isna().sum()
    df[col] = pd.to_datetime(df[col], dayfirst=dayfirst, errors="coerce")
    after_nulls = df[col].isna().sum()
    newly_null = after_nulls - before_nulls

    if newly_null > 0:
        logger.warning(
            f"parse_date_column: '{df_name}.{col}' — {newly_null} values "
            f"failed to parse and became NaT."
        )
    else:
        date_range = f"{df[col].min().date()} → {df[col].max().date()}"
        logger.info(
            f"parse_date_column: '{df_name}.{col}' parsed cleanly. "
            f"Range: {date_range}"
        )
    return df


def extract_year_month(
    df: pd.DataFrame,
    date_col: str = COL_DATE,
    ym_col: str = "Year_Month",
) -> pd.DataFrame:
    """
    Add a ``pd.Period`` (monthly frequency) column from a datetime column.

    This period column is used as the merge key when joining fund return data
    with factor data (both reduced to YYYY-MM granularity).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed datetime column.
    date_col : str, optional
        Name of the datetime column.
    ym_col : str, optional
        Name for the new period column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new period column appended.
    """
    if date_col not in df.columns:
        raise KeyError(f"extract_year_month: '{date_col}' not found in DataFrame.")

    df[ym_col] = df[date_col].dt.to_period("M")
    logger.debug(
        f"extract_year_month: '{ym_col}' created from '{date_col}' "
        f"({df[ym_col].nunique()} unique months)."
    )
    return df


def filter_date_range(
    df: pd.DataFrame,
    date_col: str,
    start: str,
    end: str,
    df_name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Filter a DataFrame to rows whose date column falls within [start, end].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter.
    date_col : str
        Name of the datetime column to filter on.
    start : str
        Inclusive lower bound in "YYYY-MM-DD" format.
    end : str
        Inclusive upper bound in "YYYY-MM-DD" format.
    df_name : str, optional
        Label for log messages.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (new object — original is unchanged).
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    mask = (df[date_col] >= start_ts) & (df[date_col] <= end_ts)
    filtered = df.loc[mask].copy()

    n_dropped = len(df) - len(filtered)
    logger.info(
        f"filter_date_range: '{df_name}' — retained {len(filtered):,} / {len(df):,} rows "
        f"({n_dropped:,} dropped outside [{start} → {end}])."
    )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RETURN / FINANCIAL CALCULATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_return_from_nav(nav_series: pd.Series) -> pd.Series:
    """
    Compute month-over-month simple returns from a NAV series.

    Formula:
        R_t = (NAV_t - NAV_{t-1}) / NAV_{t-1}

    Parameters
    ----------
    nav_series : pd.Series
        Ordered series of NAV values for a single fund.

    Returns
    -------
    pd.Series
        Monthly simple returns (decimal).  The first element is NaN because
        there is no prior period.

    Notes
    -----
    Uses ``pd.Series.pct_change()`` which implements the same formula and
    correctly returns NaN for the first observation.  The result is NOT
    multiplied by 100 — we keep returns in decimal form throughout.
    """
    return nav_series.pct_change()


def annualize_return(monthly_return_mean: float) -> float:
    """
    Convert mean monthly return to annualised return using compounding.

    Formula:
        Annualized Return = (1 + mean_monthly_return)^12 - 1

    Parameters
    ----------
    monthly_return_mean : float
        Mean monthly simple return in decimal form.

    Returns
    -------
    float
        Annualised return in decimal form.

    Examples
    --------
    >>> annualize_return(0.01)   # 1% monthly → ~12.68% annualised
    0.1268250301319698
    """
    return (1.0 + monthly_return_mean) ** 12 - 1.0


def annualize_volatility(monthly_return_std: float) -> float:
    """
    Scale monthly return standard deviation to annualised volatility.

    Formula:
        Annualized Volatility = monthly_std × √12

    Parameters
    ----------
    monthly_return_std : float
        Standard deviation of monthly simple returns (decimal).

    Returns
    -------
    float
        Annualised volatility in decimal form.
    """
    return monthly_return_std * np.sqrt(12.0)


def compute_cagr(
    initial_nav: float,
    final_nav: float,
    n_months: int,
) -> float:
    """
    Compute Compound Annual Growth Rate from start/end NAV values.

    Formula:
        CAGR = (Final_NAV / Initial_NAV)^(12 / n_months) - 1

    Parameters
    ----------
    initial_nav : float
        NAV at the start of the observation period.
    final_nav : float
        NAV at the end of the observation period.
    n_months : int
        Number of months in the observation period.

    Returns
    -------
    float
        CAGR in decimal form, or NaN if inputs are invalid.
    """
    if initial_nav <= 0 or final_nav <= 0 or n_months <= 0:
        logger.warning(
            f"compute_cagr: invalid inputs "
            f"(initial={initial_nav}, final={final_nav}, months={n_months})."
        )
        return np.nan
    return (final_nav / initial_nav) ** (12.0 / n_months) - 1.0


def compute_sharpe_ratio(
    excess_returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Compute the Sharpe ratio from a series of excess returns.

    The Sharpe ratio is:
        Sharpe = mean(Excess_Return) / std(Excess_Return)

    When ``annualize=True``, both numerator and denominator are scaled to an
    annual basis before dividing (equivalent to multiplying the monthly Sharpe
    by √12).

    Parameters
    ----------
    excess_returns : pd.Series
        Series of monthly excess returns (Return − RF), in decimal form.
    annualize : bool, optional
        If True, return the annualised Sharpe ratio.  Default True.

    Returns
    -------
    float
        Sharpe ratio (annualised if requested), or NaN if insufficient data.
    """
    clean = excess_returns.dropna()
    if len(clean) < 2:
        return np.nan
    mean_er = clean.mean()
    std_er  = clean.std(ddof=1)
    if std_er == 0:
        return np.nan
    monthly_sharpe = mean_er / std_er
    if annualize:
        return monthly_sharpe * np.sqrt(12.0)
    return monthly_sharpe


def compute_max_drawdown(nav_or_cumret: pd.Series) -> float:
    """
    Compute maximum drawdown from a NAV series or cumulative return series.

    Maximum drawdown measures the largest peak-to-trough decline before a new
    peak is reached.

    Formula:
        For each time t, drawdown_t = (peak_t - value_t) / peak_t
        Max Drawdown = max(drawdown_t) over all t

    Parameters
    ----------
    nav_or_cumret : pd.Series
        Ordered series of NAV values or cumulative return index values.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.30 = 30% drawdown).
        Returns NaN if the series has fewer than 2 values.
    """
    clean = nav_or_cumret.dropna()
    if len(clean) < 2:
        return np.nan
    running_max = clean.cummax()
    drawdown    = (running_max - clean) / running_max
    return float(drawdown.max())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTLIER DETECTION & WINSORISATION
# ══════════════════════════════════════════════════════════════════════════════

def winsorize_returns(
    series: pd.Series,
    lower: float = RETURN_LOWER_BOUND,
    upper: float = RETURN_UPPER_BOUND,
    col_name: str = "returns",
) -> pd.Series:
    """
    Winsorise (clip) a return series to the range [lower, upper].

    Values outside the bounds are replaced by the bound value — they are NOT
    removed.  Each capped value is logged so the user can inspect them.

    Parameters
    ----------
    series : pd.Series
        Monthly return series in decimal form.
    lower : float, optional
        Lower clip bound.  Default: ``config.RETURN_LOWER_BOUND`` (-50%).
    upper : float, optional
        Upper clip bound.  Default: ``config.RETURN_UPPER_BOUND`` (+100%).
    col_name : str, optional
        Column name for log messages.

    Returns
    -------
    pd.Series
        Winsorised series.

    Notes
    -----
    The -90.38% minimum observed in the passive fund data triggered this
    guard.  That value likely represents an ETF corporate-action or data error.
    """
    n_lower = (series < lower).sum()
    n_upper = (series > upper).sum()

    if n_lower > 0 or n_upper > 0:
        logger.warning(
            f"winsorize_returns: '{col_name}' — clipping {n_lower} values below "
            f"{lower:.0%} and {n_upper} values above {upper:.0%}."
        )

    return series.clip(lower=lower, upper=upper)


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 3.0,
    col_name: str = "series",
) -> pd.Series:
    """
    Return a boolean mask identifying outliers using the IQR method.

    A value is an outlier if it falls outside:
        [Q1 - multiplier × IQR,  Q3 + multiplier × IQR]

    Parameters
    ----------
    series : pd.Series
        Numeric series to analyse.
    multiplier : float, optional
        IQR multiplier.  3.0 (default) is conservative — flags only extreme
        outliers without flagging moderate fat tails.
    col_name : str, optional
        Column name for log messages.

    Returns
    -------
    pd.Series
        Boolean Series: True where the value is an outlier.
    """
    clean = series.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - multiplier * iqr
    upper_fence = q3 + multiplier * iqr
    mask = (series < lower_fence) | (series > upper_fence)
    n_outliers = mask.sum()
    if n_outliers > 0:
        logger.warning(
            f"detect_outliers_iqr: '{col_name}' — {n_outliers} outliers detected "
            f"(IQR×{multiplier}: [{lower_fence:.4f}, {upper_fence:.4f}])."
        )
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    description: str = "DataFrame",
    index: bool = CSV_INDEX,
    encoding: str = CSV_ENCODING,
    date_format: str = CSV_DATE_FORMAT,
) -> None:
    """
    Save a DataFrame to CSV with structured logging and error handling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Destination file path (directory is created if absent).
    description : str, optional
        Human-readable label for log messages.
    index : bool, optional
        Whether to write the row index.  Default: ``config.CSV_INDEX`` (False).
    encoding : str, optional
        File encoding.  Default: ``config.CSV_ENCODING`` ("utf-8-sig" for
        Excel compatibility).
    date_format : str, optional
        strftime format string for datetime columns.

    Raises
    ------
    OSError
        If the file cannot be written (permissions, disk full, etc.).
    """
    ensure_dir(path.parent)
    try:
        df.to_csv(
            path,
            index=index,
            encoding=encoding,
            date_format=date_format,
        )
        size_kb = path.stat().st_size / 1024
        logger.info(
            f"Saved '{description}' → {path.name}  "
            f"({len(df):,} rows × {len(df.columns)} cols, {size_kb:.1f} KB)"
        )
    except OSError as exc:
        logger.error(f"Failed to save '{description}' to {path}: {exc}")
        raise


def save_text_report(
    content: str,
    path: Path,
    description: str = "Report",
) -> None:
    """
    Write a plain-text report string to a file.

    Parameters
    ----------
    content : str
        Report content.
    path : Path
        Destination file path.
    description : str, optional
        Label for log messages.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    ensure_dir(path.parent)
    try:
        path.write_text(content, encoding="utf-8")
        logger.info(f"Saved '{description}' → {path.name}  ({len(content):,} chars)")
    except OSError as exc:
        logger.error(f"Failed to save report to {path}: {exc}")
        raise


def format_pct(value: float, decimals: int = 2) -> str:
    """
    Format a decimal fraction as a percentage string.

    Parameters
    ----------
    value : float
        Decimal value (e.g. 0.0612 → "6.12%").
    decimals : int, optional
        Number of decimal places.  Default 2.

    Returns
    -------
    str
        Formatted percentage string.

    Examples
    --------
    >>> format_pct(0.0612)
    '6.12%'
    >>> format_pct(np.nan)
    'N/A'
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """
    Format a float to a fixed-decimal string, handling NaN gracefully.

    Parameters
    ----------
    value : float
        Numeric value to format.
    decimals : int, optional
        Number of decimal places.  Default 4.

    Returns
    -------
    str
        Formatted string, or "N/A" for NaN/Inf.
    """
    if pd.isna(value) or np.isinf(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def summarize_series(series: pd.Series, name: str = "Series") -> dict[str, Any]:
    """
    Compute a comprehensive summary dictionary for a numeric series.
    Useful for building report tables without repeating boilerplate.

    Parameters
    ----------
    series : pd.Series
        Numeric series to summarise.
    name : str, optional
        Label for the "Name" key in the output dict.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: Name, N, Mean, Median, Std, Min, Max, Q1, Q3.
    """
    clean = series.dropna()
    return {
        "Name":   name,
        "N":      len(clean),
        "Mean":   clean.mean(),
        "Median": clean.median(),
        "Std":    clean.std(ddof=1),
        "Min":    clean.min(),
        "Max":    clean.max(),
        "Q1":     clean.quantile(0.25),
        "Q3":     clean.quantile(0.75),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SELF-TEST (run directly: python src/utils.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys as _sys
    import traceback
    # Allow running directly as: python src/utils.py from the project root
    _proj_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    if _proj_root not in _sys.path:
        _sys.path.insert(0, _proj_root)

    _log = setup_logger("utils_selftest", level="DEBUG")
    _log.info("═" * 60)
    _log.info("  UTILS SELF-TEST")
    _log.info("═" * 60)

    errors: list[str] = []

    # ── Test 1: normalize_fund_name ───────────────────────────────────────────
    test_cases = [
        ("HDFC NIFTY 50 ETF - Growth Plan",                  "hdfc nifty 50 etf"),
        ("Kotak Large Cap Fund - Growth - Direct",            "kotak large cap"),
        ("  ICICI Prudential BSE Sensex ETF  ",               "icici prudential bse sensex etf"),
        ("Franklin India INDEX FUND- NSE NIFTY 50 INDEX FUND- Direct - Growth",
                                                              "franklin india index nse nifty 50 index"),
        ("UTI Nifty 50 Index Fund - Direct Plan - IDCW",      "uti nifty 50 index"),
    ]
    _log.info("\nTest 1: normalize_fund_name")
    for raw, expected in test_cases:
        result = normalize_fund_name(raw)
        status = "✅" if result == expected else "❌"
        _log.info(f"  {status}  '{raw[:50]}...' → '{result}'  (expected: '{expected}')")
        if result != expected:
            errors.append(f"normalize_fund_name: '{raw}' → '{result}' ≠ '{expected}'")

    # ── Test 2: find_best_match ────────────────────────────────────────────────
    _log.info("\nTest 2: find_best_match (Jaccard similarity)")
    candidates = [
        "hdfc nifty 50 etf",
        "icici prudential bse sensex etf",
        "kotak large cap",
    ]
    result2 = find_best_match("hdfc nifty 50 index", candidates)
    _log.info(f"  Query: 'hdfc nifty 50 index' → Best match: '{result2}'")

    # ── Test 3: Financial math ─────────────────────────────────────────────────
    _log.info("\nTest 3: Financial calculation helpers")
    ann_ret = annualize_return(0.01)
    _log.info(f"  annualize_return(0.01)         = {ann_ret:.6f}  (expected ~0.1268)")
    ann_vol = annualize_volatility(0.03)
    _log.info(f"  annualize_volatility(0.03)     = {ann_vol:.6f}  (expected ~0.1039)")
    cagr = compute_cagr(100.0, 200.0, 60)
    _log.info(f"  compute_cagr(100→200, 60mo)    = {cagr:.6f}  (expected ~0.1487)")

    # ── Test 4: Sharpe & drawdown ──────────────────────────────────────────────
    _log.info("\nTest 4: Sharpe ratio & max drawdown")
    np.random.seed(42)
    fake_er = pd.Series(np.random.normal(0.01, 0.04, 60))
    sharpe = compute_sharpe_ratio(fake_er)
    _log.info(f"  Sharpe (60 random obs, ann)    = {sharpe:.4f}")
    nav_series = pd.Series([100, 110, 105, 95, 115, 108, 120])
    mdd = compute_max_drawdown(nav_series)
    _log.info(f"  Max Drawdown [100…120 series]  = {mdd:.4f}  (expected ~0.1364)")

    # ── Test 5: Winsorisation ──────────────────────────────────────────────────
    _log.info("\nTest 5: winsorize_returns")
    raw_returns = pd.Series([-0.95, -0.30, 0.01, 0.05, 1.50])
    winsorised  = winsorize_returns(raw_returns, col_name="test_returns")
    _log.info(f"  Input:  {raw_returns.tolist()}")
    _log.info(f"  Output: {winsorised.tolist()}")

    # ── Test 6: format_pct / format_number ────────────────────────────────────
    _log.info("\nTest 6: format helpers")
    _log.info(f"  format_pct(0.0612)      = '{format_pct(0.0612)}'")
    _log.info(f"  format_pct(np.nan)      = '{format_pct(np.nan)}'")
    _log.info(f"  format_number(3.14159)  = '{format_number(3.14159)}'")

    # ── Summary ────────────────────────────────────────────────────────────────
    _log.info("\n" + "─" * 60)
    if errors:
        _log.error(f"❌  {len(errors)} test(s) FAILED:")
        for e in errors:
            _log.error(f"    {e}")
    else:
        _log.info("✅  All utils self-tests passed.")
    _log.info("═" * 60 + "\n")
