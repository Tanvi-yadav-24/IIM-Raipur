"""
return_calculations.py
======================
Dedicated return-engineering module for the Carhart Four-Factor Pipeline.

Responsibility
--------------
Accepts the clean master DataFrame produced by ``preprocessing.py`` and adds
every return-derived feature needed by downstream modules:

  1.  Re-verify / recompute monthly simple returns from NAV (authoritative).
  2.  Compute log returns  ln(NAV_t / NAV_{t-1}).
  3.  Compute excess returns  R_t − RF_t  (with actual or synthetic RF).
  4.  Build per-fund cumulative return wealth index (starting at 1.0).
  5.  Compute rolling trailing-window mean returns (3 M, 6 M, 12 M).
  6.  Produce a fund-level summary DataFrame (annualised return, vol,
      Sharpe, CAGR, max drawdown, start/end NAV, n_obs).
  7.  Extract a regression-ready long-format DataFrame (eligible funds only,
      with all factor columns present — initially NaN; filled by factor_merge).

Why a dedicated module?
-----------------------
``preprocessing.py`` focuses on data quality (validation, normalisation, merge).
``return_calculations.py`` focuses on financial feature engineering.  Keeping
them separate means:
  * The preprocessing pipeline can be rerun cheaply without re-engineering features.
  * Return feature logic has a single home, making it easy to add new metrics.
  * Unit-testing each layer independently is straightforward.

Ground-truth context (from dataset inspection — DO NOT REMOVE)
--------------------------------------------------------------
* Monthly_Return is already pre-computed in the raw files and cross-validated
  in preprocessing.py.  This module recomputes it from NAV once more as the
  definitive value (pct_change within each fund group).
* Excess_Return currently uses SYNTHETIC_RF_MONTHLY (≈ 0.5%/month) as a
  placeholder.  ``factor_merge.py`` will overwrite this with actual RF values
  once the synthetic factor table is built.
* The -90.38% outlier in passive data was already winsorised to -50% in
  preprocessing.py, so no additional capping is needed here.
* Bajaj Finserv (20 obs) and Motilal Oswal (26 obs) are in the master frame
  but flagged as Eligible_For_Regression = False.

Module layout
-------------
  §1  compute_simple_returns       — NAV → pct_change per fund
  §2  compute_log_returns          — NAV → ln(NAV_t/NAV_{t-1}) per fund
  §3  compute_excess_returns       — Return − RF (scalar or per-row series)
  §4  compute_cumulative_returns   — wealth index per fund
  §5  compute_rolling_returns      — rolling trailing mean (3 / 6 / 12 M)
  §6  build_fund_summary           — per-fund annualised metrics DataFrame
  §7  get_regression_df            — regression-ready long format (eligible only)
  §8  enrich_master_frame          — orchestrator: calls §1–§5, returns enriched df
  §9  Self-test (python src/return_calculations.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Path fix for direct execution ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    COL_ANN_RETURN,
    COL_ANN_VOL,
    COL_CAGR,
    COL_CUMULATIVE_RETURN,
    COL_DATE,
    COL_ER,
    COL_EXCESS_RETURN,
    COL_FUND_NAME,
    COL_FUND_TYPE,
    COL_HML,
    COL_LOG_RETURN,
    COL_MAX_DRAWDOWN,
    COL_MF,
    COL_NAV,
    COL_RF,
    COL_RF_SOURCE,
    COL_RETURN,
    COL_ROLLING_12M,
    COL_ROLLING_3M,
    COL_ROLLING_6M,
    COL_SHARPE,
    COL_SMB,
    COL_WML,
    COL_YEAR_MONTH,
    FUND_TYPE_ACTIVE,
    FUND_TYPE_PASSIVE,
    MIN_OBS,
    MONTHS_PER_YEAR,
    SYNTHETIC_RF_MONTHLY,
)
from src.utils import (
    annualize_return,
    annualize_volatility,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe_ratio,
    report_missing,
    save_dataframe,
    setup_logger,
    validate_dataframe,
)

logger = setup_logger(__name__)

# Column name for the Eligible_For_Regression flag (set in preprocessing)
_ELIGIBLE_COL = "Eligible_For_Regression"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIMPLE MONTHLY RETURNS (authoritative recomputation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Re-)compute monthly simple returns from NAV within each fund group and
    write the result into ``COL_RETURN``.

    Formula
    -------
    R_t = (NAV_t − NAV_{t-1}) / NAV_{t-1}   ≡   NAV.pct_change()

    This overwrites the pre-computed column that was validated in
    ``preprocessing.py``.  Running it here makes ``return_calculations.py``
    self-sufficient and ensures every downstream module sees a freshly derived,
    fund-boundary-safe return series.

    Parameters
    ----------
    df : pd.DataFrame
        Master frame sorted by (Fund_Name, Date), containing COL_NAV.

    Returns
    -------
    pd.DataFrame
        Same frame with COL_RETURN recomputed in-place.

    Notes
    -----
    * ``groupby(..., sort=False)`` preserves the existing sort order.
    * The first observation per fund becomes NaN; these rows are dropped
      afterwards by ``enrich_master_frame``.
    * Numeric precision: float64 throughout — no rounding applied.
    """
    logger.info("  §1 compute_simple_returns: recomputing Monthly_Return from NAV …")
    df = df.copy()

    df[COL_RETURN] = (
        df.groupby(COL_FUND_NAME, sort=False)[COL_NAV]
        .transform(lambda s: s.pct_change())
    )

    n_nan = df[COL_RETURN].isna().sum()
    n_funds = df[COL_FUND_NAME].nunique()
    logger.info(
        f"    Recomputed returns: {len(df) - n_nan:,} valid  |  "
        f"{n_nan} NaN (1 per fund × {n_funds} funds = {n_funds} expected)"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOG RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute continuously-compounded (log) monthly returns per fund.

    Formula
    -------
    r_t = ln(NAV_t / NAV_{t-1})   ≡   ln(1 + R_t)

    Log returns are preferred for:
      * Additive aggregation over time (convenient for multi-period analysis).
      * More symmetric distribution than simple returns.
      * Some risk model specifications.

    They are computed here for completeness but the Carhart regression uses
    simple excess returns (standard in academic factor models).

    Parameters
    ----------
    df : pd.DataFrame
        Master frame sorted by (Fund_Name, Date), containing COL_NAV.

    Returns
    -------
    pd.DataFrame
        Same frame with ``COL_LOG_RETURN`` column added.

    Notes
    -----
    * ``np.log(NAV_t / NAV_{t-1})`` is numerically equivalent to
      ``np.log1p(R_t)`` when both are finite.
    * The first observation per fund is NaN (no prior NAV).
    * Negative NAV would produce NaN; guarded with a positivity check.
    """
    logger.info("  §2 compute_log_returns: computing Log_Return …")
    df = df.copy()

    # Compute log return within each fund group
    def _log_ret(nav_series: pd.Series) -> pd.Series:
        shifted = nav_series.shift(1)
        ratio   = nav_series / shifted
        # Guard against zero / negative NAV
        ratio   = ratio.where(ratio > 0, other=np.nan)
        return np.log(ratio)

    df[COL_LOG_RETURN] = (
        df.groupby(COL_FUND_NAME, sort=False)[COL_NAV]
        .transform(_log_ret)
    )

    n_nan = df[COL_LOG_RETURN].isna().sum()
    logger.info(
        f"    Log returns computed.  NaN count: {n_nan} "
        f"(expected ≈ {df[COL_FUND_NAME].nunique()})."
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXCESS RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_excess_returns(
    df: pd.DataFrame,
    rf_series: Optional[pd.Series] = None,
    rf_scalar: float = SYNTHETIC_RF_MONTHLY,
) -> pd.DataFrame:
    """
    Compute monthly excess returns   Excess_Return = R_t − RF_t.

    Two modes
    ---------
    1.  **Per-row RF series** (preferred — used when ``factor_merge.py`` has
        produced a per-month RF column in the DataFrame).
        ``rf_series`` must be a Series aligned to ``df``'s index, containing
        monthly RF values in decimal form (e.g. 0.005 = 0.5%).

    2.  **Scalar RF** (fallback — used with synthetic factors).
        A single constant monthly RF is subtracted from every return.
        Default: ``config.SYNTHETIC_RF_MONTHLY`` (6% p.a. / 12 = 0.5%/month).

    Also adds ``COL_RF_SOURCE`` to record which RF was used — critical for
    reproducibility and for interpreting alpha magnitudes correctly.

    Parameters
    ----------
    df : pd.DataFrame
        Master frame with COL_RETURN column.
    rf_series : pd.Series, optional
        Per-row RF values (decimal, aligned to df.index).
        If supplied, COL_RF must already exist in ``df`` (from factor_merge).
    rf_scalar : float, optional
        Constant monthly RF in decimal form.
        Used when ``rf_series`` is None.

    Returns
    -------
    pd.DataFrame
        Frame with updated COL_EXCESS_RETURN and COL_RF_SOURCE columns.

    Notes
    -----
    * ``COL_RF`` is also written into the DataFrame so regression_analysis.py
      can use it directly.
    * Excess_Return NaN propagation: if R_t is NaN, Excess_Return_t is NaN.
    """
    logger.info("  §3 compute_excess_returns: updating Excess_Return …")
    df = df.copy()

    if rf_series is not None and COL_RF in df.columns:
        # ── Mode 1: Per-row RF from factor table ─────────────────────────────
        df[COL_EXCESS_RETURN] = df[COL_RETURN] - df[COL_RF]
        df[COL_RF_SOURCE] = "provided"
        rf_mean = df[COL_RF].mean()
        logger.info(
            f"    Mode: per-row RF from factor table.  "
            f"Mean RF = {rf_mean:.4%}/month."
        )
    else:
        # ── Mode 2: Constant synthetic RF ────────────────────────────────────
        df[COL_EXCESS_RETURN] = df[COL_RETURN] - rf_scalar
        df[COL_RF] = rf_scalar          # constant RF column for regression
        df[COL_RF_SOURCE] = "synthetic"
        logger.info(
            f"    Mode: constant synthetic RF = {rf_scalar:.4%}/month  "
            f"({rf_scalar * MONTHS_PER_YEAR:.2%} annualised).  "
            f"Will be overwritten by factor_merge.py."
        )

    # Quick sanity: mean excess return should be close to mean simple return
    mean_ret = df[COL_RETURN].mean()
    mean_exc = df[COL_EXCESS_RETURN].mean()
    logger.info(
        f"    Mean simple return : {mean_ret:.4%}/month  |  "
        f"Mean excess return : {mean_exc:.4%}/month  |  "
        f"Difference (≈ RF): {mean_ret - mean_exc:.4%}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CUMULATIVE RETURN (WEALTH INDEX)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-fund cumulative return wealth index starting at 1.0 at the
    fund's first available observation.

    Formula
    -------
    CumReturn_t = ∏(1 + R_i)   for i = 1 … t
                = cumprod(1 + Monthly_Return)   within each fund

    The index starts at 1.0 at the first non-NaN return (so the fund's
    first observation, which has NaN return, gets CumReturn = 1.0).

    Parameters
    ----------
    df : pd.DataFrame
        Master frame sorted by (Fund_Name, Date) with COL_RETURN.

    Returns
    -------
    pd.DataFrame
        Frame with COL_CUMULATIVE_RETURN added.

    Notes
    -----
    * NaN returns within the series (not just the first obs) are treated as
      0% return for the cumulative product — i.e., wealth stays flat during
      gap months.  The gap is already flagged by ``validate_monthly_continuity``
      in preprocessing.py.
    * The wealth index is useful for plotting NAV-normalised performance
      curves across funds with very different starting NAV levels.
    """
    logger.info("  §4 compute_cumulative_returns: building wealth index …")
    df = df.copy()

    def _cumprod_wealth(ret_series: pd.Series) -> pd.Series:
        # Fill NaN (first-obs) with 0.0 so the first row gives (1+0) = 1.0
        filled = ret_series.fillna(0.0)
        return (1.0 + filled).cumprod()

    df[COL_CUMULATIVE_RETURN] = (
        df.groupby(COL_FUND_NAME, sort=False)[COL_RETURN]
        .transform(_cumprod_wealth)
    )

    # Sanity: min value should be positive (can't go below 0 with simple returns)
    min_cum = df[COL_CUMULATIVE_RETURN].min()
    max_cum = df[COL_CUMULATIVE_RETURN].max()
    logger.info(
        f"    Wealth index range across all funds/months: "
        f"[{min_cum:.4f}, {max_cum:.4f}]"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ROLLING TRAILING-WINDOW RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_returns(
    df: pd.DataFrame,
    windows: list[int] = [3, 6, 12],
) -> pd.DataFrame:
    """
    Compute trailing rolling mean simple returns for specified window lengths.

    Formula (for window w)
    ----------------------
    RollingReturn_t(w) = mean(R_{t−w+1}, …, R_t)

    This is the arithmetic average over the trailing ``w`` months.  It differs
    from annualised geometric rolling returns (used in professional attribution)
    but is simpler, faster, and sufficient for visual trend analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Master frame sorted by (Fund_Name, Date) with COL_RETURN.
    windows : list[int], optional
        Rolling window sizes in months.  Default: [3, 6, 12].

    Returns
    -------
    pd.DataFrame
        Frame with new columns: Rolling_Return_3M, Rolling_Return_6M,
        Rolling_Return_12M (or whatever windows are requested).

    Notes
    -----
    * ``min_periods = window`` — a full window is required before any value
      is output (leading NaN for the first ``window - 1`` rows per fund).
    * Computed within each fund group so fund boundaries are respected.
    * Maps window size to config column constants:
        3  → COL_ROLLING_3M
        6  → COL_ROLLING_6M
        12 → COL_ROLLING_12M
    """
    logger.info(
        f"  §5 compute_rolling_returns: windows = {windows} months …"
    )
    df = df.copy()

    # Map window size → column name
    _WINDOW_COL_MAP: dict[int, str] = {
        3:  COL_ROLLING_3M,
        6:  COL_ROLLING_6M,
        12: COL_ROLLING_12M,
    }

    for w in windows:
        col_name = _WINDOW_COL_MAP.get(w, f"Rolling_Return_{w}M")

        df[col_name] = (
            df.groupby(COL_FUND_NAME, sort=False)[COL_RETURN]
            .transform(
                lambda s, window=w: s.rolling(window=window, min_periods=window).mean()
            )
        )

        n_valid = df[col_name].notna().sum()
        logger.info(
            f"    Window {w:>2}M → '{col_name}':  {n_valid:,} valid values  "
            f"(NaN for first {w - 1} months per fund)."
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PER-FUND SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_fund_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a comprehensive per-fund performance summary table.

    Metrics computed
    ----------------
    +---------------------+------------------------------------------------------+
    | Column              | Definition                                           |
    +=====================+======================================================+
    | Fund_Type           | "Active" / "Passive"                                 |
    | N_Obs               | # monthly return observations                        |
    | Start_Date          | First available date in analysis window              |
    | End_Date            | Last available date in analysis window               |
    | Start_NAV           | NAV at Start_Date                                    |
    | End_NAV             | NAV at End_Date                                      |
    | CAGR                | Compound annual growth rate (decimal)                |
    | Annualized_Return   | (1 + mean_monthly_ret)^12 − 1                        |
    | Annualized_Vol      | std(monthly_ret) × √12                               |
    | Sharpe_Ratio        | Ann. excess return / Ann. volatility                 |
    | Max_Drawdown        | Largest peak-to-trough NAV decline                   |
    | Expense_Ratio       | Annual expense ratio (%)                             |
    | Eligible_For_Regression | True if N_Obs ≥ MIN_OBS                        |
    +---------------------+------------------------------------------------------+

    Parameters
    ----------
    df : pd.DataFrame
        Enriched master frame (output of ``enrich_master_frame``), with
        Monthly_Return, Excess_Return, NAV, Date, Expense_Ratio.

    Returns
    -------
    pd.DataFrame
        One row per fund.  Indexed by Fund_Name.

    Notes
    -----
    * Uses utility functions from ``utils.py`` (annualize_return,
      annualize_volatility, compute_cagr, compute_sharpe_ratio,
      compute_max_drawdown) which already have their own NaN guards.
    * Sharpe ratio is computed on Excess_Return (not simple return) and
      then annualised by ×√12.
    """
    logger.info("  §6 build_fund_summary: computing per-fund metrics …")

    records: list[dict] = []

    for fund_name, grp in df.groupby(COL_FUND_NAME, sort=True):
        grp = grp.sort_values(COL_DATE)
        rets  = grp[COL_RETURN].dropna()
        exc_r = grp[COL_EXCESS_RETURN].dropna()

        if rets.empty:
            logger.warning(f"    Fund '{fund_name}' has no valid returns — skipped.")
            continue

        n_obs      = len(rets)
        start_date = grp[COL_DATE].min()
        end_date   = grp[COL_DATE].max()
        start_nav  = grp.loc[grp[COL_DATE] == start_date, COL_NAV].iloc[0]
        end_nav    = grp.loc[grp[COL_DATE] == end_date,   COL_NAV].iloc[0]
        fund_type  = grp[COL_FUND_TYPE].iloc[0]
        er         = grp[COL_ER].iloc[0]
        eligible   = grp[_ELIGIBLE_COL].iloc[0] if _ELIGIBLE_COL in grp.columns else (n_obs >= MIN_OBS)

        # ── Performance metrics ───────────────────────────────────────────────
        ann_ret    = annualize_return(rets.mean())
        ann_vol    = annualize_volatility(rets.std(ddof=1))
        cagr_val   = compute_cagr(start_nav, end_nav, n_obs)
        sharpe_val = compute_sharpe_ratio(exc_r, annualize=True)
        mdd_val    = compute_max_drawdown(grp.set_index(COL_DATE)[COL_NAV])

        records.append({
            COL_FUND_NAME:          fund_name,
            COL_FUND_TYPE:          fund_type,
            "N_Obs":                n_obs,
            "Start_Date":           start_date,
            "End_Date":             end_date,
            "Start_NAV":            round(start_nav, 4),
            "End_NAV":              round(end_nav, 4),
            COL_CAGR:               round(cagr_val, 6),
            COL_ANN_RETURN:         round(ann_ret, 6),
            COL_ANN_VOL:            round(ann_vol, 6),
            COL_SHARPE:             round(sharpe_val, 6),
            COL_MAX_DRAWDOWN:       round(mdd_val, 6),
            COL_ER:                 er,
            _ELIGIBLE_COL:          eligible,
        })

    summary_df = pd.DataFrame(records).set_index(COL_FUND_NAME)

    # ── Group-level summary ───────────────────────────────────────────────────
    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        subset = summary_df[summary_df[COL_FUND_TYPE] == ft]
        if subset.empty:
            continue
        logger.info(
            f"\n    ── {ft} funds ({len(subset)}) ──\n"
            f"      Mean CAGR         : {subset[COL_CAGR].mean():.2%}\n"
            f"      Mean Ann. Return  : {subset[COL_ANN_RETURN].mean():.2%}\n"
            f"      Mean Ann. Vol     : {subset[COL_ANN_VOL].mean():.2%}\n"
            f"      Mean Sharpe       : {subset[COL_SHARPE].mean():.4f}\n"
            f"      Mean Max Drawdown : {subset[COL_MAX_DRAWDOWN].mean():.2%}\n"
            f"      Mean Expense Ratio: {subset[COL_ER].mean():.2f}%"
        )

    logger.info(
        f"\n    Fund summary table built: {len(summary_df)} funds  "
        f"({summary_df[_ELIGIBLE_COL].sum()} eligible for regression)."
    )
    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — REGRESSION-READY DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def get_regression_df(
    df: pd.DataFrame,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """
    Extract a clean long-format DataFrame containing only the columns needed
    by ``regression_analysis.py``.

    Columns in output
    -----------------
    Fund_Name, Fund_Name_Norm, Fund_Type, Date, Year_Month,
    Monthly_Return, Excess_Return, RF,
    MF, SMB, HML, WML,         ← NaN until factor_merge.py fills them
    Expense_Ratio, Eligible_For_Regression

    Parameters
    ----------
    df : pd.DataFrame
        Enriched master frame.
    eligible_only : bool, optional
        If True (default), keep only funds where Eligible_For_Regression=True.
        Set to False to include short-history funds (for descriptive analysis).

    Returns
    -------
    pd.DataFrame
        Filtered, column-selected regression DataFrame.

    Notes
    -----
    * Factor columns (MF, SMB, HML, WML) will be NaN at this point — they are
      populated by ``factor_merge.py`` in the next pipeline stage.
    * RF may be a constant scalar column (synthetic) or a per-row series
      (provided); in either case it is present in the output.
    """
    logger.info(
        f"  §7 get_regression_df: extracting regression frame "
        f"({'eligible only' if eligible_only else 'all funds'}) …"
    )

    # ── Columns always required for regression ────────────────────────────────
    base_cols = [
        COL_FUND_NAME,
        COL_FUND_TYPE,
        COL_DATE,
        COL_YEAR_MONTH,
        COL_RETURN,
        COL_EXCESS_RETURN,
        COL_ER,
        _ELIGIBLE_COL,
    ]
    # Factor columns — may be NaN at this stage
    factor_cols = [COL_MF, COL_SMB, COL_HML, COL_WML, COL_RF]

    # Optional normalised name for joins
    if "Fund_Name_Norm" in df.columns:
        base_cols.insert(1, "Fund_Name_Norm")

    # Add factor columns to frame if missing (placeholders)
    for fc in factor_cols:
        if fc not in df.columns:
            df[fc] = np.nan

    all_cols = base_cols + factor_cols
    reg_df = df[all_cols].copy()

    # ── Filter by eligibility ─────────────────────────────────────────────────
    if eligible_only and _ELIGIBLE_COL in reg_df.columns:
        before = reg_df[COL_FUND_NAME].nunique()
        reg_df = reg_df[reg_df[_ELIGIBLE_COL]].copy()
        after  = reg_df[COL_FUND_NAME].nunique()
        logger.info(
            f"    Eligibility filter: {before} → {after} funds "
            f"({before - after} excluded with < {MIN_OBS} observations)."
        )

    # Drop NaN returns (first-obs rows that slipped through)
    n_before = len(reg_df)
    reg_df = reg_df.dropna(subset=[COL_RETURN, COL_EXCESS_RETURN])
    n_dropped = n_before - len(reg_df)
    if n_dropped > 0:
        logger.info(f"    Dropped {n_dropped} rows with NaN return/excess_return.")

    reg_df = reg_df.sort_values([COL_FUND_NAME, COL_DATE]).reset_index(drop=True)

    logger.info(
        f"    Regression frame ready: {len(reg_df):,} rows  |  "
        f"{reg_df[COL_FUND_NAME].nunique()} funds  |  "
        f"Factor cols (NaN until factor_merge): {factor_cols}"
    )
    return reg_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MASTER ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def enrich_master_frame(
    master_df: pd.DataFrame,
    rf_scalar: float = SYNTHETIC_RF_MONTHLY,
    rolling_windows: list[int] = [3, 6, 12],
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate all return-engineering steps and return three DataFrames.

    Pipeline
    --------
    1.  Recompute simple returns from NAV (§1).
    2.  Drop NaN returns (first-obs rows).
    3.  Compute log returns (§2).
    4.  Compute excess returns using ``rf_scalar`` (§3).
    5.  Compute cumulative wealth index (§4).
    6.  Compute rolling trailing-window means (§5).
    7.  Build per-fund summary table (§6).
    8.  Extract regression-ready DataFrame (§7).
    9.  Save enriched master and summary to CSV (optional).

    Parameters
    ----------
    master_df : pd.DataFrame
        Output of ``preprocessing.build_master_frame``.
    rf_scalar : float, optional
        Monthly risk-free rate for excess return computation.
        Overwritten by ``factor_merge.py`` once factor data is available.
    rolling_windows : list[int], optional
        Window sizes for rolling return computation.
    save_outputs : bool, optional
        If True, save enriched master and summary CSV files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (enriched_master, fund_summary, regression_df)

        enriched_master : full long-format frame with all engineered columns.
        fund_summary    : one row per fund with annualised performance metrics.
        regression_df   : long format, eligible funds only, factor cols NaN.
    """
    logger.info("\n" + "=" * 68)
    logger.info("  RETURN CALCULATIONS PHASE — enrich_master_frame")
    logger.info("=" * 68)

    # Validate input
    validate_dataframe(
        master_df,
        required_columns=[COL_FUND_NAME, COL_NAV, COL_RETURN, COL_DATE],
        df_name="master_df (input to return_calculations)",
        min_rows=100,
    )

    # ── §1: Recompute simple returns ──────────────────────────────────────────
    df = compute_simple_returns(master_df)

    # ── Drop NaN returns (first-obs per fund) ─────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=[COL_RETURN]).reset_index(drop=True)
    logger.info(
        f"  Dropped {n_before - len(df)} NaN-return rows (first obs per fund).  "
        f"Remaining: {len(df):,}"
    )

    # ── §2: Log returns ───────────────────────────────────────────────────────
    df = compute_log_returns(df)

    # ── §3: Excess returns ────────────────────────────────────────────────────
    df = compute_excess_returns(df, rf_scalar=rf_scalar)

    # ── §4: Cumulative wealth index ───────────────────────────────────────────
    df = compute_cumulative_returns(df)

    # ── §5: Rolling returns ───────────────────────────────────────────────────
    df = compute_rolling_returns(df, windows=rolling_windows)

    # ── §6: Fund summary ──────────────────────────────────────────────────────
    fund_summary = build_fund_summary(df)

    # ── §7: Regression-ready DataFrame ───────────────────────────────────────
    reg_df = get_regression_df(df, eligible_only=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if save_outputs:
        from src.config import CLEANED_DATA_DIR, REGRESSION_RESULTS_DIR
        from src.utils import ensure_dir
        ensure_dir(CLEANED_DATA_DIR)
        ensure_dir(REGRESSION_RESULTS_DIR)

        enriched_path = CLEANED_DATA_DIR / "master_enriched.csv"
        summary_path  = REGRESSION_RESULTS_DIR / "fund_summary_pre_regression.csv"

        # Year_Month is Period[M] — must be cast to string for CSV export
        export_df = df.copy()
        if COL_YEAR_MONTH in export_df.columns:
            export_df[COL_YEAR_MONTH] = export_df[COL_YEAR_MONTH].astype(str)

        save_dataframe(export_df, enriched_path, description="Enriched master frame")
        save_dataframe(
            fund_summary.reset_index(),
            summary_path,
            description="Pre-regression fund summary",
        )

    # ── Final report ──────────────────────────────────────────────────────────
    type_counts = df.groupby(COL_FUND_TYPE)[COL_FUND_NAME].nunique()
    logger.info(f"\n  ╔══════ RETURN CALCULATIONS SUMMARY ══════╗")
    logger.info(f"  ║  Enriched rows        : {len(df):>8,}        ║")
    logger.info(f"  ║  Funds (Active)       : {type_counts.get(FUND_TYPE_ACTIVE, 0):>8}        ║")
    logger.info(f"  ║  Funds (Passive)      : {type_counts.get(FUND_TYPE_PASSIVE, 0):>8}        ║")
    logger.info(f"  ║  Regression-ready rows: {len(reg_df):>8,}        ║")
    logger.info(f"  ║  Regression funds     : {reg_df[COL_FUND_NAME].nunique():>8}        ║")
    logger.info(f"  ║  New columns added    : {6:>8}        ║")
    logger.info(f"  ║    Log_Return, Cumulative_Return,           ║")
    logger.info(f"  ║    Rolling_3M/6M/12M, RF_Source             ║")
    logger.info(f"  ╚═════════════════════════════════════════╝")
    logger.info("\n  Return calculations complete.  Proceed to factor_merge.py.")
    logger.info("=" * 68 + "\n")

    return df, fund_summary, reg_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SELF-TEST (python src/return_calculations.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import load_all_data
    from src.preprocessing import build_master_frame

    _log = setup_logger("return_calc_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  RETURN CALCULATIONS SELF-TEST")
    _log.info("═" * 68)

    try:
        # ── Load and preprocess ───────────────────────────────────────────────
        raw    = load_all_data()
        master = build_master_frame(
            passive_df=raw["passive_nav"],
            active_df=raw["active_nav"],
            er_df=raw["expense_ratios"],
            save_intermediates=False,
        )

        # ── Enrich ───────────────────────────────────────────────────────────
        enriched, summary, reg_df = enrich_master_frame(
            master_df=master,
            rf_scalar=SYNTHETIC_RF_MONTHLY,
            rolling_windows=[3, 6, 12],
            save_outputs=True,
        )

        _log.info("\n── Assertions ──")

        # ── A1: New columns exist ─────────────────────────────────────────────
        expected_new = [
            COL_LOG_RETURN, COL_CUMULATIVE_RETURN,
            COL_ROLLING_3M, COL_ROLLING_6M, COL_ROLLING_12M,
            COL_EXCESS_RETURN, COL_RF, COL_RF_SOURCE,
        ]
        missing = [c for c in expected_new if c not in enriched.columns]
        assert not missing, f"Missing columns: {missing}"
        _log.info(f"  ✅ All {len(expected_new)} engineered columns present.")

        # ── A2: No NaN returns in enriched frame ──────────────────────────────
        assert enriched[COL_RETURN].isna().sum() == 0, "NaN returns in enriched!"
        _log.info("  ✅ No NaN Monthly_Return in enriched frame.")

        # ── A3: Cumulative return starts at 1.0 for each fund ────────────────
        first_cum = (
            enriched.sort_values(COL_DATE)
            .groupby(COL_FUND_NAME)[COL_CUMULATIVE_RETURN]
            .first()
        )
        # First cum value = (1 + first_return), not 1.0 — because we
        # already dropped the NaN first-obs row
        assert (first_cum > 0).all(), "Non-positive cumulative return!"
        _log.info(
            f"  ✅ Cumulative return positive for all funds. "
            f"Range: [{first_cum.min():.4f}, {first_cum.max():.4f}]"
        )

        # ── A4: Log return ≈ ln(1 + simple_return) ────────────────────────────
        sample = enriched[enriched[COL_RETURN].notna() & enriched[COL_LOG_RETURN].notna()].head(200)
        expected_log = np.log1p(sample[COL_RETURN])
        diff = (sample[COL_LOG_RETURN] - expected_log).abs().max()
        assert diff < 1e-10, f"Log return computation error: max diff = {diff}"
        _log.info(f"  ✅ Log return = ln(1 + simple_return) — max diff: {diff:.2e}")

        # ── A5: Excess return = Return - RF ────────────────────────────────────
        expected_exc = enriched[COL_RETURN] - enriched[COL_RF]
        exc_diff = (enriched[COL_EXCESS_RETURN] - expected_exc).abs().max()
        assert exc_diff < 1e-10, f"Excess return error: max diff = {exc_diff}"
        _log.info(f"  ✅ Excess_Return = Monthly_Return − RF — max diff: {exc_diff:.2e}")

        # ── A6: RF_Source column ──────────────────────────────────────────────
        assert enriched[COL_RF_SOURCE].nunique() == 1, "Mixed RF sources!"
        assert enriched[COL_RF_SOURCE].iloc[0] == "synthetic", "Expected synthetic RF"
        _log.info("  ✅ RF_Source = 'synthetic' (consistent across all rows).")

        # ── A7: Fund summary shape and columns ───────────────────────────────
        assert len(summary) == enriched[COL_FUND_NAME].nunique(), \
            "Fund summary row count mismatch"
        for c in [COL_CAGR, COL_ANN_RETURN, COL_ANN_VOL, COL_SHARPE, COL_MAX_DRAWDOWN]:
            assert c in summary.columns, f"Missing metric column: {c}"
        _log.info(
            f"  ✅ Fund summary: {len(summary)} funds  ×  "
            f"{len(summary.columns)} metrics."
        )

        # ── A8: Regression df — eligible funds only ────────────────────────────
        assert reg_df[COL_FUND_NAME].nunique() == 64, \
            f"Expected 64 eligible funds, got {reg_df[COL_FUND_NAME].nunique()}"
        assert reg_df[COL_RETURN].isna().sum() == 0, "NaN returns in reg_df!"
        _log.info(
            f"  ✅ Regression df: {len(reg_df):,} rows  ×  "
            f"{reg_df[COL_FUND_NAME].nunique()} eligible funds."
        )

        # ── A9: Spot-check summary stats for active vs passive ─────────────────
        active_sharpe  = summary[summary[COL_FUND_TYPE] == FUND_TYPE_ACTIVE][COL_SHARPE].mean()
        passive_sharpe = summary[summary[COL_FUND_TYPE] == FUND_TYPE_PASSIVE][COL_SHARPE].mean()
        active_er      = summary[summary[COL_FUND_TYPE] == FUND_TYPE_ACTIVE][COL_ER].mean()
        passive_er     = summary[summary[COL_FUND_TYPE] == FUND_TYPE_PASSIVE][COL_ER].mean()
        _log.info(
            f"\n  ── Preliminary Active vs Passive comparison ──"
        )
        _log.info(
            f"  {'Metric':<25}  {'Active':>10}  {'Passive':>10}"
        )
        _log.info(f"  {'-'*45}")
        _log.info(
            f"  {'Mean Sharpe Ratio':<25}  {active_sharpe:>10.4f}  {passive_sharpe:>10.4f}"
        )
        _log.info(
            f"  {'Mean CAGR':<25}  "
            f"{summary[summary[COL_FUND_TYPE]==FUND_TYPE_ACTIVE][COL_CAGR].mean():>10.2%}  "
            f"{summary[summary[COL_FUND_TYPE]==FUND_TYPE_PASSIVE][COL_CAGR].mean():>10.2%}"
        )
        _log.info(
            f"  {'Mean Annualised Vol':<25}  "
            f"{summary[summary[COL_FUND_TYPE]==FUND_TYPE_ACTIVE][COL_ANN_VOL].mean():>10.2%}  "
            f"{summary[summary[COL_FUND_TYPE]==FUND_TYPE_PASSIVE][COL_ANN_VOL].mean():>10.2%}"
        )
        _log.info(
            f"  {'Mean Expense Ratio':<25}  {active_er:>9.2f}%  {passive_er:>9.2f}%"
        )

        _log.info("\n✅  All return_calculations self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Self-test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        raise

    _log.info("═" * 68 + "\n")
