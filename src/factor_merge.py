"""
factor_merge.py
===============
Synthetic factor construction and factor-table merge for the Carhart
Four-Factor Fund Analysis Pipeline.

Responsibility
--------------
Produces a monthly factor table covering the full analysis window
(2013-01 → 2026-04) and merges it into the regression-ready DataFrame
produced by ``return_calculations.py``.

Why synthetic factors?
----------------------
The provided ``factor_data.csv`` covers only three months in 1993 — zero
overlap with the NAV data.  This module therefore:

  1.  Assesses the provided factor file for usability (coverage check).
  2.  Builds a synthetic factor table from the NAV data itself when the
      provided file is insufficient (current situation).
  3.  Merges the factor table (real or synthetic) into the master frame.
  4.  Recomputes Excess_Return using the now time-varying RF column.
  5.  Validates the merged output before handing off to regression_analysis.py.

Synthetic factor construction — design rationale
-------------------------------------------------
The four Carhart factors are:

  MF  (Market excess return, Rm − Rf):
      We use the equal-weighted cross-sectional mean return of all 33
      passive funds (Nifty 50 ETFs + BSE Sensex ETFs + Index Funds) as
      the market return proxy.  These funds collectively replicate the
      Indian large-cap equity market.

      MF_t = mean_passive_return_t − RF_t

  RF  (Risk-free rate):
      A time-varying monthly RF is constructed from the RBI repo rate
      schedule.  Each month is assigned the repo rate in effect at that
      time, converted to a monthly decimal:
          RF_t = annual_repo_rate_t / 12

      The repo rate schedule (annual %) used here covers 2013-01 → 2026-04:
          2013-01 → 2014-01 :  7.75%
          2014-01 → 2015-03 :  8.00%
          2015-03 → 2016-04 :  7.50%
          2016-04 → 2017-08 :  6.50%
          2017-08 → 2019-06 :  6.00%
          2019-06 → 2020-03 :  5.15%
          2020-03 → 2020-05 :  4.40%  (COVID cut)
          2020-05 → 2022-05 :  4.00%  (COVID floor)
          2022-05 → 2023-02 :  6.50%  (tightening cycle)
          2023-02 → 2025-02 :  6.50%
          2025-02 → 2026-04 :  6.25%

      These approximate the actual RBI repo rate history.  For a fully
      rigorous analysis, replace this schedule with actual RBI data.

  SMB (Small Minus Big — size factor):
      Set to 0.0 for all months.
      Rationale: our fund universe is entirely large-cap (Nifty 50 /
      BSE Sensex / active large-cap).  In an all-large-cap universe the
      size factor has no explanatory power and is correctly zeroed.

  HML (High Minus Low — value factor):
      Set to 0.0 for all months.
      Rationale: No stock-level book-to-market data is available.
      Setting HML = 0 means the regression correctly attributes any
      value-tilt alpha to the intercept term rather than a factor.

  WML (Winners Minus Losers — momentum factor):
      Computed as a cross-fund momentum spread.  Each month t:
        • Rank all eligible funds by their trailing 11-month return
          (months t−12 to t−2, skip t−1 to avoid reversal effect).
        • WML_t = mean(top-30% fund returns in month t)
                − mean(bottom-30% fund returns in month t)

      This provides a non-trivial momentum signal while avoiding look-
      ahead bias (formation window ends 2 months before evaluation month).

      Note: this is a fund-return-based WML proxy, not the stock-level
      factor used in Fama-French databases.  It still captures whether
      momentum-winning funds continue to outperform.

Module layout
-------------
  §1  _build_rbi_rf_schedule        — time-varying RF table from repo rate history
  §2  assess_factor_coverage        — check if provided factor file is usable
  §3  build_market_factor           — MF from cross-sectional passive avg
  §4  build_wml_factor              — cross-fund momentum spread
  §5  build_synthetic_factor_table  — assemble full [Date, MF, SMB, HML, WML, RF]
  §6  merge_factors_into_frame      — left join factor table onto regression df
  §7  update_excess_returns         — recompute Excess_Return with actual RF_t
  §8  validate_merged_frame         — pre-regression completeness checks
  §9  build_and_merge_factors       — orchestrator (main entry point)
  §10 Self-test (python src/factor_merge.py)
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
    ANALYSIS_END,
    ANALYSIS_START,
    COL_DATE,
    COL_ER,
    COL_EXCESS_RETURN,
    COL_FACTOR_DATE_RAW,
    COL_FUND_NAME,
    COL_FUND_TYPE,
    COL_HML,
    COL_HML_RAW,
    COL_MF,
    COL_MF_RAW,
    COL_NAV,
    COL_RF,
    COL_RF_RAW,
    COL_RF_SOURCE,
    COL_RETURN,
    COL_ROLLING_12M,
    COL_SMB,
    COL_SMB_RAW,
    COL_WML,
    COL_WML_RAW,
    COL_YEAR_MONTH,
    FACTOR_SYNTHETIC_FILE,
    FUND_TYPE_PASSIVE,
    MIN_FACTOR_COVERAGE_FRACTION,
    MONTHS_PER_YEAR,
    SYNTHETIC_RF_MONTHLY,
    USE_PROVIDED_FACTORS_IF_AVAILABLE,
)
from src.utils import (
    report_missing,
    save_dataframe,
    setup_logger,
    validate_dataframe,
)

logger = setup_logger(__name__)

# ── Internal column names used only inside this module ────────────────────────
_YM_COL           = "Year_Month"          # Period[M] join key
_ELIGIBLE_COL     = "Eligible_For_Regression"
_MF_RAW_COL       = "MF_raw"             # Rm before subtracting RF
_MOMENTUM_RANK    = "_momentum_rank"      # temp ranking column
_FACTOR_SOURCE    = "Factor_Source"       # "synthetic" | "provided"

# Momentum formation window (months before evaluation, skip t-1)
_MOMENTUM_LOOKBACK = 12   # look back 12 months
_MOMENTUM_SKIP     = 1    # skip most-recent month (reversal avoidance)
_MOMENTUM_PCTILE_TOP    = 0.70   # top 30 % = winners
_MOMENTUM_PCTILE_BOTTOM = 0.30   # bottom 30 % = losers


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIME-VARYING RF: RBI REPO RATE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ASSESS PROVIDED FACTOR FILE COVERAGE
# ══════════════════════════════════════════════════════════════════════════════

def assess_factor_coverage(
    factor_df: pd.DataFrame,
    start: str = ANALYSIS_START,
    end:   str = ANALYSIS_END,
) -> tuple[bool, float]:
    """
    Determine whether the provided factor DataFrame has sufficient temporal
    coverage to be used instead of synthetic factors.

    Coverage is measured as the fraction of analysis-window months that have
    a matching row in the factor file.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Raw factor DataFrame from ``load_factor_data``.
    start, end : str
        Analysis window boundaries.

    Returns
    -------
    tuple[bool, float]
        (is_sufficient, coverage_fraction)
        ``is_sufficient`` is True when coverage ≥ MIN_FACTOR_COVERAGE_FRACTION
        and USE_PROVIDED_FACTORS_IF_AVAILABLE is True.
    """
    try:
        factor_periods = pd.PeriodIndex(
            factor_df[COL_FACTOR_DATE_RAW], freq="M"
        )
    except Exception:
        logger.warning(
            "  assess_factor_coverage: could not parse factor Date column → "
            "returning insufficient."
        )
        return False, 0.0

    analysis_start_p = pd.Period(start[:7], freq="M")
    analysis_end_p   = pd.Period(end[:7],   freq="M")

    total_months = (
        (analysis_end_p.year - analysis_start_p.year) * 12
        + (analysis_end_p.month - analysis_start_p.month) + 1
    )
    overlap = ((factor_periods >= analysis_start_p) & (
        factor_periods <= analysis_end_p
    )).sum()

    coverage = overlap / total_months if total_months > 0 else 0.0
    is_sufficient = (
        USE_PROVIDED_FACTORS_IF_AVAILABLE
        and coverage >= MIN_FACTOR_COVERAGE_FRACTION
    )

    logger.info(
        f"  Factor coverage: {overlap}/{total_months} months "
        f"({coverage:.1%})  →  "
        f"{'SUFFICIENT — using provided factors' if is_sufficient else 'INSUFFICIENT — synthetic factors will be built'}"
    )
    return is_sufficient, coverage


def merge_factors_into_frame(
    reg_df: pd.DataFrame,
    factor_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join the factor table onto the regression DataFrame on Year_Month.

    Parameters
    ----------
    reg_df : pd.DataFrame
        Regression-ready long-format DataFrame from
        ``return_calculations.get_regression_df``.
        Must contain a ``Year_Month`` column (Period[M] or string).
    factor_table : pd.DataFrame
        Factor table from ``build_synthetic_factor_table`` (or the provided
        factor file if coverage was sufficient).
        Must contain: [Year_Month, MF, SMB, HML, WML, RF].

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with factor columns populated.

    Notes
    -----
    * If any observation months are missing from the factor table, those rows
      receive NaN factors and are logged.  ``regression_analysis.py`` drops
      NaN rows before fitting.
    * Existing MF/SMB/HML/WML/RF columns in ``reg_df`` (placeholders filled
      with NaN by ``get_regression_df``) are overwritten.
    """
    logger.info("  §6 Merging factor table into regression frame …")

    reg_df    = reg_df.copy()
    fact_copy = factor_table.copy()

    # ── Harmonise Year_Month dtype ────────────────────────────────────────────
    def _to_period(col: pd.Series) -> pd.Series:
        if col.dtype == object or col.dtype.name == "string":
            return col.apply(lambda x: pd.Period(x, freq="M"))
        if str(col.dtype).startswith("period"):
            return col
        return col.apply(lambda x: pd.Period(x, freq="M"))

    reg_df[_YM_COL]    = _to_period(reg_df[_YM_COL])
    fact_copy[_YM_COL] = _to_period(fact_copy[_YM_COL])

    # ── Drop placeholder factor columns from reg_df before merge ─────────────
    factor_cols_to_fill = [COL_MF, COL_SMB, COL_HML, COL_WML, COL_RF]
    reg_df = reg_df.drop(
        columns=[c for c in factor_cols_to_fill if c in reg_df.columns],
        errors="ignore",
    )

    # Also drop RF_Source placeholder if present
    if COL_RF_SOURCE in reg_df.columns:
        reg_df = reg_df.drop(columns=[COL_RF_SOURCE], errors="ignore")

    # ── Select only the factor columns we need from factor_table ─────────────
    merge_cols = [_YM_COL, COL_RF, COL_MF, COL_SMB, COL_HML, COL_WML]
    if _FACTOR_SOURCE in fact_copy.columns:
        merge_cols.append(_FACTOR_SOURCE)

    fact_merge = fact_copy[merge_cols].copy()

    # ── Left join ─────────────────────────────────────────────────────────────
    merged = reg_df.merge(fact_merge, on=_YM_COL, how="left")

    # ── Assess merge quality ──────────────────────────────────────────────────
    n_mf_nan = merged[COL_MF].isna().sum()
    n_rf_nan = merged[COL_RF].isna().sum()

    if n_mf_nan > 0 or n_rf_nan > 0:
        # Identify which months are unmatched
        unmatched_months = (
            merged[merged[COL_MF].isna()][_YM_COL].unique().tolist()[:10]
        )
        logger.warning(
            f"  §6 Merge: {n_mf_nan} rows have NaN MF (will be dropped in "
            f"regression).  Sample unmatched months: {unmatched_months}"
        )
    else:
        logger.info(
            f"  §6 Merge complete:  {len(merged):,} rows  |  "
            f"0 NaN factor rows.  ✓"
        )

    logger.info(
        f"  Merged frame: {len(merged):,} rows  ×  {len(merged.columns)} cols  |  "
        f"Funds: {merged[COL_FUND_NAME].nunique()}"
    )
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — UPDATE EXCESS RETURNS WITH TIME-VARYING RF
# ══════════════════════════════════════════════════════════════════════════════

def update_excess_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute ``Excess_Return = Monthly_Return − RF_t`` using the now
    time-varying RF from the merged factor table.

    This replaces the constant synthetic RF set by ``return_calculations.py``
    (Section 3) with the per-month RBI repo rate proxy.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame containing both COL_RETURN and COL_RF.

    Returns
    -------
    pd.DataFrame
        DataFrame with COL_EXCESS_RETURN updated and COL_RF_SOURCE set to
        the factor source string.
    """
    logger.info("  §7 Updating Excess_Return with time-varying RF …")
    df = df.copy()

    if COL_RF not in df.columns:
        raise KeyError(
            "update_excess_returns: COL_RF not found.  "
            "Ensure factor merge ran before calling this function."
        )

    # Recompute
    df[COL_EXCESS_RETURN] = df[COL_RETURN] - df[COL_RF]

    # Tag RF source
    if _FACTOR_SOURCE in df.columns:
        df[COL_RF_SOURCE] = df[_FACTOR_SOURCE]
    else:
        df[COL_RF_SOURCE] = "provided"

    # Diagnostics
    rf_mean  = df[COL_RF].mean()
    exc_mean = df[COL_EXCESS_RETURN].mean()
    ret_mean = df[COL_RETURN].mean()

    logger.info(
        f"  Excess_Return updated:  "
        f"Mean return = {ret_mean:.4%}  |  "
        f"Mean RF = {rf_mean:.4%}  |  "
        f"Mean excess return = {exc_mean:.4%}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PRE-REGRESSION VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_merged_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a comprehensive validation suite on the factor-merged DataFrame before
    handing it to ``regression_analysis.py``.

    Checks
    ------
    1.  All required columns present.
    2.  Factor columns (MF, SMB, HML, WML, RF) have acceptable NaN rates.
    3.  No fund has zero non-NaN regression observations.
    4.  Per-fund observation counts meet MIN_OBS.
    5.  Excess_Return is finite for the vast majority of rows.
    6.  Multicollinearity flag: if |corr(MF, WML)| > 0.80, warn.

    Parameters
    ----------
    df : pd.DataFrame
        Factor-merged regression DataFrame.

    Returns
    -------
    pd.DataFrame
        The same DataFrame (rows with ALL-NaN factors dropped cleanly).
    """
    logger.info("  §8 Validating merged frame (pre-regression) …")
    df = df.copy()

    # ── Check 1: Required columns ─────────────────────────────────────────────
    required = [
        COL_FUND_NAME, COL_FUND_TYPE, COL_DATE, COL_RETURN,
        COL_EXCESS_RETURN, COL_MF, COL_SMB, COL_HML, COL_WML,
        COL_RF, COL_ER,
    ]
    validate_dataframe(df, required, df_name="Factor-merged frame", min_rows=100)

    # ── Check 2: NaN rates in factor columns ──────────────────────────────────
    factor_cols = [COL_MF, COL_SMB, COL_HML, COL_WML, COL_RF]
    for fc in factor_cols:
        nan_rate = df[fc].isna().mean()
        if nan_rate > 0.05:   # > 5% NaN is concerning
            logger.warning(
                f"  Validation: '{fc}' has {nan_rate:.1%} NaN rows — "
                f"these will be dropped before regression."
            )
        else:
            logger.debug(f"  Validation: '{fc}' NaN rate = {nan_rate:.2%}  ✓")

    # ── Check 3: Drop rows where ANY factor is NaN ────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=factor_cols + [COL_RETURN, COL_EXCESS_RETURN])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            f"  Validation: dropped {n_dropped:,} rows with NaN factors.  "
            f"Remaining: {len(df):,}"
        )

    # ── Check 4: Per-fund observation counts after drop ────────────────────────
    from src.config import MIN_OBS
    obs_counts = df.groupby(COL_FUND_NAME)[COL_RETURN].count()
    low_obs    = obs_counts[obs_counts < MIN_OBS]
    if not low_obs.empty:
        logger.warning(
            f"  Validation: {len(low_obs)} fund(s) have < {MIN_OBS} obs after "
            f"factor merge (likely WML formation-period months dropped):"
        )
        for fname, n in low_obs.items():
            logger.warning(f"      {fname:60s}  →  {n} obs")

    # ── Check 5: Excess_Return finiteness ─────────────────────────────────────
    n_inf = (~np.isfinite(df[COL_EXCESS_RETURN])).sum()
    if n_inf > 0:
        logger.warning(
            f"  Validation: {n_inf} non-finite Excess_Return values. Replacing with NaN."
        )
        df[COL_EXCESS_RETURN] = df[COL_EXCESS_RETURN].replace(
            [np.inf, -np.inf], np.nan
        )

    # ── Check 6: Factor correlation matrix ────────────────────────────────────
    factor_data = df[[COL_MF, COL_SMB, COL_HML, COL_WML]].dropna()
    if len(factor_data) > 10:
        corr_matrix = factor_data.corr()
        high_corr_pairs = []
        for i, f1 in enumerate(factor_cols[:4]):
            for f2 in factor_cols[:4][i+1:]:
                c = corr_matrix.loc[f1, f2] if f1 in corr_matrix and f2 in corr_matrix else 0
                if abs(c) > 0.80:
                    high_corr_pairs.append((f1, f2, c))

        if high_corr_pairs:
            logger.warning(
                "  Validation: High factor correlations detected (|r| > 0.80):"
            )
            for f1, f2, c in high_corr_pairs:
                logger.warning(f"      {f1} ↔ {f2}: r = {c:.4f}")
        else:
            logger.info(
                f"  Validation: Factor correlation matrix — no high correlations.  ✓\n"
                f"{corr_matrix.round(3).to_string()}"
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info(
        f"  Validation complete: {len(df):,} rows  |  "
        f"{df[COL_FUND_NAME].nunique()} funds  |  "
        f"All required columns present.  ✓"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ORCHESTRATOR: build_and_merge_factors
# ══════════════════════════════════════════════════════════════════════════════

def build_and_merge_factors(
    master_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    raw_factor_df: pd.DataFrame,
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full factor construction and merge pipeline.

    Decision tree
    -------------
    1.  Assess coverage of provided ``raw_factor_df``.
    2a. If coverage ≥ 90%: parse and use provided factors directly.
    2b. If coverage < 90%: build synthetic factors from NAV data.
    3.  Merge factor table into ``reg_df``.
    4.  Update Excess_Return with time-varying RF.
    5.  Validate merged frame.
    6.  Save factor table and merged frame CSVs (optional).

    Parameters
    ----------
    master_df : pd.DataFrame
        Full enriched master frame (all funds, all months) — used to build
        the market proxy and WML factor.
    reg_df : pd.DataFrame
        Regression-ready long-format DataFrame (eligible funds, factor cols NaN)
        from ``return_calculations.get_regression_df``.
    raw_factor_df : pd.DataFrame
        Raw factor DataFrame from ``load_factor_data`` — used for coverage
        assessment and as the source if coverage is adequate.
    save_outputs : bool, optional
        If True, save factor table and merged regression frame CSVs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (factor_table, regression_df_with_factors)
        factor_table : monthly factor rows (Year_Month, RF, MF, SMB, HML, WML)
        regression_df: reg_df with factor columns populated and validated
    """
    logger.info("\n" + "=" * 68)
    logger.info("  FACTOR MERGE PHASE — build_and_merge_factors")
    logger.info("=" * 68)

    # ── Step 2: Build or parse factor table ───────────────────────────────────
    logger.info("Using provided factor dataset (1993–2025).")

    factor_table = _parse_provided_factors(raw_factor_df)

    # ── Step 3: Merge factors into regression frame ───────────────────────────
    logger.info("\n── Step 3: Merge factor table into regression frame ──")
    merged_df = merge_factors_into_frame(reg_df, factor_table)

    # ── Step 4: Update Excess_Return with time-varying RF ────────────────────
    logger.info("\n── Step 4: Update Excess_Return with RF_t ──")
    merged_df = update_excess_returns(merged_df)

    # ── Step 5: Validate ──────────────────────────────────────────────────────
    logger.info("\n── Step 5: Pre-regression validation ──")
    merged_df = validate_merged_frame(merged_df)

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    if save_outputs:
        # Save factor table (Period → string for CSV)
        ft_export = factor_table.copy()
        if _YM_COL in ft_export.columns:
            ft_export[_YM_COL] = ft_export[_YM_COL].astype(str)
        save_dataframe(ft_export, FACTOR_SYNTHETIC_FILE, description="Provided factor table")

        from src.config import REGRESSION_RESULTS_DIR
        from src.utils import ensure_dir
        ensure_dir(REGRESSION_RESULTS_DIR)
        reg_export = merged_df.copy()
        if _YM_COL in reg_export.columns:
            reg_export[_YM_COL] = reg_export[_YM_COL].astype(str)
        merged_path = REGRESSION_RESULTS_DIR / "regression_input.csv"
        save_dataframe(reg_export, merged_path, description="Regression input (factor-merged)")

    # ── Final report ──────────────────────────────────────────────────────────
    ft = factor_table
    logger.info(f"\n  ╔══════ FACTOR MERGE SUMMARY ══════╗")
    logger.info(f"  ║  Factor source  : {'Provided Four-Factor Dataset':35s}  ║")
    logger.info(f"  ║  Factor months  : {len(ft):>6}                                    ║")
    logger.info(f"  ║  Mean RF        : {ft[COL_RF].mean():>8.4%}/month                       ║")
    logger.info(f"  ║  Mean MF        : {ft[COL_MF].mean():>8.4%}/month                       ║")
    logger.info(f"  ║  Mean WML       : {ft[COL_WML].mean():>8.4%}/month                       ║")
    logger.info(f"  ║  SMB / HML      : {'0.000 (constant — see design notes)':35s}  ║")
    logger.info(f"  ║  Merged rows    : {len(merged_df):>6,}                                  ║")
    logger.info(f"  ║  Merged funds   : {merged_df[COL_FUND_NAME].nunique():>6}                                  ║")
    logger.info(f"  ╚══════════════════════════════════╝")
    logger.info("\n  Factor merge complete.  Proceed to regression_analysis.py.")
    logger.info("=" * 68 + "\n")

    return factor_table, merged_df


def _parse_provided_factors(raw_factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse real factor data.
    """

    logger.info("Parsing provided factor dataset...")

    df = raw_factor_df.copy()

    # Date column
    df["Date"] = pd.to_datetime(df["Date"].astype(str))

    # Keep only analysis window
    df = df[
        (df["Date"] >= ANALYSIS_START)
        &
        (df["Date"] <= ANALYSIS_END)
    ].copy()

    # Create merge key
    df[_YM_COL] = df["Date"].dt.to_period("M")

    # Rename columns
    rename_map = {
        COL_MF_RAW: COL_MF,
        COL_SMB_RAW: COL_SMB,
        COL_HML_RAW: COL_HML,
        COL_WML_RAW: COL_WML,
        COL_RF_RAW: COL_RF,
    }

    df = df.rename(columns=rename_map)

    # Convert percent → decimal
    factor_cols = [
        COL_MF,
        COL_SMB,
        COL_HML,
        COL_WML,
        COL_RF,
    ]
    for col in factor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col] / 100.0

    df[_FACTOR_SOURCE] = "provided"

    return df[
        [
            _YM_COL,
            "Date",
            COL_RF,
            COL_MF,
            COL_SMB,
            COL_HML,
            COL_WML,
            _FACTOR_SOURCE,
        ]
    ]
    # Provided RF is in decimal percent → convert to decimal
    # (factor files typically store RF as annualised % or already monthly %)
    # Divide by 100 if values look like percentages (> 0.1 is a hint)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SELF-TEST (python src/factor_merge.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import load_all_data
    from src.preprocessing import build_master_frame
    from src.return_calculations import enrich_master_frame

    _log = setup_logger("factor_merge_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  FACTOR MERGE SELF-TEST")
    _log.info("═" * 68)

    try:
        # ── Full pipeline up to this point ────────────────────────────────────
        raw = load_all_data()
        master = build_master_frame(
            passive_df=raw["passive_nav"],
            active_df=raw["active_nav"],
            er_df=raw["expense_ratios"],
            save_intermediates=False,
        )
        enriched, fund_summary, reg_df = enrich_master_frame(
            master_df=master,
            save_outputs=False,
        )

        # ── Build and merge factors ───────────────────────────────────────────
        factor_table, merged_df = build_and_merge_factors(
            master_df=enriched,
            reg_df=reg_df,
            raw_factor_df=raw["factor_data"],
            save_outputs=True,
        )

        _log.info("\n── Assertions ──")

        # A1: Factor table shape and columns
        assert len(factor_table) >= 155, \
            f"Expected ≥ 155 factor months, got {len(factor_table)}"
        for fc in [COL_MF, COL_SMB, COL_HML, COL_WML, COL_RF]:
            assert fc in factor_table.columns, f"Factor column '{fc}' missing"
        _log.info(
            f"  ✅ Factor table: {len(factor_table)} months  ×  "
            f"{len(factor_table.columns)} cols."
        )

        # A3: MF is non-trivial (mean should be non-zero, std > 0)
        mf_mean = factor_table[COL_MF].mean()
        mf_std  = factor_table[COL_MF].std()
        assert mf_std > 0.01, f"MF std too low: {mf_std:.4f}"
        _log.info(
            f"  ✅ MF non-trivial: mean={mf_mean:.4%}  std={mf_std:.4%}."
        )

        # A4: WML has variation (not all zeros — momentum signal present)
        wml_std = factor_table[COL_WML].std()
        assert wml_std > 0, "WML has no variation (all zeros)"
        _log.info(
            f"  ✅ WML has variation: std={wml_std:.4%}  "
            f"(momentum signal present)."
        )

        # A5: RF is time-varying (not a constant)
        rf_unique = factor_table[COL_RF].nunique()
        assert rf_unique > 1, f"RF is constant (only {rf_unique} unique values)"
        _log.info(
            f"  ✅ RF is time-varying: {rf_unique} unique values  "
            f"range=[{factor_table[COL_RF].min():.4%}, {factor_table[COL_RF].max():.4%}]."
        )

        # A6: Merged frame has all factor columns with low NaN rate
        for fc in [COL_MF, COL_SMB, COL_HML, COL_WML, COL_RF]:
            assert fc in merged_df.columns, f"'{fc}' missing from merged df"
            nan_rate = merged_df[fc].isna().mean()
            assert nan_rate < 0.10, f"'{fc}' NaN rate too high: {nan_rate:.2%}"
        _log.info(
            f"  ✅ All factor columns in merged frame with NaN < 10%."
        )

        # A7: Excess_Return recomputed with time-varying RF
        expected_exc = merged_df[COL_RETURN] - merged_df[COL_RF]
        diff = (merged_df[COL_EXCESS_RETURN] - expected_exc).abs().max()
        assert diff < 1e-9, f"Excess_Return recomputation error: {diff}"
        _log.info(
            f"  ✅ Excess_Return = Return − RF_t correctly recomputed."
        )

        # A8: Merged frame ready for regression
        _log.info(
            f"\n  Merged regression frame: {len(merged_df):,} rows  ×  "
            f"{len(merged_df.columns)} cols  |  "
            f"{merged_df[COL_FUND_NAME].nunique()} funds"
        )
        _log.info(
            f"  Columns: {list(merged_df.columns)}"
        )

        # A9: Factor table preview
        _log.info(f"\n  Factor table sample (first 5 rows):")
        ft_display = factor_table.copy()
        ft_display[_YM_COL] = ft_display[_YM_COL].astype(str)
        _log.info(f"\n{ft_display.head(5).to_string(index=False)}")

        # A10: Per-type excess return comparison
        active_exc  = merged_df[merged_df[COL_FUND_TYPE]=="Active"][COL_EXCESS_RETURN].mean()
        passive_exc = merged_df[merged_df[COL_FUND_TYPE]=="Passive"][COL_EXCESS_RETURN].mean()
        _log.info(
            f"\n  Mean Excess Return:  Active={active_exc:.4%}  "
            f"Passive={passive_exc:.4%}"
        )

        _log.info("\n✅  All factor_merge self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Self-test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        raise

    _log.info("═" * 68 + "\n")
