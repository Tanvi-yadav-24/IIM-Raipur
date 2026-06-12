"""
preprocessing.py
================
Data cleaning, standardisation, and master-frame construction for the
Carhart Four-Factor Fund Analysis Pipeline.

Responsibility
--------------
Receives the four raw DataFrames produced by ``data_loader.py`` and transforms
them into a single, analysis-ready master DataFrame.  Specifically:

  1.  Assign FUND_TYPE labels         ("Active" / "Passive")
  2.  Parse Date columns              (str → datetime64)
  3.  Extract Year_Month              (datetime → Period[M], for factor merge)
  4.  Normalise fund names            (add Fund_Name_Norm merge key)
  5.  Standardise & validate returns  (recompute if discrepant, drop first-obs NaN)
  6.  Winsorise extreme returns       (cap at [-50%, +100%])
  7.  Validate monthly continuity     (detect gaps, flag duplicates)
  8.  Concatenate passive + active    (union of FILE 1 and FILE 4)
  9.  Filter to analysis window       (2013-01-01 → 2026-04-30)
  10. Merge expense ratios            (normalised-name left join, fuzzy fallback)
  11. Validate final master frame     (column completeness, row counts, NaN checks)
  12. Save intermediate cleaned files (cleaned_data/)

Ground-truth reminders (from inspection — DO NOT REMOVE)
---------------------------------------------------------
* FILE 1 Month column: full English names ("July") — IGNORED; Date used instead.
* FILE 4 Month column: YYYY-MM ("2013-01")         — IGNORED; Date used instead.
* Each fund has exactly 1 NaN Monthly_Return (its very first observation).
* Passive outlier: one return of -90.38% (likely data/corporate-action error).
* 2 active funds below MIN_OBS=36: Bajaj Finserv (21m), Motilal Oswal (27m).
* 4 active rows beyond 2026-04-30 (Axis, Invesco, Mahindra, PGIM → May 2026).
* 3 ER funds have no NAV match: Parag Parikh, Samco, Franklin India (partial).

Module layout
-------------
  1. _preprocess_nav_frame        — per-file cleaning (common to both files)
  2. assign_fund_type             — label Active / Passive
  3. parse_and_enrich_dates       — Date → datetime, Year_Month Period
  4. validate_returns             — cross-check pre-computed vs NAV-derived
  5. standardise_returns          — drop first-obs NaN, winsorise extremes
  6. validate_monthly_continuity  — gap detection, duplicate (fund, month) check
  7. merge_expense_ratios         — normalised-name join with fuzzy fallback
  8. build_master_frame           — orchestrates all steps, returns master df
  9. Self-test  (python src/preprocessing.py)
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
    ACTIVE_CLEAN_FILE,
    ANALYSIS_END,
    ANALYSIS_START,
    COL_DATE,
    COL_DATE_RAW,
    COL_ER,
    COL_ER_FUND_RAW,
    COL_ER_VALUE_RAW,
    COL_EXCESS_RETURN,
    COL_FUND_NAME,
    COL_FUND_NAME_NORM,
    COL_FUND_NAME_RAW,
    COL_FUND_TYPE,
    COL_MONTH_RAW,
    COL_NAV,
    COL_RETURN,
    COL_RETURN_PCT_RAW,
    COL_RETURN_RAW,
    COL_YEAR_MONTH,
    FUND_TYPE_ACTIVE,
    FUND_TYPE_PASSIVE,
    MASTER_DATA_FILE,
    MIN_OBS,
    PASSIVE_CLEAN_FILE,
    SYNTHETIC_RF_MONTHLY,
)
from src.utils import (
    build_norm_column,
    coerce_numeric_columns,
    detect_outliers_iqr,
    filter_date_range,
    extract_year_month,
    find_best_match,
    normalize_fund_name,
    parse_date_column,
    report_duplicates,
    report_missing,
    save_dataframe,
    setup_logger,
    validate_dataframe,
    winsorize_returns,
)

logger = setup_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PER-FILE CLEANING (COMMON TO BOTH NAV FILES)
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_nav_frame(
    df: pd.DataFrame,
    fund_type: str,
    df_name: str,
) -> pd.DataFrame:
    """
    Apply all cleaning steps that are identical for both passive and active
    NAV files.  Returns a cleaned copy with standardised column names.

    Steps (in order)
    ----------------
    1.  Drop the ``Month`` column (inconsistent format; Date is authoritative).
    2.  Rename ``Fund_Name`` to standardised ``COL_FUND_NAME`` (no-op currently
        but guards against upstream renames).
    3.  Add ``COL_FUND_TYPE`` label.
    4.  Parse ``Date`` string → datetime64.
    5.  Extract ``Year_Month`` Period[M].
    6.  Coerce numeric columns.
    7.  Add ``Fund_Name_Norm`` normalised key.
    8.  Drop complete duplicates.
    9.  Sort by (Fund_Name, Date).

    Parameters
    ----------
    df : pd.DataFrame
        Raw NAV DataFrame as returned by a data_loader function.
    fund_type : str
        ``"Active"`` or ``"Passive"`` — applied to every row.
    df_name : str
        Human-readable label for log messages.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    logger.info(f"  Preprocessing [{df_name}]  ({len(df):,} rows, {fund_type})")
    df = df.copy()

    # ── Step 1: Drop Month column (format differs across files; Date is canon) ─
    if COL_MONTH_RAW in df.columns:
        df = df.drop(columns=[COL_MONTH_RAW])
        logger.debug(f"  [{df_name}] Dropped '{COL_MONTH_RAW}' column.")

    # ── Step 2: Rename raw column to pipeline standard ────────────────────────
    if COL_FUND_NAME_RAW in df.columns and COL_FUND_NAME_RAW != COL_FUND_NAME:
        df = df.rename(columns={COL_FUND_NAME_RAW: COL_FUND_NAME})

    if COL_RETURN_RAW in df.columns and COL_RETURN_RAW != COL_RETURN:
        df = df.rename(columns={COL_RETURN_RAW: COL_RETURN})

    # Drop the redundant percentage column (keep decimal form only)
    if COL_RETURN_PCT_RAW in df.columns:
        df = df.drop(columns=[COL_RETURN_PCT_RAW])
        logger.debug(f"  [{df_name}] Dropped '{COL_RETURN_PCT_RAW}' (redundant).")

    # ── Step 3: Assign fund type label ────────────────────────────────────────
    df[COL_FUND_TYPE] = fund_type
    logger.debug(f"  [{df_name}] FUND_TYPE = '{fund_type}' assigned to all rows.")

    # ── Step 4: Parse Date ────────────────────────────────────────────────────
    df = parse_date_column(df, col=COL_DATE_RAW, df_name=df_name)
    df = df.rename(columns={COL_DATE_RAW: COL_DATE})

    # Drop rows where Date could not be parsed (NaT)
    n_nat = df[COL_DATE].isna().sum()
    if n_nat > 0:
        logger.warning(
            f"  [{df_name}] Dropping {n_nat} rows with unparseable dates (NaT)."
        )
        df = df.dropna(subset=[COL_DATE])

    # ── Step 5: Extract Year_Month Period ─────────────────────────────────────
    df = extract_year_month(df, date_col=COL_DATE, ym_col=COL_YEAR_MONTH)

    # ── Step 6: Coerce numeric columns ────────────────────────────────────────
    df = coerce_numeric_columns(df, [COL_NAV, COL_RETURN], df_name=df_name)

    # ── Step 7: Add normalised fund name ──────────────────────────────────────
    df = build_norm_column(df, raw_col=COL_FUND_NAME, norm_col=COL_FUND_NAME_NORM)

    # ── Step 8: Drop complete row duplicates ──────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            f"  [{df_name}] Dropped {n_dropped} complete duplicate rows."
        )

    # ── Step 9: Sort ──────────────────────────────────────────────────────────
    df = df.sort_values([COL_FUND_NAME, COL_DATE]).reset_index(drop=True)

    logger.info(
        f"  [{df_name}] After basic cleaning: {len(df):,} rows  |  "
        f"{df[COL_FUND_NAME].nunique()} funds"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ASSIGN FUND TYPE (public wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def assign_fund_type(
    passive_df: pd.DataFrame,
    active_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign ``FUND_TYPE`` labels and run per-file cleaning via
    ``_preprocess_nav_frame``.

    Parameters
    ----------
    passive_df : pd.DataFrame
        Raw passive NAV DataFrame from ``load_passive_nav``.
    active_df : pd.DataFrame
        Raw active NAV DataFrame from ``load_active_nav``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (cleaned_passive, cleaned_active) — both with FUND_TYPE assigned.
    """
    logger.info("\n── Step: Assign FUND_TYPE & basic cleaning ──")

    cleaned_passive = _preprocess_nav_frame(
        passive_df,
        fund_type=FUND_TYPE_PASSIVE,
        df_name="Passive NAV",
    )
    cleaned_active = _preprocess_nav_frame(
        active_df,
        fund_type=FUND_TYPE_ACTIVE,
        df_name="Active NAV",
    )

    logger.info(
        f"  FUND_TYPE assigned:  "
        f"{FUND_TYPE_PASSIVE}={cleaned_passive[COL_FUND_NAME].nunique()} funds  |  "
        f"{FUND_TYPE_ACTIVE}={cleaned_active[COL_FUND_NAME].nunique()} funds"
    )
    return cleaned_passive, cleaned_active


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PARSE & ENRICH DATES (already done inside _preprocess_nav_frame;
#             this function is exposed for testing / external use)
# ══════════════════════════════════════════════════════════════════════════════

def parse_and_enrich_dates(df: pd.DataFrame, df_name: str = "DataFrame") -> pd.DataFrame:
    """
    Ensure the Date column is datetime64 and Year_Month Period[M] exists.
    Safe to call even if the frame was already processed by
    ``_preprocess_nav_frame`` (idempotent).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain a ``Date`` column.
    df_name : str, optional
        Label for log messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with guaranteed datetime ``Date`` and ``Year_Month``.
    """
    # Parse Date if still string
    if df[COL_DATE].dtype == object:
        df = parse_date_column(df, col=COL_DATE, df_name=df_name)

    # Add Year_Month if not already present
    if COL_YEAR_MONTH not in df.columns:
        df = extract_year_month(df, date_col=COL_DATE, ym_col=COL_YEAR_MONTH)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RETURN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_returns(df: pd.DataFrame, df_name: str = "DataFrame") -> pd.DataFrame:
    """
    Cross-validate pre-computed monthly returns against NAV-derived returns and
    flag significant discrepancies.

    The file already contains ``Monthly_Return`` computed as:
        R_t = (NAV_t - NAV_{t-1}) / NAV_{t-1}

    We recompute this from the NAV column within each fund group and compare.
    If the mean absolute difference exceeds 0.001 (0.1%), the pre-computed
    column is replaced by the freshly derived series and the event is logged.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with ``Fund_Name``, ``Date``, ``NAV``, and
        ``Monthly_Return`` columns.
    df_name : str, optional
        Label for log messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with a validated (or freshly recomputed) ``Monthly_Return``
        column.

    Notes
    -----
    * Validation is done per-fund to avoid cross-fund contamination at fund
      boundaries (the last NAV of one fund should NOT be compared to the first
      NAV of the next).
    * The tolerance of 0.001 allows for minor floating-point rounding
      differences in the pre-computed values.
    """
    logger.info(f"\n── Step: Validate returns [{df_name}] ──")

    TOLERANCE = 0.001   # 0.1% mean absolute difference threshold
    discrepant_funds: list[str] = []

    df = df.copy()
    computed_return = df.groupby(COL_FUND_NAME, sort=False)[COL_NAV].transform(
        lambda s: s.pct_change()
    )

    # Compare where both are non-NaN
    both_valid = df[COL_RETURN].notna() & computed_return.notna()
    if both_valid.sum() == 0:
        logger.warning(
            f"  [{df_name}] No valid return pairs for cross-validation."
        )
        return df

    diff = (df.loc[both_valid, COL_RETURN] - computed_return[both_valid]).abs()
    mean_diff = diff.mean()

    if mean_diff > TOLERANCE:
        logger.warning(
            f"  [{df_name}] Mean |pre-computed − NAV-derived| = {mean_diff:.6f} "
            f"> tolerance {TOLERANCE}.  Replacing with NAV-derived returns."
        )
        df[COL_RETURN] = computed_return
        # Per-fund diagnosis
        per_fund_diff = (
            df.assign(_diff=abs(df[COL_RETURN] - computed_return))
            .groupby(COL_FUND_NAME)["_diff"]
            .mean()
        )
        for fname, fdiff in per_fund_diff[per_fund_diff > TOLERANCE].items():
            discrepant_funds.append(fname)
            logger.warning(
                f"      {fname[:60]}  →  mean abs diff = {fdiff:.6f}"
            )
    else:
        logger.info(
            f"  [{df_name}] Return cross-validation passed.  "
            f"Mean |diff| = {mean_diff:.8f} (< {TOLERANCE})"
        )

    if discrepant_funds:
        logger.warning(
            f"  [{df_name}] {len(discrepant_funds)} funds had discrepant returns "
            f"(now replaced with NAV-derived)."
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STANDARDISE RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def standardise_returns(
    df: pd.DataFrame,
    df_name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Drop first-observation NaN returns, winsorise extreme values, flag
    persistent outliers via IQR, and add the ``Excess_Return`` column.

    Steps
    -----
    1.  Drop rows where ``Monthly_Return`` is NaN (always the first row per
        fund, expected by construction).
    2.  Winsorise returns to [-50%, +100%] using ``config.RETURN_LOWER_BOUND``
        / ``config.RETURN_UPPER_BOUND``.
    3.  IQR outlier detection (3 × IQR) — additional logging only; these
        rows are NOT dropped (winsorisation already handled extreme values).
    4.  Compute ``Excess_Return = Monthly_Return - RF_proxy`` where RF_proxy
        is the synthetic monthly risk-free rate (``SYNTHETIC_RF_MONTHLY``).
        This column is updated later in ``factor_merge.py`` when real RF data
        is available, but a proxy value ensures the column always exists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Monthly_Return`` column.
    df_name : str, optional
        Label for log messages.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``Excess_Return`` column added.

    Notes
    -----
    The -90.38% return observed in passive fund FILE 1 is almost certainly a
    data/corporate-action artefact.  Winsorisation caps it at -50%.
    """
    logger.info(f"\n── Step: Standardise returns [{df_name}] ──")
    df = df.copy()

    # ── Drop NaN returns (first observation per fund) ─────────────────────────
    n_before = len(df)
    df = df.dropna(subset=[COL_RETURN])
    n_nan_dropped = n_before - len(df)
    logger.info(
        f"  [{df_name}] Dropped {n_nan_dropped} NaN returns "
        f"(expected: 1 per fund = {df[COL_FUND_NAME].nunique()} total).  "
        f"Remaining: {len(df):,} rows."
    )
    if n_nan_dropped != df[COL_FUND_NAME].nunique():
        logger.warning(
            f"  [{df_name}] Expected exactly 1 NaN per fund "
            f"({df[COL_FUND_NAME].nunique()} funds) but dropped {n_nan_dropped}. "
            f"Investigate if this is significantly different."
        )

    # ── Winsorise ─────────────────────────────────────────────────────────────
    df[COL_RETURN] = winsorize_returns(
        df[COL_RETURN], col_name=f"{df_name}.{COL_RETURN}"
    )

    # ── IQR outlier detection (informational only) ────────────────────────────
    outlier_mask = detect_outliers_iqr(
        df[COL_RETURN], multiplier=3.0, col_name=f"{df_name}.{COL_RETURN}"
    )
    if outlier_mask.sum() > 0:
        outlier_funds = df.loc[outlier_mask, COL_FUND_NAME].value_counts()
        logger.warning(
            f"  [{df_name}] IQR outlier counts by fund (rows retained, "
            f"already winsorised):\n{outlier_funds.to_string()}"
        )

    # ── Compute proxy Excess Return ───────────────────────────────────────────
    # Will be overwritten by factor_merge.py with actual RF; this ensures
    # the column always exists in the master frame.
    df[COL_EXCESS_RETURN] = df[COL_RETURN] - SYNTHETIC_RF_MONTHLY
    logger.info(
        f"  [{df_name}] Excess_Return added "
        f"(proxy RF = {SYNTHETIC_RF_MONTHLY:.4%}/month).  "
        f"Will be updated with actual RF in factor_merge.py."
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MONTHLY CONTINUITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_monthly_continuity(
    df: pd.DataFrame,
    df_name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Detect and report gaps in the monthly time-series for each fund.

    A "gap" is a month where the consecutive difference between two Date values
    for the same fund exceeds ~45 days (i.e., a month was skipped).  Gaps are
    logged as warnings but rows are NOT dropped — the regression module handles
    them through the ``statsmodels`` OLS incomplete-panel approach.

    Also detects duplicate (Fund_Name, Year_Month) pairs, which would produce
    incorrect return calculations.  Duplicates are de-duplicated by keeping
    the last occurrence (latest data wins) and logged as warnings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Fund_Name``, ``Date``, ``Year_Month`` columns,
        sorted by (Fund_Name, Date).
    df_name : str, optional
        Label for log messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with (Fund_Name, Year_Month) duplicates resolved.
    """
    logger.info(f"\n── Step: Validate monthly continuity [{df_name}] ──")
    df = df.copy()

    # ── Detect (Fund_Name, Year_Month) duplicates ─────────────────────────────
    dup_key = [COL_FUND_NAME, COL_YEAR_MONTH]
    n_dups = report_duplicates(df, subset=dup_key, df_name=df_name)
    if n_dups > 0:
        logger.warning(
            f"  [{df_name}] Resolving {n_dups} (Fund_Name, Year_Month) duplicates "
            f"by keeping the last occurrence."
        )
        df = df.drop_duplicates(subset=dup_key, keep="last")
        df = df.sort_values([COL_FUND_NAME, COL_DATE]).reset_index(drop=True)

    # ── Gap detection per fund ────────────────────────────────────────────────
    GAP_THRESHOLD_DAYS = 45   # more than ~1.5 months = a skipped month

    total_gaps = 0
    gap_report: list[dict] = []

    for fund_name, group in df.groupby(COL_FUND_NAME, sort=False):
        dates = group[COL_DATE].sort_values()
        deltas = dates.diff().dt.days.dropna()
        gap_mask = deltas > GAP_THRESHOLD_DAYS
        n_gaps = gap_mask.sum()
        if n_gaps > 0:
            total_gaps += n_gaps
            gap_dates = dates.iloc[1:][gap_mask.values].dt.strftime("%Y-%m").tolist()
            gap_report.append(
                {"Fund": fund_name, "Gaps": n_gaps, "At months": gap_dates}
            )

    if total_gaps == 0:
        logger.info(
            f"  [{df_name}] Monthly continuity check passed — no gaps detected."
        )
    else:
        logger.warning(
            f"  [{df_name}] {total_gaps} month-gap(s) detected across "
            f"{len(gap_report)} fund(s).  Rows are retained; regression will "
            f"proceed on available observations."
        )
        for entry in gap_report:
            logger.warning(
                f"      {entry['Fund'][:60]:60s}  "
                f"gaps={entry['Gaps']}  at: {entry['At months']}"
            )

    logger.info(
        f"  [{df_name}] Continuity check complete.  "
        f"Frame shape: {df.shape}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EXPENSE RATIO MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_expense_ratios(
    master_df: pd.DataFrame,
    er_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join expense ratio data onto the master fund DataFrame using a two-pass
    merge strategy: (1) exact normalised-name match, then (2) Jaccard fuzzy
    fallback for any unmatched funds.

    Strategy
    --------
    Pass 1 — Exact match on ``Fund_Name_Norm``
        The normalised ER fund names are joined directly to the master frame
        normalised keys.

    Pass 2 — Fuzzy fallback (Jaccard similarity ≥ 0.50)
        For funds that did not get an ER match in Pass 1, we attempt to find
        the closest ER fund by token overlap.  This handles the known edge
        cases discovered during inspection:
        * "Franklin India NSE Nifty 50 Index" ↔ "franklin india index nse nifty 50 index"

    Post-merge
        Funds with no ER match after both passes are filled with the median
        ER of their ``FUND_TYPE`` group and flagged in the log.

    Parameters
    ----------
    master_df : pd.DataFrame
        The concatenated, cleaned master DataFrame.
    er_df : pd.DataFrame
        Raw expense ratio DataFrame from ``load_expense_ratios``.

    Returns
    -------
    pd.DataFrame
        Master DataFrame with ``Expense_Ratio`` column added.
    """
    logger.info("\n── Step: Merge expense ratios ──")

    master_df = master_df.copy()
    er_df = er_df.copy()

    # ── Normalise ER fund names ───────────────────────────────────────────────
    er_df["er_norm"] = er_df[COL_ER_FUND_RAW].apply(normalize_fund_name)
    er_df = er_df.drop_duplicates(subset="er_norm", keep="first")

    # Build lookup: norm_key → expense ratio
    er_lookup: dict[str, float] = dict(
        zip(er_df["er_norm"], er_df[COL_ER_VALUE_RAW])
    )
    er_norm_keys = list(er_lookup.keys())

    # ── Pass 1: Exact normalised-name match ───────────────────────────────────
    fund_norm_keys = master_df[COL_FUND_NAME_NORM].unique()
    pass1_matched: dict[str, float] = {}
    pass1_unmatched: list[str] = []

    for norm_key in fund_norm_keys:
        if norm_key in er_lookup:
            pass1_matched[norm_key] = er_lookup[norm_key]
        else:
            pass1_unmatched.append(norm_key)

    logger.info(
        f"  ER merge Pass 1 (exact):  "
        f"{len(pass1_matched)}/{len(fund_norm_keys)} funds matched."
    )

    # ── Pass 2: Fuzzy fallback for unmatched funds ────────────────────────────
    pass2_matched: dict[str, float] = {}
    pass2_unmatched: list[str] = []

    for norm_key in pass1_unmatched:
        best = find_best_match(norm_key, er_norm_keys, threshold=0.50)
        if best is not None:
            er_val = er_lookup[best]
            pass2_matched[norm_key] = er_val
            logger.info(
                f"  ER merge Pass 2 (fuzzy):  "
                f"'{norm_key}'  →  '{best}'  ({er_val:.2f}%)"
            )
        else:
            pass2_unmatched.append(norm_key)

    # ── Combine both passes ───────────────────────────────────────────────────
    all_matched = {**pass1_matched, **pass2_matched}
    master_df[COL_ER] = master_df[COL_FUND_NAME_NORM].map(all_matched)

    # ── Fill unmatched with FUND_TYPE median ──────────────────────────────────
    n_missing_er = master_df[COL_ER].isna().sum()
    if n_missing_er > 0:
        type_medians = master_df.groupby(COL_FUND_TYPE)[COL_ER].median()
        for fund_type, median_er in type_medians.items():
            mask = master_df[COL_ER].isna() & (master_df[COL_FUND_TYPE] == fund_type)
            master_df.loc[mask, COL_ER] = median_er
            affected_funds = master_df.loc[mask, COL_FUND_NAME].unique()
            if len(affected_funds) > 0:
                logger.warning(
                    f"  ER merge: {len(affected_funds)} {fund_type} fund(s) had "
                    f"no ER match — filled with {fund_type} median "
                    f"({median_er:.2f}%):"
                )
                for fn in affected_funds[:5]:
                    logger.warning(f"      {fn}")

        if len(pass2_unmatched) > 0:
            logger.warning(
                f"  ER merge: {len(pass2_unmatched)} fund(s) unmatched after both "
                f"passes (ER filled with type median):"
            )
            for fn in pass2_unmatched:
                logger.warning(f"      {fn}")

    # ── Final ER coverage summary ─────────────────────────────────────────────
    er_by_type = master_df.drop_duplicates(COL_FUND_NAME).groupby(COL_FUND_TYPE)[
        COL_ER
    ].agg(["mean", "median", "min", "max"])
    logger.info(f"  Expense ratio summary by FUND_TYPE:\n{er_by_type.to_string()}")

    return master_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MASTER FRAME BUILDER (MAIN ORCHESTRATOR)
# ══════════════════════════════════════════════════════════════════════════════

def build_master_frame(
    passive_df: pd.DataFrame,
    active_df: pd.DataFrame,
    er_df: pd.DataFrame,
    save_intermediates: bool = True,
) -> pd.DataFrame:
    """
    Orchestrate all preprocessing steps and return the final, analysis-ready
    master DataFrame.

    Pipeline (in order)
    -------------------
    1.  Clean + label passive and active DataFrames independently.
    2.  Validate returns for each (cross-check vs NAV-derived).
    3.  Standardise returns for each (drop NaN, winsorise, add Excess_Return).
    4.  Validate monthly continuity for each.
    5.  Filter both to the common analysis window (2013-01 → 2026-04).
    6.  Save per-type cleaned files (optional).
    7.  Concatenate into a single master DataFrame.
    8.  Apply (Fund_Name, Year_Month) deduplication on the combined frame.
    9.  Flag funds below MIN_OBS threshold (not dropped; flagged for downstream).
    10. Merge expense ratios (two-pass: exact + fuzzy).
    11. Final validation of the master frame.
    12. Save master CSV.

    Parameters
    ----------
    passive_df : pd.DataFrame
        Raw passive NAV DataFrame from ``load_passive_nav``.
    active_df : pd.DataFrame
        Raw active NAV DataFrame from ``load_active_nav``.
    er_df : pd.DataFrame
        Raw expense ratio DataFrame from ``load_expense_ratios``.
    save_intermediates : bool, optional
        If True, save cleaned passive, active, and master DataFrames to CSV.
        Default True.

    Returns
    -------
    pd.DataFrame
        Master DataFrame ready for ``factor_merge.py``.

        Guaranteed columns:
            Fund_Name, Fund_Name_Norm, Fund_Type, Date, Year_Month,
            NAV, Monthly_Return, Excess_Return, Expense_Ratio

        Shape: approximately (7 900, 9) after filtering to 2013-01 → 2026-04.

    Raises
    ------
    ValueError
        If the resulting master frame has fewer than 100 rows (catastrophic
        data loss) or is missing required columns.
    """
    logger.info("\n" + "=" * 68)
    logger.info("  PREPROCESSING PHASE — build_master_frame")
    logger.info("=" * 68)

    # ═══ STEP 1: Per-file cleaning & FUND_TYPE assignment ═════════════════════
    cleaned_passive, cleaned_active = assign_fund_type(passive_df, active_df)

    # ═══ STEP 2: Return cross-validation ══════════════════════════════════════
    cleaned_passive = validate_returns(cleaned_passive, df_name="Passive NAV")
    cleaned_active  = validate_returns(cleaned_active,  df_name="Active NAV")

    # ═══ STEP 3: Standardise returns ══════════════════════════════════════════
    cleaned_passive = standardise_returns(cleaned_passive, df_name="Passive NAV")
    cleaned_active  = standardise_returns(cleaned_active,  df_name="Active NAV")

    # ═══ STEP 4: Monthly continuity validation ═════════════════════════════════
    cleaned_passive = validate_monthly_continuity(
        cleaned_passive, df_name="Passive NAV"
    )
    cleaned_active = validate_monthly_continuity(
        cleaned_active, df_name="Active NAV"
    )

    # ═══ STEP 5: Filter to common analysis window ════════════════════════════
    logger.info(f"\n── Step: Filter to analysis window [{ANALYSIS_START} → {ANALYSIS_END}] ──")

    cleaned_passive = filter_date_range(
        cleaned_passive, COL_DATE, ANALYSIS_START, ANALYSIS_END,
        df_name="Passive NAV (post-filter)",
    )
    cleaned_active = filter_date_range(
        cleaned_active, COL_DATE, ANALYSIS_START, ANALYSIS_END,
        df_name="Active NAV (post-filter)",
    )

    # ═══ STEP 6: Save per-type cleaned files ══════════════════════════════════
    if save_intermediates:
        save_dataframe(cleaned_passive, PASSIVE_CLEAN_FILE, description="Passive clean")
        save_dataframe(cleaned_active,  ACTIVE_CLEAN_FILE,  description="Active clean")

    # ═══ STEP 7: Concatenate ══════════════════════════════════════════════════
    logger.info(f"\n── Step: Concatenate passive + active ──")
    master_df = pd.concat(
        [cleaned_passive, cleaned_active],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    master_df = master_df.sort_values(
        [COL_FUND_TYPE, COL_FUND_NAME, COL_DATE]
    ).reset_index(drop=True)

    logger.info(
        f"  Concatenated master frame: {len(master_df):,} rows  |  "
        f"{master_df[COL_FUND_NAME].nunique()} funds  |  "
        f"{master_df[COL_FUND_TYPE].value_counts().to_dict()}"
    )

    # ═══ STEP 8: Global (Fund_Name, Year_Month) dedup on combined frame ════════
    master_df = validate_monthly_continuity(
        master_df, df_name="Master (combined)"
    )

    # ═══ STEP 9: Flag funds below MIN_OBS ════════════════════════════════════
    logger.info(f"\n── Step: Flag low-observation funds (MIN_OBS={MIN_OBS}) ──")
    obs_per_fund = master_df.groupby(COL_FUND_NAME)[COL_RETURN].count()
    below_min = obs_per_fund[obs_per_fund < MIN_OBS]

    if not below_min.empty:
        logger.warning(
            f"  {len(below_min)} fund(s) have < {MIN_OBS} return observations "
            f"(will be EXCLUDED from regression but retained in master frame):"
        )
        for fname, n_obs in below_min.items():
            fund_type = master_df.loc[
                master_df[COL_FUND_NAME] == fname, COL_FUND_TYPE
            ].iloc[0]
            logger.warning(
                f"      [{fund_type:7s}]  {fname:55s}  →  {n_obs} months"
            )
    else:
        logger.info(
            f"  All funds meet the MIN_OBS={MIN_OBS} threshold. ✓"
        )

    # Add a flag column so regression_analysis.py can filter easily
    master_df["Eligible_For_Regression"] = master_df[COL_FUND_NAME].map(
        obs_per_fund
    ) >= MIN_OBS

    n_eligible = master_df[COL_FUND_NAME].isin(
        obs_per_fund[obs_per_fund >= MIN_OBS].index
    ).sum()
    logger.info(
        f"  Eligible rows: {n_eligible:,} / {len(master_df):,}  |  "
        f"Eligible funds: {(obs_per_fund >= MIN_OBS).sum()} / "
        f"{len(obs_per_fund)}"
    )

    # ═══ STEP 10: Merge expense ratios ════════════════════════════════════════
    master_df = merge_expense_ratios(master_df, er_df)

    # ═══ STEP 11: Final validation ════════════════════════════════════════════
    logger.info("\n── Step: Final master frame validation ──")

    required_final_cols = [
        COL_FUND_NAME,
        COL_FUND_NAME_NORM,
        COL_FUND_TYPE,
        COL_DATE,
        COL_YEAR_MONTH,
        COL_NAV,
        COL_RETURN,
        COL_EXCESS_RETURN,
        COL_ER,
    ]
    validate_dataframe(
        master_df,
        required_final_cols,
        df_name="Master Frame",
        min_rows=100,
    )
    report_missing(master_df, df_name="Master Frame (final)")

    # ── Fund composition report ────────────────────────────────────────────────
    type_counts = master_df.groupby(COL_FUND_TYPE)[COL_FUND_NAME].nunique()
    date_range = (
        f"{master_df[COL_DATE].min().date()} → {master_df[COL_DATE].max().date()}"
    )
    er_summary = master_df.drop_duplicates(COL_FUND_NAME).groupby(COL_FUND_TYPE)[
        COL_ER
    ].mean()

    logger.info(f"\n  ╔══════════ MASTER FRAME SUMMARY ══════════╗")
    logger.info(f"  ║  Total rows        : {len(master_df):>8,}              ║")
    logger.info(f"  ║  Total funds       : {master_df[COL_FUND_NAME].nunique():>8}              ║")
    logger.info(f"  ║  Active funds      : {type_counts.get(FUND_TYPE_ACTIVE, 0):>8}              ║")
    logger.info(f"  ║  Passive funds     : {type_counts.get(FUND_TYPE_PASSIVE, 0):>8}              ║")
    logger.info(f"  ║  Date range        : {date_range:<27} ║")
    logger.info(f"  ║  Mean ER (Active)  : {er_summary.get(FUND_TYPE_ACTIVE, 0):>7.2f}%              ║")
    logger.info(f"  ║  Mean ER (Passive) : {er_summary.get(FUND_TYPE_PASSIVE, 0):>7.2f}%              ║")
    logger.info(f"  ╚══════════════════════════════════════════╝")

    # ═══ STEP 12: Save master frame ═══════════════════════════════════════════
    if save_intermediates:
        save_dataframe(master_df, MASTER_DATA_FILE, description="Master Frame")

    logger.info("\n" + "=" * 68)
    logger.info("  PREPROCESSING COMPLETE — master frame ready.")
    logger.info("=" * 68 + "\n")

    return master_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SELF-TEST (python src/preprocessing.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import load_all_data

    _log = setup_logger("preprocessing_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  PREPROCESSING SELF-TEST")
    _log.info("═" * 68)

    try:
        # ── Load raw data ─────────────────────────────────────────────────────
        raw = load_all_data()

        # ── Build master frame ────────────────────────────────────────────────
        master = build_master_frame(
            passive_df=raw["passive_nav"],
            active_df=raw["active_nav"],
            er_df=raw["expense_ratios"],
            save_intermediates=True,
        )

        # ── Assertions ────────────────────────────────────────────────────────
        _log.info("\n── Spot-check assertions ──")

        # Column completeness
        required = [
            COL_FUND_NAME, COL_FUND_NAME_NORM, COL_FUND_TYPE, COL_DATE,
            COL_YEAR_MONTH, COL_NAV, COL_RETURN, COL_EXCESS_RETURN, COL_ER,
        ]
        missing_cols = [c for c in required if c not in master.columns]
        assert not missing_cols, f"Missing columns: {missing_cols}"
        _log.info(f"  ✅ All {len(required)} required columns present.")

        # FUND_TYPE distribution
        type_dist = master[COL_FUND_TYPE].value_counts()
        assert FUND_TYPE_ACTIVE  in type_dist.index, "No Active rows in master"
        assert FUND_TYPE_PASSIVE in type_dist.index, "No Passive rows in master"
        _log.info(
            f"  ✅ FUND_TYPE distribution: "
            f"Active={type_dist.get(FUND_TYPE_ACTIVE, 0):,}  "
            f"Passive={type_dist.get(FUND_TYPE_PASSIVE, 0):,}"
        )

        # Date range within analysis window
        assert master[COL_DATE].min() >= pd.Timestamp(ANALYSIS_START), \
            "Date below ANALYSIS_START"
        assert master[COL_DATE].max() <= pd.Timestamp(ANALYSIS_END), \
            "Date above ANALYSIS_END"
        _log.info(
            f"  ✅ Date range within analysis window: "
            f"{master[COL_DATE].min().date()} → {master[COL_DATE].max().date()}"
        )

        # No NaN in Monthly_Return (should have been dropped)
        n_nan_ret = master[COL_RETURN].isna().sum()
        assert n_nan_ret == 0, f"Found {n_nan_ret} NaN returns in master!"
        _log.info("  ✅ No NaN Monthly_Return rows in master frame.")

        # Expense ratio populated
        n_nan_er = master[COL_ER].isna().sum()
        assert n_nan_er == 0, f"Found {n_nan_er} NaN Expense_Ratio rows!"
        _log.info(
            f"  ✅ Expense_Ratio fully populated.  "
            f"Range: {master[COL_ER].min():.2f}% – {master[COL_ER].max():.2f}%"
        )

        # Return values in reasonable range
        assert master[COL_RETURN].min() >= -0.50, "Return below -50% after winsorisation!"
        assert master[COL_RETURN].max() <=  1.00, "Return above 100% after winsorisation!"
        _log.info(
            f"  ✅ Returns bounded: "
            f"[{master[COL_RETURN].min():.4f}, {master[COL_RETURN].max():.4f}]"
        )

        # Excess return column exists and is numeric
        assert master[COL_EXCESS_RETURN].dtype in [np.float64, np.float32], \
            "Excess_Return is not float"
        _log.info("  ✅ Excess_Return column present and numeric.")

        # Saved files exist
        assert MASTER_DATA_FILE.exists(),   "Master CSV not saved!"
        assert PASSIVE_CLEAN_FILE.exists(), "Passive clean CSV not saved!"
        assert ACTIVE_CLEAN_FILE.exists(),  "Active clean CSV not saved!"
        _log.info("  ✅ All intermediate CSV files saved successfully.")

        _log.info(
            f"\n  Master frame shape: {master.shape}"
        )
        _log.info(
            f"  Columns: {list(master.columns)}"
        )
        _log.info(
            f"\n  Sample rows (first 3 passive, first 3 active):"
        )
        sample = pd.concat([
            master[master[COL_FUND_TYPE] == FUND_TYPE_PASSIVE].head(3),
            master[master[COL_FUND_TYPE] == FUND_TYPE_ACTIVE].head(3),
        ])
        display_cols = [
            COL_FUND_NAME, COL_FUND_TYPE, COL_DATE,
            COL_NAV, COL_RETURN, COL_EXCESS_RETURN, COL_ER
        ]
        _log.info(f"\n{sample[display_cols].to_string(index=False)}")

        _log.info("\n✅  All preprocessing self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Preprocessing self-test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        raise

    _log.info("═" * 68 + "\n")
