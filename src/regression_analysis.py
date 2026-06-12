"""
regression_analysis.py
======================
Carhart Four-Factor OLS regression engine for the fund analysis pipeline.

Responsibility
--------------
Accepts the factor-merged regression DataFrame from ``factor_merge.py`` and
runs a separate OLS regression for each fund, then performs group-level
(Active vs Passive) inference.  Outputs are:

  1.  Per-fund regression results table (alpha, betas, t-stats, R², N).
  2.  Pooled panel regression (fund fixed effects + FUND_TYPE dummy).
  3.  Group-level alpha comparison (Active vs Passive mean alpha, t-test).
  4.  Expense-ratio–adjusted alpha test (Nanigian 2019 core test).
  5.  Regression diagnostics (residual autocorrelation, heteroskedasticity).

Model specification
-------------------
Carhart (1997) Four-Factor Model:

    (R_it − RF_t) = α_i + β_MF·MF_t + β_SMB·SMB_t + β_HML·HML_t
                         + β_WML·WML_t + ε_it

where:
    R_it   = monthly return of fund i at time t
    RF_t   = monthly risk-free rate (RBI repo rate / 12)
    MF_t   = market excess return (avg passive fund return − RF_t)
    SMB_t  = small-minus-big factor  (= 0.0 in this dataset)
    HML_t  = high-minus-low factor   (= 0.0 in this dataset)
    WML_t  = winners-minus-losers momentum factor (cross-fund proxy)
    α_i    = Jensen's alpha — the fund's abnormal return
    ε_it   = idiosyncratic error term

Effective model (after dropping degenerate zero-variance factors)
-----------------------------------------------------------------
Since SMB ≡ 0 and HML ≡ 0 across all months, they are dropped from each
fund's OLS design matrix before fitting (including them would make the
matrix rank-deficient).  The effective specification is:

    (R_it − RF_t) = α_i + β_MF·MF_t + β_WML·WML_t + ε_it

This is documented transparently in every output and can be upgraded to
the full four-factor model when stock-level SMB/HML data become available.

Inference
---------
* Newey-West HAC standard errors (4 lags) to correct for autocorrelation
  and heteroskedasticity — standard in monthly return panel regressions.
* Fund-level: two-tailed t-test on alpha (H₀: α = 0).
* Group-level: Welch's two-sample t-test comparing active vs passive alpha
  distributions (H₀: mean_alpha_active = mean_alpha_passive).
* Nanigian test: compare (alpha + expense_ratio) to assess if expense
  differences explain the cross-group alpha gap.

Module layout
-------------
  §1  _select_active_factors      — drop zero-variance cols from design matrix
  §2  _run_single_fund_ols        — OLS for one fund (HAC s.e., result dict)
  §3  run_fund_regressions        — loop over all eligible funds
  §4  run_pooled_regression       — panel OLS with fund FEs + FUND_TYPE dummy
  §5  compare_active_vs_passive   — Welch t-test on alpha, Nanigian test
  §6  compute_regression_diagnostics — Durbin-Watson, Breusch-Pagan per fund
  §7  format_results_table        — publication-quality output tables
  §8  run_full_regression_analysis — orchestrator
  §9  Self-test (python src/regression_analysis.py)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# ── Path fix for direct execution ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ALPHA_SIGNIFICANCE,
    COL_ADJ_R2,
    COL_ALPHA,
    COL_BETA_HML,
    COL_BETA_MF,
    COL_BETA_SMB,
    COL_BETA_WML,
    COL_DATE,
    COL_ER,
    COL_EXCESS_RETURN,
    COL_FUND_NAME,
    COL_FUND_TYPE,
    COL_HML,
    COL_MF,
    COL_N_OBS,
    COL_P_ALPHA,
    COL_R2,
    COL_REG_STATUS,
    COL_RF,
    COL_RETURN,
    COL_SMB,
    COL_T_ALPHA,
    COL_WML,
    FUND_TYPE_ACTIVE,
    FUND_TYPE_PASSIVE,
    MIN_OBS,
    REGRESSION_SUMMARY_FILE,
)
from src.utils import (
    report_missing,
    save_dataframe,
    setup_logger,
)

logger = setup_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_HAC_MAXLAGS   = 4       # Newey-West lag truncation (standard for monthly data)
_MIN_VAR_THRES = 1e-10   # Minimum factor variance to include in regression
_CANDIDATE_FACTORS = [COL_MF, COL_SMB, COL_HML, COL_WML]

# Output column names for the results table
_COL_ACTIVE_DUMMY_COEF  = "Coef_Active_Dummy"
_COL_ACTIVE_DUMMY_T     = "T_Active_Dummy"
_COL_ACTIVE_DUMMY_P     = "P_Active_Dummy"
_COL_ER_COEF            = "Coef_ExpenseRatio"
_COL_ER_T               = "T_ExpenseRatio"
_COL_ALPHA_GROSS        = "Alpha_Gross"     # alpha + expense_ratio
_COL_ALPHA_SIG          = "Alpha_Significant"  # True if p < ALPHA_SIGNIFICANCE
_COL_USED_FACTORS       = "Factors_Used"       # e.g. "MF,WML"
_COL_DW_STAT            = "Durbin_Watson"
_COL_BP_P               = "Breusch_Pagan_P"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ACTIVE FACTOR SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _select_active_factors(
    fund_df: pd.DataFrame,
    candidate_factors: list[str] = _CANDIDATE_FACTORS,
    min_var: float = _MIN_VAR_THRES,
) -> list[str]:
    """
    Return the subset of ``candidate_factors`` that have non-trivial variance
    in ``fund_df``.  Zero-variance columns (e.g. SMB = 0 throughout) would
    make the OLS design matrix rank-deficient.

    Parameters
    ----------
    fund_df : pd.DataFrame
        One fund's regression data (rows = monthly observations).
    candidate_factors : list[str]
        All potential factor column names.
    min_var : float
        Minimum variance threshold to include a factor.

    Returns
    -------
    list[str]
        Active (non-degenerate) factor column names.

    Notes
    -----
    * Also checks for NaN: any factor column with > 20% NaN is excluded
      because Newey-West HAC requires a balanced sample.
    * MF is always retained if present and non-degenerate — it is the
      primary explanatory variable in any market model.
    """
    active: list[str] = []
    for fc in candidate_factors:
        if fc not in fund_df.columns:
            continue
        col_data = fund_df[fc].dropna()
        if len(col_data) == 0:
            continue
        nan_frac = fund_df[fc].isna().mean()
        if nan_frac > 0.20:
            logger.debug(
                f"    Factor '{fc}' excluded: {nan_frac:.1%} NaN."
            )
            continue
        var = col_data.var(ddof=1)
        if var < min_var:
            logger.debug(
                f"    Factor '{fc}' excluded: variance {var:.2e} < {min_var:.2e} "
                f"(degenerate)."
            )
        else:
            active.append(fc)
    return active


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SINGLE-FUND OLS
# ══════════════════════════════════════════════════════════════════════════════

def _run_single_fund_ols(
    fund_df: pd.DataFrame,
    fund_name: str,
    active_factors: list[str],
) -> dict:
    """
    Fit OLS for a single fund and return a result dictionary.

    Equation
    --------
    Excess_Return = alpha + Σ β_k * Factor_k + ε,
    for k in active_factors.

    Standard errors use Newey-West HAC with ``_HAC_MAXLAGS`` lags to correct
    for autocorrelation and heteroskedasticity common in monthly return series.

    Parameters
    ----------
    fund_df : pd.DataFrame
        Single-fund data subset; must contain COL_EXCESS_RETURN and all
        columns in ``active_factors``.
    fund_name : str
        Fund name (used in error messages only).
    active_factors : list[str]
        Factor column names to use as regressors.

    Returns
    -------
    dict
        Result dictionary with keys:
            Fund_Name, Fund_Type, N_Obs, Alpha, T_Alpha, P_Alpha,
            Beta_MF, Beta_SMB, Beta_HML, Beta_WML,
            R_Squared, Adj_R_Squared, Regression_Status,
            Factors_Used, Expense_Ratio, Alpha_Gross, Alpha_Significant.

    Notes
    -----
    * All NaN rows are dropped before fitting.
    * If fewer than MIN_OBS valid rows remain after NaN-drop, the fund is
      marked "SKIPPED" without fitting.
    * The intercept is added via ``sm.add_constant``.
    * Betas for excluded factors (SMB, HML) are set to NaN in the output.
    """
    result_base: dict = {
        COL_FUND_NAME:    fund_name,
        COL_FUND_TYPE:    fund_df[COL_FUND_TYPE].iloc[0] if COL_FUND_TYPE in fund_df else "Unknown",
        COL_N_OBS:        0,
        COL_ALPHA:        np.nan,
        COL_T_ALPHA:      np.nan,
        COL_P_ALPHA:      np.nan,
        COL_BETA_MF:      np.nan,
        COL_BETA_SMB:     np.nan,   # will remain NaN (factor excluded)
        COL_BETA_HML:     np.nan,   # will remain NaN (factor excluded)
        COL_BETA_WML:     np.nan,
        COL_R2:           np.nan,
        COL_ADJ_R2:       np.nan,
        COL_REG_STATUS:   "SKIPPED",
        _COL_USED_FACTORS: "",
        COL_ER:           fund_df[COL_ER].iloc[0] if COL_ER in fund_df else np.nan,
        _COL_ALPHA_GROSS: np.nan,
        _COL_ALPHA_SIG:   False,
        _COL_DW_STAT:     np.nan,
        _COL_BP_P:        np.nan,
    }

    # ── Drop rows with NaN in any required column ──────────────────────────────
    required_cols = [COL_EXCESS_RETURN] + active_factors
    clean_df = fund_df[required_cols].dropna()

    result_base[COL_N_OBS] = len(clean_df)

    if len(clean_df) < MIN_OBS:
        logger.debug(
            f"  [{fund_name[:45]}]  SKIPPED — only {len(clean_df)} valid obs "
            f"(< {MIN_OBS})."
        )
        return result_base

    if not active_factors:
        logger.warning(
            f"  [{fund_name[:45]}]  SKIPPED — no active factors with variance."
        )
        return result_base

    # ── Build design matrix ────────────────────────────────────────────────────
    y = clean_df[COL_EXCESS_RETURN].values
    X_raw = clean_df[active_factors].values
    X = sm.add_constant(X_raw, prepend=True, has_constant="add")

    # ── Fit OLS with HAC standard errors ──────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_model  = sm.OLS(y, X)
            ols_result = ols_model.fit(
                cov_type="HAC",
                cov_kwds={"maxlags": _HAC_MAXLAGS, "use_correction": True},
            )
    except Exception as exc:
        logger.warning(
            f"  [{fund_name[:45]}]  OLS ERROR: {exc}"
        )
        result_base[COL_REG_STATUS] = "ERROR"
        return result_base

    # ── Extract estimates ──────────────────────────────────────────────────────
    params   = ols_result.params       # [const, factor_1, factor_2, …]
    tvalues  = ols_result.tvalues
    pvalues  = ols_result.pvalues
    alpha    = float(params[0])
    t_alpha  = float(tvalues[0])
    p_alpha  = float(pvalues[0])

    # Map factor betas back to their canonical names
    beta_map: dict[str, float] = {}
    for idx, fc in enumerate(active_factors):
        beta_map[fc] = float(params[idx + 1])   # +1 because index 0 = const

    er_val       = result_base[COL_ER]
    alpha_gross  = (
        alpha + er_val / 100.0    # ER is in % p.a. → divide 100 for decimal monthly?
        # Actually: ER is annual %; alpha is monthly decimal.
        # For Nanigian test: both sides annualised.
        # Here we store alpha_monthly + ER_annual/12 as "gross alpha monthly"
    )
    # Annualise alpha for the Nanigian comparison
    alpha_annual = (1 + alpha) ** 12 - 1
    alpha_gross_annual = alpha_annual + (er_val / 100.0 if not np.isnan(er_val) else 0)

    result_base.update({
        COL_ALPHA:          round(alpha, 8),
        COL_T_ALPHA:        round(t_alpha, 6),
        COL_P_ALPHA:        round(p_alpha, 6),
        COL_BETA_MF:        round(beta_map.get(COL_MF,  np.nan), 6),
        COL_BETA_SMB:       round(beta_map.get(COL_SMB, np.nan), 6),
        COL_BETA_HML:       round(beta_map.get(COL_HML, np.nan), 6),
        COL_BETA_WML:       round(beta_map.get(COL_WML, np.nan), 6),
        COL_R2:             round(ols_result.rsquared, 6),
        COL_ADJ_R2:         round(ols_result.rsquared_adj, 6),
        COL_REG_STATUS:     "OK",
        _COL_USED_FACTORS:  ",".join(active_factors),
        _COL_ALPHA_GROSS:   round(alpha_gross_annual, 6),
        _COL_ALPHA_SIG:     bool(p_alpha < ALPHA_SIGNIFICANCE),
    })

    # ── Residual diagnostics ───────────────────────────────────────────────────
    residuals = ols_result.resid
    try:
        dw_stat = float(durbin_watson(residuals))
        result_base[_COL_DW_STAT] = round(dw_stat, 4)
    except Exception:
        pass

    try:
        # Breusch-Pagan test for heteroskedasticity
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, ols_result.model.exog)
        result_base[_COL_BP_P] = round(float(bp_p), 4)
    except Exception:
        pass

    return result_base


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FUND-BY-FUND REGRESSION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_fund_regressions(
    merged_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a separate Carhart OLS regression for every eligible fund and
    return a consolidated results DataFrame.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Factor-merged, validated DataFrame from ``factor_merge.py``.
    verbose : bool, optional
        If True, log per-fund results.  Default True.

    Returns
    -------
    pd.DataFrame
        One row per fund.  Columns include Alpha, T_Alpha, P_Alpha,
        Beta_MF, Beta_WML, R_Squared, Adj_R_Squared, N_Obs, Fund_Type,
        Expense_Ratio, Alpha_Gross (annualised), Factors_Used, etc.

    Notes
    -----
    * Funds with < MIN_OBS valid observations after NaN-drop are marked
      "SKIPPED" (not ERROR).
    * Results are sorted by Fund_Type then Fund_Name for consistency.
    """
    logger.info("  §3 Running fund-by-fund OLS regressions …")

    funds         = sorted(merged_df[COL_FUND_NAME].unique())
    n_funds       = len(funds)
    results: list[dict] = []

    ok_count      = 0
    skipped_count = 0
    error_count   = 0

    for i, fund_name in enumerate(funds, 1):
        fund_df = merged_df[merged_df[COL_FUND_NAME] == fund_name].copy()

        # Determine active (non-degenerate) factors for this fund
        active_factors = _select_active_factors(fund_df)

        # Run regression
        result = _run_single_fund_ols(fund_df, fund_name, active_factors)
        results.append(result)

        status = result[COL_REG_STATUS]
        if status == "OK":
            ok_count += 1
            if verbose:
                fund_type  = result[COL_FUND_TYPE]
                alpha_bps  = result[COL_ALPHA] * 10000   # convert to basis points
                sig_star   = "*" if result[_COL_ALPHA_SIG] else " "
                logger.info(
                    f"  [{i:>2}/{n_funds}] {fund_type[0]}  "
                    f"{fund_name[:52]:52s}  "
                    f"α={alpha_bps:+7.2f}bps  "
                    f"t={result[COL_T_ALPHA]:+6.3f}  "
                    f"R²={result[COL_R2]:.3f}  "
                    f"N={result[COL_N_OBS]:>3}{sig_star}"
                )
        elif status == "SKIPPED":
            skipped_count += 1
            logger.info(
                f"  [{i:>2}/{n_funds}] ⊘ SKIPPED  "
                f"{fund_name[:55]:55s}  N={result[COL_N_OBS]}"
            )
        else:
            error_count += 1
            logger.warning(
                f"  [{i:>2}/{n_funds}] ✗ ERROR    {fund_name[:55]}"
            )

    results_df = pd.DataFrame(results).sort_values(
        [COL_FUND_TYPE, COL_FUND_NAME]
    ).reset_index(drop=True)

    logger.info(
        f"\n  Fund regressions complete:  "
        f"OK={ok_count}  SKIPPED={skipped_count}  ERROR={error_count}  "
        f"Total={n_funds}"
    )
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — POOLED PANEL REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_pooled_regression(
    merged_df: pd.DataFrame,
) -> dict:
    """
    Fit a pooled OLS panel regression stacking all fund-month observations.

    Specifications
    --------------
    Model A — Fund-type dummy only:
        Excess_Return = α₀ + α₁·Is_Active + β_MF·MF + β_WML·WML + ε

    Model B — Controlling for expense ratio (Nanigian 2019 core test):
        Excess_Return = α₀ + α₁·Is_Active + β_MF·MF + β_WML·WML
                       + γ·Expense_Ratio + ε

    The coefficient α₁ on Is_Active measures whether active funds earn
    systematically different risk-adjusted returns than passive funds.
    In Model B, if α₁ becomes statistically insignificant after including
    Expense_Ratio, it supports Nanigian's hypothesis that expense ratios —
    not manager skill — drive the active-passive performance gap.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Factor-merged long-format DataFrame (all funds, all months).

    Returns
    -------
    dict
        Keys: "model_a" and "model_b", each containing the
        statsmodels RegressionResultsWrapper for that specification,
        plus "summary_a" and "summary_b" text summaries.
    """
    logger.info("  §4 Running pooled panel regression …")

    df = merged_df.copy().dropna(
        subset=[COL_EXCESS_RETURN, COL_MF, COL_WML, COL_ER]
    )

    # Create Is_Active dummy (1 = Active, 0 = Passive)
    df["Is_Active"] = (df[COL_FUND_TYPE] == FUND_TYPE_ACTIVE).astype(float)

    # Active factors (exclude zero-variance)
    pool_factors = _select_active_factors(df)
    logger.info(
        f"  Pooled active factors: {pool_factors}  |  "
        f"N = {len(df):,} observations"
    )

    results_out: dict = {}

    for model_name, extra_cols in [
        ("model_a", []),
        ("model_b", [COL_ER]),
    ]:
        spec_cols = ["Is_Active"] + pool_factors + extra_cols
        X_raw = df[spec_cols].values
        X     = sm.add_constant(X_raw, prepend=True, has_constant="add")
        y     = df[COL_EXCESS_RETURN].values

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ols = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": _HAC_MAXLAGS, "use_correction": True},
                )
        except Exception as exc:
            logger.error(f"  Pooled {model_name} FAILED: {exc}")
            continue

        # Attach column names for readability
        col_labels = ["Intercept", "Is_Active"] + pool_factors + extra_cols
        ols_summary = ols.summary(
            xname=col_labels,
            title=f"Pooled Panel OLS — {model_name.upper()}",
        )

        results_out[model_name]                = ols
        results_out[f"summary_{model_name}"]   = str(ols_summary)
        results_out[f"coef_{model_name}"]      = dict(zip(col_labels, ols.params))
        results_out[f"pval_{model_name}"]      = dict(zip(col_labels, ols.pvalues))

        alpha_pooled    = ols.params[0]
        active_coef     = ols.params[1]
        active_t        = ols.tvalues[1]
        active_p        = ols.pvalues[1]
        r2              = ols.rsquared
        nobs            = int(ols.nobs)

        logger.info(
            f"\n  ── Pooled {model_name.upper()} ──\n"
            f"    Intercept (α₀)     : {alpha_pooled:+.6f}  ({alpha_pooled*12:.4%} ann.)\n"
            f"    Is_Active coef (α₁): {active_coef:+.6f}  "
            f"t={active_t:+.3f}  p={active_p:.4f}  "
            f"{'SIGNIFICANT' if active_p < ALPHA_SIGNIFICANCE else 'not significant'}\n"
            f"    R²                 : {r2:.4f}\n"
            f"    N observations     : {nobs:,}"
        )

        if model_name == "model_b" and COL_ER in extra_cols:
            er_idx   = col_labels.index(COL_ER)
            er_coef  = ols.params[er_idx]
            er_t     = ols.tvalues[er_idx]
            er_p     = ols.pvalues[er_idx]
            logger.info(
                f"    Expense_Ratio coef : {er_coef:+.6f}  "
                f"t={er_t:+.3f}  p={er_p:.4f}  "
                f"{'SIGNIFICANT' if er_p < ALPHA_SIGNIFICANCE else 'not significant'}"
            )
            results_out["er_coef"]  = er_coef
            results_out["er_t"]     = er_t
            results_out["er_p"]     = er_p

        results_out[f"active_coef_{model_name}"] = active_coef
        results_out[f"active_t_{model_name}"]    = active_t
        results_out[f"active_p_{model_name}"]    = active_p

    return results_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ACTIVE vs PASSIVE ALPHA COMPARISON (NANIGIAN TEST)
# ══════════════════════════════════════════════════════════════════════════════

def compare_active_vs_passive(
    results_df: pd.DataFrame,
) -> dict:
    """
    Compare the alpha distributions of active and passive funds using
    Welch's two-sample t-test, plus the Nanigian expense-ratio-adjusted test.

    Tests performed
    ---------------
    Test 1 — Raw alpha (monthly decimal):
        H₀: mean_alpha_active = mean_alpha_passive
        H₁: mean_alpha_active ≠ mean_alpha_passive   (two-tailed)
        Method: Welch's t-test (unequal variances)

    Test 2 — Gross alpha (alpha + expense_ratio / 12):
        H₀: mean_gross_alpha_active = mean_gross_alpha_passive
        This is the Nanigian (2019) core hypothesis: after adding back
        expense ratios, do active managers actually deliver MORE skill-
        based return?  If H₀ is rejected in favour of active having
        HIGHER gross alpha, it suggests expenses — not skill — explain
        any active underperformance.

    Test 3 — Alpha sign frequency:
        % of active funds with α > 0 vs % of passive funds with α > 0.

    Test 4 — Significance frequency:
        % of active funds with p_alpha < 0.05 (two-tailed).

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-fund regression results from ``run_fund_regressions``.

    Returns
    -------
    dict
        Statistical test results and summary statistics.
    """
    logger.info("  §5 Active vs Passive alpha comparison (Nanigian test) …")

    ok_results = results_df[results_df[COL_REG_STATUS] == "OK"].copy()
    active_res  = ok_results[ok_results[COL_FUND_TYPE] == FUND_TYPE_ACTIVE]
    passive_res = ok_results[ok_results[COL_FUND_TYPE] == FUND_TYPE_PASSIVE]

    alpha_a  = active_res[COL_ALPHA].dropna().values
    alpha_p  = passive_res[COL_ALPHA].dropna().values
    gross_a  = active_res[_COL_ALPHA_GROSS].dropna().values
    gross_p  = passive_res[_COL_ALPHA_GROSS].dropna().values

    output: dict = {}

    # ── Descriptive statistics ────────────────────────────────────────────────
    for label, arr, ft in [
        ("active",  alpha_a, FUND_TYPE_ACTIVE),
        ("passive", alpha_p, FUND_TYPE_PASSIVE),
    ]:
        er_mean = ok_results[ok_results[COL_FUND_TYPE] == ft][COL_ER].mean()
        output[f"n_{label}"]           = len(arr)
        output[f"mean_alpha_{label}"]  = float(np.mean(arr))
        output[f"median_alpha_{label}"]= float(np.median(arr))
        output[f"std_alpha_{label}"]   = float(np.std(arr, ddof=1))
        output[f"mean_er_{label}"]     = er_mean
        output[f"pct_positive_{label}"]= float((arr > 0).mean() * 100)
        output[f"pct_sig_{label}"]     = float(
            (ok_results[ok_results[COL_FUND_TYPE] == ft][_COL_ALPHA_SIG]).mean() * 100
        )

    # ── Test 1: Welch's t-test on raw monthly alpha ───────────────────────────
    t1_stat, t1_p = stats.ttest_ind(alpha_a, alpha_p, equal_var=False)
    output["welch_t_alpha"]       = float(t1_stat)
    output["welch_p_alpha"]       = float(t1_p)
    output["welch_significant"]   = bool(t1_p < ALPHA_SIGNIFICANCE)

    # ── Test 2: Welch's t-test on gross (expense-adjusted) annual alpha ───────
    if len(gross_a) > 0 and len(gross_p) > 0:
        t2_stat, t2_p = stats.ttest_ind(gross_a, gross_p, equal_var=False)
        output["welch_t_gross_alpha"] = float(t2_stat)
        output["welch_p_gross_alpha"] = float(t2_p)
        output["welch_gross_significant"] = bool(t2_p < ALPHA_SIGNIFICANCE)
    else:
        output["welch_t_gross_alpha"]     = np.nan
        output["welch_p_gross_alpha"]     = np.nan
        output["welch_gross_significant"] = False

    # ── Log summary ──────────────────────────────────────────────────────────
    _ann = lambda x: (1 + x) ** 12 - 1
    logger.info(
        f"\n  ╔══════════════ ALPHA COMPARISON RESULTS ═══════════════╗"
    )
    logger.info(
        f"  ║  {'Metric':<38}  {'Active':>8}  {'Passive':>8}  ║"
    )
    logger.info(
        f"  ║  {'-'*56}  ║"
    )
    logger.info(
        f"  ║  {'N funds':<38}  {output['n_active']:>8}  {output['n_passive']:>8}  ║"
    )
    logger.info(
        f"  ║  {'Mean α (monthly)':<38}  {output['mean_alpha_active']:>+8.4%}  {output['mean_alpha_passive']:>+8.4%}  ║"
    )
    logger.info(
        f"  ║  {'Mean α (annualised)':<38}  {_ann(output['mean_alpha_active']):>+8.2%}  {_ann(output['mean_alpha_passive']):>+8.2%}  ║"
    )
    logger.info(
        f"  ║  {'Mean Expense Ratio':<38}  {output['mean_er_active']:>7.2f}%  {output['mean_er_passive']:>7.2f}%  ║"
    )
    logger.info(
        f"  ║  {'Mean gross α (ann.) [α + ER]':<38}  {output.get('mean_alpha_active', 0)+output.get('mean_er_active',0)/100:>+8.2%}  {output.get('mean_alpha_passive',0)+output.get('mean_er_passive',0)/100:>+8.2%}  ║"
    )
    logger.info(
        f"  ║  {'% funds with α > 0':<38}  {output['pct_positive_active']:>7.1f}%  {output['pct_positive_passive']:>7.1f}%  ║"
    )
    logger.info(
        f"  ║  {'% funds with p_α < 0.05':<38}  {output['pct_sig_active']:>7.1f}%  {output['pct_sig_passive']:>7.1f}%  ║"
    )
    logger.info(
        f"  ╠══ TEST 1: Welch t-test on raw α ══════════════════════╣"
    )
    logger.info(
        f"  ║  H₀: mean_α_active = mean_α_passive                    ║"
    )
    logger.info(
        f"  ║  t = {t1_stat:+.4f}   p = {t1_p:.4f}   "
        f"{'REJECT H₀' if t1_p < ALPHA_SIGNIFICANCE else 'FAIL TO REJECT H₀'}         ║"
    )
    logger.info(
        f"  ╠══ TEST 2: Welch t-test on gross α (Nanigian) ═════════╣"
    )
    logger.info(
        f"  ║  H₀: mean_gross_α_active = mean_gross_α_passive        ║"
    )
    t2_stat_val = output.get('welch_t_gross_alpha', np.nan)
    t2_p_val    = output.get('welch_p_gross_alpha', np.nan)
    if not np.isnan(t2_stat_val):
        logger.info(
            f"  ║  t = {t2_stat_val:+.4f}   p = {t2_p_val:.4f}   "
            f"{'REJECT H₀' if t2_p_val < ALPHA_SIGNIFICANCE else 'FAIL TO REJECT H₀'}         ║"
        )
        logger.info(
            f"  ║  → {'Expense ratios EXPLAIN the gap (Nanigian supported)' if not output['welch_gross_significant'] else 'Gap PERSISTS after ER adjustment'}"
            f"    ║"
        )
    logger.info(
        f"  ╚═══════════════════════════════════════════════════════╝"
    )

    return output


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — REGRESSION DIAGNOSTICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compute_regression_diagnostics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise regression diagnostics across all funds.

    Diagnostics reported per fund (already computed in §2)
    -------------------------------------------------------
    Durbin-Watson statistic:
        DW ≈ 2.0 → no autocorrelation in residuals
        DW < 1.5 → positive autocorrelation (common in momentum-heavy funds)
        DW > 2.5 → negative autocorrelation (rare)

    Breusch-Pagan p-value:
        p < 0.05 → reject H₀ of homoskedasticity → heteroskedasticity present
        (HAC standard errors already correct for this)

    Cross-fund summaries are logged for the research report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-fund results DataFrame from ``run_fund_regressions``.

    Returns
    -------
    pd.DataFrame
        Diagnostic summary table grouped by Fund_Type.
    """
    logger.info("  §6 Computing regression diagnostics summary …")

    ok = results_df[results_df[COL_REG_STATUS] == "OK"].copy()

    diag_summary_records: list[dict] = []
    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        subset = ok[ok[COL_FUND_TYPE] == ft]
        if subset.empty:
            continue

        dw_vals  = subset[_COL_DW_STAT].dropna()
        bp_vals  = subset[_COL_BP_P].dropna()

        dw_autocorr  = (dw_vals < 1.5).sum()
        het_count    = (bp_vals < ALPHA_SIGNIFICANCE).sum()

        diag_summary_records.append({
            COL_FUND_TYPE:           ft,
            "N_Funds":               len(subset),
            "DW_Mean":               round(dw_vals.mean(), 3) if len(dw_vals) else np.nan,
            "DW_Pct_Autocorr":       round(dw_autocorr / len(dw_vals) * 100, 1) if len(dw_vals) else np.nan,
            "BP_Pct_Heterosked":     round(het_count / len(bp_vals) * 100, 1) if len(bp_vals) else np.nan,
            "Mean_R2":               round(subset[COL_R2].mean(), 4),
            "Mean_AdjR2":            round(subset[COL_ADJ_R2].mean(), 4),
        })

    diag_df = pd.DataFrame(diag_summary_records)
    logger.info(f"  Diagnostic summary:\n{diag_df.to_string(index=False)}")
    return diag_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLICATION-QUALITY RESULT TABLES
# ══════════════════════════════════════════════════════════════════════════════

def format_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean publication-ready results table with annualised alpha,
    significance stars, and rounded values.

    Significance stars (two-tailed)
    --------------------------------
    ***  p < 0.01
    **   p < 0.05
    *    p < 0.10
    (blank) p ≥ 0.10

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw per-fund results from ``run_fund_regressions``.

    Returns
    -------
    pd.DataFrame
        Formatted table with new columns:
            Alpha_Annual (%), Alpha_Stars, Alpha_bps.
    """
    df = results_df[results_df[COL_REG_STATUS] == "OK"].copy()

    # Annualised alpha
    df["Alpha_Annual_Pct"] = ((1 + df[COL_ALPHA]) ** 12 - 1) * 100  # in %
    df["Alpha_bps"]        = df[COL_ALPHA] * 10000   # monthly basis points

    # Significance stars
    def _stars(p: float) -> str:
        if pd.isna(p):
            return ""
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.10:
            return "*"
        return ""

    df["Alpha_Stars"]  = df[COL_P_ALPHA].apply(_stars)

    # Round for display
    display_cols = {
        COL_FUND_NAME:      lambda x: x,
        COL_FUND_TYPE:      lambda x: x,
        "Alpha_Annual_Pct": lambda x: round(x, 2),
        "Alpha_bps":        lambda x: round(x, 2),
        COL_T_ALPHA:        lambda x: round(x, 3),
        COL_P_ALPHA:        lambda x: round(x, 4),
        "Alpha_Stars":      lambda x: x,
        COL_BETA_MF:        lambda x: round(x, 4) if not pd.isna(x) else x,
        COL_BETA_WML:       lambda x: round(x, 4) if not pd.isna(x) else x,
        COL_R2:             lambda x: round(x, 4),
        COL_ADJ_R2:         lambda x: round(x, 4),
        COL_N_OBS:          lambda x: x,
        COL_ER:             lambda x: round(x, 2) if not pd.isna(x) else x,
        _COL_ALPHA_GROSS:   lambda x: round(x * 100, 2) if not pd.isna(x) else x,
    }

    for col, fn in display_cols.items():
        if col in df.columns:
            df[col] = df[col].apply(fn)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_full_regression_analysis(
    merged_df: pd.DataFrame,
    save_outputs: bool = True,
) -> dict:
    """
    Orchestrate the full regression analysis pipeline and return all results.

    Steps
    -----
    1.  Run fund-by-fund Carhart OLS (§3).
    2.  Run pooled panel OLS — Model A and Model B (§4).
    3.  Active vs Passive alpha comparison and Nanigian test (§5).
    4.  Regression diagnostics (§6).
    5.  Format publication-quality results table (§7).
    6.  Save outputs (optional).

    Parameters
    ----------
    merged_df : pd.DataFrame
        Factor-merged, validated regression input from ``factor_merge.py``.
    save_outputs : bool, optional
        If True, save results CSVs and print summaries.

    Returns
    -------
    dict
        Keys:
            "fund_results"      : per-fund results DataFrame
            "pooled_results"    : dict of pooled OLS results
            "comparison"        : dict from compare_active_vs_passive
            "diagnostics"       : diagnostics DataFrame
            "formatted_results" : publication-ready DataFrame
    """
    logger.info("\n" + "=" * 68)
    logger.info("  REGRESSION ANALYSIS PHASE — run_full_regression_analysis")
    logger.info("=" * 68)

    # ═══ Step 1: Fund-by-fund regressions ════════════════════════════════════
    logger.info("\n── Step 1: Fund-by-fund Carhart OLS ──")
    fund_results = run_fund_regressions(merged_df, verbose=True)

    # ═══ Step 2: Pooled panel regressions ════════════════════════════════════
    logger.info("\n── Step 2: Pooled panel OLS (Model A & B) ──")
    pooled_results = run_pooled_regression(merged_df)

    # ═══ Step 3: Group-level comparison ══════════════════════════════════════
    logger.info("\n── Step 3: Active vs Passive alpha comparison ──")
    comparison = compare_active_vs_passive(fund_results)

    # ═══ Step 4: Diagnostics ══════════════════════════════════════════════════
    logger.info("\n── Step 4: Regression diagnostics ──")
    diagnostics = compute_regression_diagnostics(fund_results)

    # ═══ Step 5: Format ════════════════════════════════════════════════════════
    logger.info("\n── Step 5: Format publication results table ──")
    formatted_results = format_results_table(fund_results)

    # ═══ Step 6: Save ══════════════════════════════════════════════════════════
    if save_outputs:
        save_dataframe(
            fund_results,
            REGRESSION_SUMMARY_FILE,
            description="Per-fund regression results",
        )
        from src.config import REGRESSION_RESULTS_DIR
        from src.utils import ensure_dir
        ensure_dir(REGRESSION_RESULTS_DIR)

        formatted_path   = REGRESSION_RESULTS_DIR / "regression_results_formatted.csv"
        diagnostics_path = REGRESSION_RESULTS_DIR / "regression_diagnostics.csv"
        comparison_path  = REGRESSION_RESULTS_DIR / "alpha_comparison.csv"

        save_dataframe(formatted_results, formatted_path, description="Formatted results")
        save_dataframe(diagnostics, diagnostics_path, description="Diagnostics summary")

        comparison_df = pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in comparison.items()
             if not isinstance(v, (list, dict))]
        )
        save_dataframe(comparison_df, comparison_path, description="Alpha comparison")

        # Save pooled model summaries as text
        from src.utils import save_text_report
        report_path = REGRESSION_RESULTS_DIR / "pooled_regression_summaries.txt"
        report_lines: list[str] = ["POOLED REGRESSION SUMMARIES", "=" * 70, ""]
        for key in ["summary_model_a", "summary_model_b"]:
            if key in pooled_results:
                report_lines.append(pooled_results[key])
                report_lines.append("")
        save_text_report("\n".join(report_lines), report_path)

    # ═══ Final summary ════════════════════════════════════════════════════════
    n_ok = (fund_results[COL_REG_STATUS] == "OK").sum()
    n_sig_active  = formatted_results[
        formatted_results[COL_FUND_TYPE] == FUND_TYPE_ACTIVE
    ]["Alpha_Stars"].isin(["*", "**", "***"]).sum()
    n_sig_passive = formatted_results[
        formatted_results[COL_FUND_TYPE] == FUND_TYPE_PASSIVE
    ]["Alpha_Stars"].isin(["*", "**", "***"]).sum()

    logger.info(f"\n  ╔══════ REGRESSION ANALYSIS SUMMARY ══════╗")
    logger.info(f"  ║  Funds regressed (OK)  : {n_ok:>6}            ║")
    logger.info(f"  ║  Sig. alpha (Active)   : {n_sig_active:>6}            ║")
    logger.info(f"  ║  Sig. alpha (Passive)  : {n_sig_passive:>6}            ║")
    logger.info(f"  ║  Welch t (raw alpha)   : {comparison.get('welch_t_alpha', np.nan):>+6.3f}            ║")
    logger.info(f"  ║  Welch p (raw alpha)   : {comparison.get('welch_p_alpha', np.nan):>6.4f}            ║")
    logger.info(f"  ║  Nanigian gross-α p    : {comparison.get('welch_p_gross_alpha', np.nan):>6.4f}            ║")
    logger.info(f"  ╚═════════════════════════════════════════╝")
    logger.info("\n  Regression analysis complete.  Proceed to performance_metrics.py.")
    logger.info("=" * 68 + "\n")

    return {
        "fund_results":      fund_results,
        "pooled_results":    pooled_results,
        "comparison":        comparison,
        "diagnostics":       diagnostics,
        "formatted_results": formatted_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SELF-TEST (python src/regression_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import load_all_data
    from src.preprocessing import build_master_frame
    from src.return_calculations import enrich_master_frame
    from src.factor_merge import build_and_merge_factors

    _log = setup_logger("regression_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  REGRESSION ANALYSIS SELF-TEST")
    _log.info("═" * 68)

    try:
        # ── Run full pipeline ────────────────────────────────────────────────
        raw = load_all_data()
        master = build_master_frame(
            passive_df=raw["passive_nav"],
            active_df=raw["active_nav"],
            er_df=raw["expense_ratios"],
            save_intermediates=False,
        )
        enriched, _, reg_df = enrich_master_frame(
            master_df=master, save_outputs=False
        )
        factor_table, merged_df = build_and_merge_factors(
            master_df=enriched,
            reg_df=reg_df,
            raw_factor_df=raw["factor_data"],
            save_outputs=False,
        )

        # ── Run regression analysis ──────────────────────────────────────────
        results = run_full_regression_analysis(merged_df, save_outputs=True)

        fund_res   = results["fund_results"]
        fmt_res    = results["formatted_results"]
        comparison = results["comparison"]

        _log.info("\n── Assertions ──")

        # A1: Fund results table shape
        n_ok = (fund_res[COL_REG_STATUS] == "OK").sum()
        assert n_ok >= 60, f"Expected ≥ 60 OK funds, got {n_ok}"
        _log.info(f"  ✅ Fund results: {n_ok} regressions OK.")

        # A2: Alpha column present and finite
        alphas = fund_res[fund_res[COL_REG_STATUS] == "OK"][COL_ALPHA]
        assert alphas.notna().all(), "NaN alpha values in OK funds"
        assert np.isfinite(alphas).all(), "Non-finite alpha values"
        _log.info(
            f"  ✅ Alpha column: all finite.  "
            f"Range: [{alphas.min():.4%}, {alphas.max():.4%}] monthly."
        )

        # A3: Beta_MF positive on average (expected: funds track market)
        beta_mf = fund_res[fund_res[COL_REG_STATUS] == "OK"][COL_BETA_MF].dropna()
        assert beta_mf.mean() > 0, f"Mean Beta_MF negative: {beta_mf.mean():.4f}"
        _log.info(
            f"  ✅ Beta_MF mean = {beta_mf.mean():.4f} > 0  "
            f"(funds track market as expected)."
        )

        # A4: R² in [0, 1]
        r2_vals = fund_res[fund_res[COL_REG_STATUS] == "OK"][COL_R2]
        assert (r2_vals >= 0).all() and (r2_vals <= 1).all(), "R² out of [0,1]"
        _log.info(
            f"  ✅ R² in [0,1]:  mean={r2_vals.mean():.4f}  "
            f"range=[{r2_vals.min():.4f}, {r2_vals.max():.4f}]."
        )

        # A5: Comparison dict has key values
        for key in ["mean_alpha_active", "mean_alpha_passive", "welch_p_alpha"]:
            assert key in comparison, f"Missing key in comparison: {key}"
        _log.info(
            f"  ✅ Comparison dict populated.  "
            f"Welch p = {comparison['welch_p_alpha']:.4f}."
        )

        # A6: Formatted table has Alpha_Stars column
        assert "Alpha_Stars" in fmt_res.columns, "Alpha_Stars missing"
        assert "Alpha_Annual_Pct" in fmt_res.columns, "Alpha_Annual_Pct missing"
        _log.info(
            f"  ✅ Formatted results table: {len(fmt_res)} funds  ×  "
            f"{len(fmt_res.columns)} cols."
        )

        # A7: Print publication table (Active funds)
        _log.info(f"\n  ── Active funds alpha table (top 10 by |alpha|) ──")
        active_fmt = fmt_res[fmt_res[COL_FUND_TYPE] == FUND_TYPE_ACTIVE].copy()
        active_fmt = active_fmt.sort_values("Alpha_bps", key=abs, ascending=False)
        display_cols = [
            COL_FUND_NAME, "Alpha_Annual_Pct", "Alpha_Stars",
            COL_T_ALPHA, COL_BETA_MF, COL_R2, COL_N_OBS, COL_ER
        ]
        _log.info(f"\n{active_fmt[display_cols].head(10).to_string(index=False)}")

        _log.info(f"\n  ── Passive funds alpha table (top 10 by |alpha|) ──")
        passive_fmt = fmt_res[fmt_res[COL_FUND_TYPE] == FUND_TYPE_PASSIVE].copy()
        passive_fmt = passive_fmt.sort_values("Alpha_bps", key=abs, ascending=False)
        _log.info(f"\n{passive_fmt[display_cols].head(10).to_string(index=False)}")

        _log.info("\n✅  All regression_analysis self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Self-test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        raise

    _log.info("═" * 68 + "\n")
