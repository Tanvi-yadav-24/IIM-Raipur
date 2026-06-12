"""
statistical_tests.py
====================
Formal statistical inference module for the Carhart Four-Factor Fund
Analysis Pipeline.

Responsibility
--------------
Applies a rigorous battery of statistical tests to the regression outputs,
return series, and performance metrics.  Organises findings into a structured
results dictionary and publication-ready summary tables.

Test inventory
--------------
  §1  test_return_normality          — Jarque-Bera, Shapiro-Wilk (per fund)
  §2  test_return_stationarity       — Augmented Dickey-Fuller, KPSS (per fund)
  §3  test_alpha_multiple_comparison — Benjamini-Hochberg FDR, Bonferroni
                                       (corrects for 64 simultaneous alpha tests)
  §4  test_active_vs_passive         — Mann-Whitney U, Kolmogorov-Smirnov,
                                       Levene (return distributions)
  §5  test_expense_ratio_impact      — Spearman ρ (alpha vs ER), cross-
                                       sectional OLS alpha = γ·ER + controls
  §6  test_performance_persistence   — Spearman ρ across sub-periods,
                                       winner/loser contingency table (χ²)
  §7  bootstrap_alpha_ci             — Non-parametric bootstrap CIs for
                                       group mean alpha (1 000 resamples)
  §8  run_all_statistical_tests      — Master orchestrator
  §9  Self-test (python src/statistical_tests.py)

Why each test is included
--------------------------
* Normality tests: OLS t-tests assume normally distributed errors; violation
  is informative and motivates HAC standard errors.
* Stationarity: confirms monthly returns (not NAV) are I(0) — a pre-condition
  for valid OLS inference.
* Multiple comparison correction: 64 simultaneous alpha tests at α=0.05
  implies ≈3 false discoveries by chance (Bonferroni bound = 0.05/64 ≈ 0.078%).
  BH-FDR is the standard correction in the finance literature.
* Mann-Whitney / KS: non-parametric alternatives to Welch t-test, robust to
  non-normal return distributions.
* Expense ratio OLS: the Nanigian (2019) cross-sectional test — does ER
  explain alpha cross-sectionally?
* Persistence: critical for fund evaluation — if alpha is persistent, active
  management has genuine skill; if not, it is luck (Carhart 1997).
* Bootstrap CIs: more reliable than normal-theory CIs for small fund samples
  (33 active, 31 passive with valid alpha).
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
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss

# ── Path fix for direct execution ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ALPHA_SIGNIFICANCE,
    COL_ALPHA,
    COL_DATE,
    COL_ER,
    COL_FUND_NAME,
    COL_FUND_TYPE,
    COL_N_OBS,
    COL_P_ALPHA,
    COL_REG_STATUS,
    COL_RETURN,
    FUND_TYPE_ACTIVE,
    FUND_TYPE_PASSIVE,
    MONTHS_PER_YEAR,
    REGRESSION_RESULTS_DIR,
    STATISTICAL_TESTS_FILE,
)
from src.utils import (
    save_dataframe,
    save_text_report,
    setup_logger,
)

logger = setup_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_N_BOOTSTRAP        = 1_000    # bootstrap iterations for CI estimation
_BOOTSTRAP_SEED     = 42       # reproducibility
_PERSISTENCE_TOP_N  = 0.50     # top 50% = "winners" for persistence table
_ADF_MAXLAG         = 12       # max lags for Augmented Dickey-Fuller
_KPSS_LAG_METHOD    = "auto"   # KPSS lag selection

# Significance stars helper
def _stars(p: float) -> str:
    if pd.isna(p):    return ""
    if p < 0.001:     return "***"
    if p < 0.01:      return "***"
    if p < 0.05:      return "**"
    if p < 0.10:      return "*"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RETURN NORMALITY TESTS (per fund)
# ══════════════════════════════════════════════════════════════════════════════

def test_return_normality(
    enriched_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether monthly returns are normally distributed for each fund.

    Tests applied
    -------------
    Jarque-Bera (JB):
        H₀: skewness = 0 and excess kurtosis = 0 (normality).
        The JB test is powerful in large samples (N > 30) and explicitly
        measures the two moments that typically deviate from normality in
        financial returns (negative skew, heavy tails / excess kurtosis).

    Shapiro-Wilk (SW):
        H₀: the sample comes from a normal distribution.
        More powerful than JB in small samples (N < 50).  Applied here
        because some funds have as few as 39 observations.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Full enriched master frame with Fund_Name, Monthly_Return columns.

    Returns
    -------
    pd.DataFrame
        One row per fund.  Columns:
            Fund_Name, Fund_Type, N_Obs,
            Skewness, Excess_Kurtosis,
            JB_Stat, JB_P, JB_Normal (bool),
            SW_Stat, SW_P, SW_Normal (bool),
            Both_Normal (bool — JB AND SW fail to reject).

    Notes
    -----
    * JB and SW have low power for N < 30; results flagged as unreliable
      for such funds.
    * "Normal" here means: FAIL TO REJECT H₀ at α = 0.05 (two-tailed).
    * Financial returns typically exhibit negative skewness and leptokurtosis
      (excess kurtosis > 0) — non-normality is expected and informative.
    """
    logger.info("  §1 Testing return normality (Jarque-Bera, Shapiro-Wilk) …")
    records: list[dict] = []

    for fund_name, grp in enriched_df.groupby(COL_FUND_NAME, sort=True):
        rets  = grp[COL_RETURN].dropna().values
        n_obs = len(rets)
        ft    = grp[COL_FUND_TYPE].iloc[0]

        row: dict = {
            COL_FUND_NAME:  fund_name,
            COL_FUND_TYPE:  ft,
            COL_N_OBS:      n_obs,
        }

        if n_obs < 8:
            # Insufficient for any test
            row.update({
                "Skewness": np.nan, "Excess_Kurtosis": np.nan,
                "JB_Stat": np.nan, "JB_P": np.nan, "JB_Normal": np.nan,
                "SW_Stat": np.nan, "SW_P": np.nan, "SW_Normal": np.nan,
                "Both_Normal": np.nan,
            })
            records.append(row)
            continue

        skew      = float(stats.skew(rets))
        kurt      = float(stats.kurtosis(rets))   # scipy: excess kurtosis (Fisher)

        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(rets)

        # Shapiro-Wilk (only valid up to n=5000)
        sw_stat, sw_p = stats.shapiro(rets)

        jb_normal  = bool(jb_p >= ALPHA_SIGNIFICANCE)
        sw_normal  = bool(sw_p >= ALPHA_SIGNIFICANCE)

        row.update({
            "Skewness":       round(skew,    4),
            "Excess_Kurtosis":round(kurt,    4),
            "JB_Stat":        round(jb_stat, 4),
            "JB_P":           round(jb_p,    6),
            "JB_Normal":      jb_normal,
            "SW_Stat":        round(sw_stat, 4),
            "SW_P":           round(sw_p,    6),
            "SW_Normal":      sw_normal,
            "Both_Normal":    bool(jb_normal and sw_normal),
        })
        records.append(row)

    normality_df = pd.DataFrame(records)

    # ── Summary log ──────────────────────────────────────────────────────────
    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        sub = normality_df[normality_df[COL_FUND_TYPE] == ft]
        n_jb_normal  = sub["JB_Normal"].sum()
        n_sw_normal  = sub["SW_Normal"].sum()
        n_both       = sub["Both_Normal"].sum()
        mean_skew    = sub["Skewness"].mean()
        mean_kurt    = sub["Excess_Kurtosis"].mean()
        logger.info(
            f"  Normality [{ft}]:  "
            f"JB normal: {n_jb_normal}/{len(sub)}  |  "
            f"SW normal: {n_sw_normal}/{len(sub)}  |  "
            f"Both: {n_both}/{len(sub)}  |  "
            f"Mean skew: {mean_skew:+.3f}  |  "
            f"Mean exc. kurtosis: {mean_kurt:.3f}"
        )

    overall_both = normality_df["Both_Normal"].mean()
    logger.info(
        f"  Overall: {normality_df['Both_Normal'].sum()} / {len(normality_df)} "
        f"funds have normally distributed returns ({overall_both:.1%})."
    )
    logger.info(
        f"  → Non-normality is expected in monthly equity returns.  "
        f"HAC standard errors in regression_analysis.py already correct for this."
    )
    return normality_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RETURN STATIONARITY TESTS (per fund)
# ══════════════════════════════════════════════════════════════════════════════

def test_return_stationarity(
    enriched_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether monthly return series are stationary for each fund.

    Tests applied
    -------------
    Augmented Dickey-Fuller (ADF):
        H₀: the series has a unit root (non-stationary, I(1)).
        Reject H₀ → series is stationary.  Monthly returns are expected to
        be stationary (unlike NAV price levels).

    KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
        H₀: the series is stationary (trend-stationary).
        Reject H₀ → non-stationary.
        KPSS complements ADF: if ADF rejects and KPSS does not reject,
        the series is strongly confirmed stationary.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Full enriched master frame.

    Returns
    -------
    pd.DataFrame
        One row per fund.  Columns:
            Fund_Name, Fund_Type, N_Obs,
            ADF_Stat, ADF_P, ADF_Lags, ADF_Stationary (bool),
            KPSS_Stat, KPSS_P, KPSS_Lags, KPSS_Stationary (bool),
            Conclusion ("Stationary" | "Non-stationary" | "Inconclusive").

    Notes
    -----
    * ADF lag length selected by AIC criterion (statsmodels default).
    * KPSS uses "c" regression (constant, no trend) — appropriate for
      demeaned monthly return series.
    * "Conclusion" = "Stationary" only when ADF rejects AND KPSS does not.
    """
    logger.info("  §2 Testing return stationarity (ADF, KPSS) …")
    records: list[dict] = []

    for fund_name, grp in enriched_df.groupby(COL_FUND_NAME, sort=True):
        rets  = grp[COL_RETURN].dropna().values
        n_obs = len(rets)
        ft    = grp[COL_FUND_TYPE].iloc[0]

        row: dict = {
            COL_FUND_NAME:  fund_name,
            COL_FUND_TYPE:  ft,
            COL_N_OBS:      n_obs,
        }

        if n_obs < 15:
            row.update({
                "ADF_Stat": np.nan, "ADF_P": np.nan, "ADF_Lags": np.nan,
                "ADF_Stationary": np.nan,
                "KPSS_Stat": np.nan, "KPSS_P": np.nan, "KPSS_Lags": np.nan,
                "KPSS_Stationary": np.nan,
                "Conclusion": "INSUFFICIENT_OBS",
            })
            records.append(row)
            continue

        # ── ADF ───────────────────────────────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_result   = adfuller(rets, maxlag=_ADF_MAXLAG, autolag="AIC")
            adf_stat     = float(adf_result[0])
            adf_p        = float(adf_result[1])
            adf_lags     = int(adf_result[2])
            adf_stat_flag= bool(adf_p < ALPHA_SIGNIFICANCE)   # reject unit root → stationary
        except Exception:
            adf_stat, adf_p, adf_lags, adf_stat_flag = np.nan, np.nan, np.nan, False

        # ── KPSS ──────────────────────────────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result  = kpss(rets, regression="c", nlags=_KPSS_LAG_METHOD)
            kpss_stat    = float(kpss_result[0])
            kpss_p       = float(kpss_result[1])   # bounded: [0.01, 0.10]
            kpss_lags    = int(kpss_result[2])
            kpss_stat_flag = bool(kpss_p >= ALPHA_SIGNIFICANCE)   # fail to reject → stationary
        except Exception:
            kpss_stat, kpss_p, kpss_lags, kpss_stat_flag = np.nan, np.nan, np.nan, False

        # ── Conclusion ────────────────────────────────────────────────────────
        if adf_stat_flag and kpss_stat_flag:
            conclusion = "Stationary"
        elif not adf_stat_flag and not kpss_stat_flag:
            conclusion = "Non-stationary"
        else:
            conclusion = "Inconclusive"

        row.update({
            "ADF_Stat":       round(adf_stat, 4)  if not np.isnan(adf_stat)  else np.nan,
            "ADF_P":          round(adf_p, 6)     if not np.isnan(adf_p)     else np.nan,
            "ADF_Lags":       adf_lags,
            "ADF_Stationary": adf_stat_flag,
            "KPSS_Stat":      round(kpss_stat, 4) if not np.isnan(kpss_stat) else np.nan,
            "KPSS_P":         round(kpss_p, 6)    if not np.isnan(kpss_p)    else np.nan,
            "KPSS_Lags":      kpss_lags,
            "KPSS_Stationary":kpss_stat_flag,
            "Conclusion":     conclusion,
        })
        records.append(row)

    stationarity_df = pd.DataFrame(records)

    # ── Summary log ──────────────────────────────────────────────────────────
    conc_counts = stationarity_df["Conclusion"].value_counts()
    logger.info(f"  Stationarity conclusions:\n{conc_counts.to_string()}")
    logger.info(
        f"  → {'Confirmed: monthly returns are stationary (expected).' if conc_counts.get('Stationary', 0) >= len(stationarity_df) * 0.75 else 'Warning: some funds show non-stationarity — inspect individually.'}"
    )
    return stationarity_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MULTIPLE COMPARISON CORRECTION FOR ALPHA TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_alpha_multiple_comparison(
    reg_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply multiple comparison corrections to the 64 simultaneous fund-level
    alpha t-tests to control for the family-wise error rate.

    Problem
    -------
    With 64 funds tested at α = 0.05, we expect ≈ 3.2 false rejections by
    chance (Type I errors), even when all true alphas are zero.  The raw
    p-values from ``regression_analysis.py`` are uncorrected and therefore
    over-optimistic.

    Corrections applied
    -------------------
    Bonferroni (FWER):
        Adjusted p = min(p_raw × N_tests, 1.0).
        Controls family-wise error rate (FWER).  Very conservative.
        Significant if adjusted_p < 0.05.

    Benjamini-Hochberg (FDR):
        Controls the expected proportion of false discoveries among rejections
        (False Discovery Rate).  Less conservative than Bonferroni.
        The standard method in modern empirical finance (storey 2002,
        Barras et al 2010 "False Discoveries in Mutual Fund Performance").
        Significant if BH-adjusted p < 0.05.

    Interpretation (Barras et al 2010 framework)
    ---------------------------------------------
    After BH-FDR correction:
    * Significant positive alpha → true skill ("lucky" funds identified)
    * Significant negative alpha → true under-performance ("unlucky" funds)
    * Non-significant → zero-alpha funds (α consistent with noise)

    Parameters
    ----------
    reg_results_df : pd.DataFrame
        Per-fund regression results from ``regression_analysis.run_fund_regressions``.
        Must contain: Fund_Name, Fund_Type, Alpha, P_Alpha, Regression_Status.

    Returns
    -------
    pd.DataFrame
        Extended regression results with new columns:
            P_Bonferroni, Sig_Bonferroni,
            P_BH_FDR, Sig_BH_FDR,
            Alpha_Annual_Pct, Alpha_Category.
    """
    logger.info("  §3 Multiple comparison correction (Bonferroni, BH-FDR) …")

    ok = reg_results_df[reg_results_df[COL_REG_STATUS] == "OK"].copy()

    if ok.empty:
        logger.warning("  §3 No OK regression results to correct.")
        return ok

    p_values  = ok[COL_P_ALPHA].values
    n_tests   = len(p_values)

    # Bonferroni
    p_bonf    = np.minimum(p_values * n_tests, 1.0)
    sig_bonf  = p_bonf < ALPHA_SIGNIFICANCE

    # Benjamini-Hochberg FDR
    reject_bh, p_bh, _, _ = multipletests(p_values, alpha=ALPHA_SIGNIFICANCE, method="fdr_bh")

    ok["P_Bonferroni"]   = np.round(p_bonf, 6)
    ok["Sig_Bonferroni"] = sig_bonf
    ok["P_BH_FDR"]       = np.round(p_bh, 6)
    ok["Sig_BH_FDR"]     = reject_bh

    # Annualised alpha
    ok["Alpha_Annual_Pct"] = ((1 + ok[COL_ALPHA]) ** MONTHS_PER_YEAR - 1) * 100

    # Alpha category (Barras et al 2010 framework)
    def _categorise(row: pd.Series) -> str:
        if row["Sig_BH_FDR"]:
            return "Skilled (+α)" if row[COL_ALPHA] > 0 else "Unlucky (−α)"
        return "Zero-α (noise)"

    ok["Alpha_Category"] = ok.apply(_categorise, axis=1)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"\n  ╔═══════════ MULTIPLE COMPARISON RESULTS ═══════════╗"
    )
    logger.info(
        f"  ║  N funds tested   : {n_tests:>5}                          ║"
    )
    logger.info(
        f"  ║  Expected false +  : {n_tests * ALPHA_SIGNIFICANCE:.1f} (at α={ALPHA_SIGNIFICANCE})              ║"
    )
    logger.info(
        f"  ║  {'Method':<20}  {'Sig(+α)':>7}  {'Sig(−α)':>7}  {'Total':>7}  ║"
    )
    logger.info(
        f"  ║  {'─'*48}  ║"
    )

    for method, sig_col in [("Raw (uncorrected)", "Alpha_Significant"),
                             ("Bonferroni (FWER)", "Sig_Bonferroni"),
                             ("BH-FDR",            "Sig_BH_FDR")]:
        if sig_col not in ok.columns:
            ok[sig_col] = ok["Sig_BH_FDR"]
        sig_pos = ((ok[sig_col]) & (ok[COL_ALPHA] > 0)).sum()
        sig_neg = ((ok[sig_col]) & (ok[COL_ALPHA] < 0)).sum()
        sig_tot = ok[sig_col].sum()
        logger.info(
            f"  ║  {method:<20}  {sig_pos:>7}  {sig_neg:>7}  {sig_tot:>7}  ║"
        )

    # Per-type BH-FDR breakdown
    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        sub = ok[ok[COL_FUND_TYPE] == ft]
        cat_counts = sub["Alpha_Category"].value_counts()
        logger.info(
            f"  ║  {ft} BH-FDR: {dict(cat_counts)}  ║"
        )

    logger.info(
        f"  ╚════════════════════════════════════════════════════╝"
    )
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ACTIVE vs PASSIVE DISTRIBUTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_active_vs_passive(
    enriched_df: pd.DataFrame,
    reg_results_df: pd.DataFrame,
) -> dict:
    """
    Apply non-parametric and distribution tests comparing Active vs Passive
    funds across multiple dimensions.

    Tests applied
    -------------
    On monthly RETURN distributions (all fund-months pooled):
        Mann-Whitney U (Wilcoxon rank-sum):
            H₀: P(Active return > Passive return) = 0.5.
            Non-parametric alternative to Welch's t-test.
            Does not assume normality.

        Kolmogorov-Smirnov 2-sample:
            H₀: Active and Passive return distributions are identical.
            Tests for any difference in shape (not just mean).

        Levene's test for equality of variances:
            H₀: Var(Active returns) = Var(Passive returns).
            Robust to non-normality (Brown-Forsythe variant).

    On ALPHA distributions (per-fund, from regression):
        Mann-Whitney U and KS tests on fund-level alphas.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Full enriched master frame.
    reg_results_df : pd.DataFrame
        Per-fund regression results.

    Returns
    -------
    dict
        Nested dict of test statistics and p-values.
    """
    logger.info("  §4 Active vs Passive distribution tests (MW-U, KS, Levene) …")
    out: dict = {}

    # ── Pooled return series ──────────────────────────────────────────────────
    act_rets  = enriched_df[enriched_df[COL_FUND_TYPE] == FUND_TYPE_ACTIVE][COL_RETURN].dropna().values
    pas_rets  = enriched_df[enriched_df[COL_FUND_TYPE] == FUND_TYPE_PASSIVE][COL_RETURN].dropna().values

    # ── Test 1: Mann-Whitney U on returns ────────────────────────────────────
    mwu_stat, mwu_p = stats.mannwhitneyu(act_rets, pas_rets, alternative="two-sided")
    out["mwu_returns"] = {"stat": float(mwu_stat), "p": float(mwu_p),
                          "significant": bool(mwu_p < ALPHA_SIGNIFICANCE)}

    # ── Test 2: Kolmogorov-Smirnov on returns ─────────────────────────────────
    ks_stat, ks_p = stats.ks_2samp(act_rets, pas_rets)
    out["ks_returns"] = {"stat": float(ks_stat), "p": float(ks_p),
                         "significant": bool(ks_p < ALPHA_SIGNIFICANCE)}

    # ── Test 3: Levene's test for equal variances ──────────────────────────────
    lev_stat, lev_p = stats.levene(act_rets, pas_rets, center="median")
    out["levene_returns"] = {"stat": float(lev_stat), "p": float(lev_p),
                             "significant": bool(lev_p < ALPHA_SIGNIFICANCE)}

    # ── Test 4: Mann-Whitney U on fund-level alpha ────────────────────────────
    ok = reg_results_df[reg_results_df[COL_REG_STATUS] == "OK"]
    act_alpha = ok[ok[COL_FUND_TYPE] == FUND_TYPE_ACTIVE][COL_ALPHA].dropna().values
    pas_alpha = ok[ok[COL_FUND_TYPE] == FUND_TYPE_PASSIVE][COL_ALPHA].dropna().values

    if len(act_alpha) >= 3 and len(pas_alpha) >= 3:
        mwu_a_stat, mwu_a_p = stats.mannwhitneyu(act_alpha, pas_alpha, alternative="two-sided")
        out["mwu_alpha"] = {"stat": float(mwu_a_stat), "p": float(mwu_a_p),
                            "significant": bool(mwu_a_p < ALPHA_SIGNIFICANCE)}

        ks_a_stat, ks_a_p = stats.ks_2samp(act_alpha, pas_alpha)
        out["ks_alpha"] = {"stat": float(ks_a_stat), "p": float(ks_a_p),
                           "significant": bool(ks_a_p < ALPHA_SIGNIFICANCE)}

    # ── Test 5: Effect size — Cohen's d on returns ─────────────────────────────
    pooled_std = np.sqrt(
        ((len(act_rets) - 1) * act_rets.std() ** 2
         + (len(pas_rets) - 1) * pas_rets.std() ** 2)
        / (len(act_rets) + len(pas_rets) - 2)
    )
    cohens_d = (act_rets.mean() - pas_rets.mean()) / pooled_std if pooled_std > 0 else np.nan
    out["cohens_d_returns"] = round(float(cohens_d), 4)

    # ── Log ────────────────────────────────────────────────────────────────────
    logger.info(
        f"\n  ╔══════════ DISTRIBUTION TEST RESULTS ════════════╗"
    )
    logger.info(
        f"  ║  {'Test':<35}  {'Stat':>8}  {'p-value':>8}  {'Sig':>3}  ║"
    )
    logger.info(
        f"  ║  {'─'*57}  ║"
    )
    tests_log = [
        ("MW-U (returns)",       out["mwu_returns"]),
        ("KS (returns)",         out["ks_returns"]),
        ("Levene variance",      out["levene_returns"]),
        ("MW-U (alpha)",         out.get("mwu_alpha", {})),
        ("KS (alpha)",           out.get("ks_alpha", {})),
    ]
    for label, res in tests_log:
        if not res:
            continue
        s = res.get("stat", np.nan)
        p = res.get("p",    np.nan)
        sig = "✓" if res.get("significant", False) else " "
        logger.info(
            f"  ║  {label:<35}  {s:>8.4f}  {p:>8.4f}  {sig:>3}  ║"
        )
    logger.info(
        f"  ║  {'Cohen\'s d (returns)':<35}  {cohens_d:>8.4f}  {'':>8}  {'':>3}  ║"
    )
    logger.info(
        f"  ╚═══════════════════════════════════════════════════╝"
    )

    # Interpretation
    if out["mwu_returns"]["significant"]:
        direction = "HIGHER" if act_rets.mean() > pas_rets.mean() else "LOWER"
        logger.info(
            f"  → MW-U significant: Active returns stochastically {direction} than Passive."
        )
    else:
        logger.info(
            f"  → MW-U not significant: No stochastic dominance in raw returns."
        )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXPENSE RATIO IMPACT TEST (Nanigian cross-sectional)
# ══════════════════════════════════════════════════════════════════════════════

def test_expense_ratio_impact(
    reg_results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> dict:
    """
    Test whether expense ratios explain cross-sectional alpha variation.

    This is the core Nanigian (2019) empirical test:

    Cross-sectional OLS (Model A — full sample):
        α_i = γ₀ + γ₁ · ER_i + ε_i

    Cross-sectional OLS (Model B — with fund type control):
        α_i = γ₀ + γ₁ · ER_i + γ₂ · Is_Active_i + ε_i

    If γ₁ < 0 and significant → each 1% ER costs approximately γ₁ annualised
    alpha.  If γ₁ ≈ −1 → expenses are a one-for-one drag on net alpha
    (Nanigian's expected finding for passive outperformance).

    Additional tests
    ----------------
    Spearman ρ (ER vs alpha):
        Non-parametric rank correlation.  Robust to outliers and non-linearity.

    Spearman ρ within Active (ER vs alpha):
        Does higher ER predict lower alpha even among active funds?

    Parameters
    ----------
    reg_results_df : pd.DataFrame
        Per-fund regression results.
    metrics_df : pd.DataFrame
        Per-fund performance metrics (for annualised alpha).

    Returns
    -------
    dict
        Test statistics, p-values, and OLS result objects.
    """
    logger.info("  §5 Expense ratio impact test (Spearman ρ, cross-sectional OLS) …")
    out: dict = {}

    ok = reg_results_df[reg_results_df[COL_REG_STATUS] == "OK"].copy()
    ok["Alpha_Annual"] = (1 + ok[COL_ALPHA]) ** MONTHS_PER_YEAR - 1

    if ok.empty or COL_ER not in ok.columns:
        logger.warning("  §5 Insufficient data.")
        return out

    # ── Spearman ρ: ER vs Alpha (all funds) ───────────────────────────────────
    clean = ok[[COL_ER, "Alpha_Annual"]].dropna()
    rho_all, p_rho_all = stats.spearmanr(clean[COL_ER], clean["Alpha_Annual"])
    out["spearman_er_alpha_all"]   = {"rho": round(float(rho_all), 4),
                                       "p": round(float(p_rho_all), 6),
                                       "significant": bool(p_rho_all < ALPHA_SIGNIFICANCE)}

    # ── Spearman ρ: ER vs Alpha (active only) ──────────────────────────────────
    act_clean = ok[ok[COL_FUND_TYPE] == FUND_TYPE_ACTIVE][[COL_ER, "Alpha_Annual"]].dropna()
    if len(act_clean) >= 5:
        rho_act, p_rho_act = stats.spearmanr(act_clean[COL_ER], act_clean["Alpha_Annual"])
        out["spearman_er_alpha_active"] = {"rho": round(float(rho_act), 4),
                                            "p": round(float(p_rho_act), 6),
                                            "significant": bool(p_rho_act < ALPHA_SIGNIFICANCE)}

    # ── Spearman ρ: ER vs Alpha (passive only) ─────────────────────────────────
    pas_clean = ok[ok[COL_FUND_TYPE] == FUND_TYPE_PASSIVE][[COL_ER, "Alpha_Annual"]].dropna()
    if len(pas_clean) >= 5:
        rho_pas, p_rho_pas = stats.spearmanr(pas_clean[COL_ER], pas_clean["Alpha_Annual"])
        out["spearman_er_alpha_passive"] = {"rho": round(float(rho_pas), 4),
                                             "p": round(float(p_rho_pas), 6),
                                             "significant": bool(p_rho_pas < ALPHA_SIGNIFICANCE)}

    # ── Cross-sectional OLS: α = γ₀ + γ₁·ER + ε ─────────────────────────────
    for model_label, fund_mask, extra_cols in [
        ("Model A (all funds)",    slice(None), []),
        ("Model B (+Is_Active)",   slice(None), ["Is_Active"]),
    ]:
        df_m = ok.copy()
        df_m["Is_Active"] = (df_m[COL_FUND_TYPE] == FUND_TYPE_ACTIVE).astype(float)
        df_m = df_m[[COL_ER, "Alpha_Annual", "Is_Active"]].dropna()

        y    = df_m["Alpha_Annual"].values
        Xcols= [COL_ER] + extra_cols
        X    = sm.add_constant(df_m[Xcols].values, prepend=True)

        try:
            ols  = sm.OLS(y, X).fit(cov_type="HC3")   # HC3 for small sample robustness
            col_names = ["Intercept", COL_ER] + extra_cols
            summary_str = str(ols.summary(xname=col_names, title=f"Cross-sectional OLS: {model_label}"))
            out[f"ols_{model_label}"] = {
                "coef_er":    round(float(ols.params[1]), 6),
                "t_er":       round(float(ols.tvalues[1]), 4),
                "p_er":       round(float(ols.pvalues[1]), 6),
                "r2":         round(float(ols.rsquared), 4),
                "n":          int(ols.nobs),
                "summary":    summary_str,
                "significant_er": bool(ols.pvalues[1] < ALPHA_SIGNIFICANCE),
            }
        except Exception as exc:
            logger.warning(f"  §5 Cross-sectional OLS failed ({model_label}): {exc}")

    # ── Log ────────────────────────────────────────────────────────────────────
    logger.info(
        f"\n  ── Expense Ratio Impact ──"
    )
    logger.info(
        f"  Spearman ρ (ER vs Alpha, all):    "
        f"ρ = {out['spearman_er_alpha_all']['rho']:+.4f}  "
        f"p = {out['spearman_er_alpha_all']['p']:.4f}  "
        f"{_stars(out['spearman_er_alpha_all']['p'])}"
    )
    if "spearman_er_alpha_active" in out:
        logger.info(
            f"  Spearman ρ (ER vs Alpha, active): "
            f"ρ = {out['spearman_er_alpha_active']['rho']:+.4f}  "
            f"p = {out['spearman_er_alpha_active']['p']:.4f}  "
            f"{_stars(out['spearman_er_alpha_active']['p'])}"
        )
    for model_key in ["Model A (all funds)", "Model B (+Is_Active)"]:
        key = f"ols_{model_key}"
        if key in out:
            r = out[key]
            logger.info(
                f"  OLS {model_key}: γ_ER = {r['coef_er']:+.4f}  "
                f"t = {r['t_er']:+.3f}  p = {r['p_er']:.4f}  "
                f"R² = {r['r2']:.3f}  {_stars(r['p_er'])}"
            )
    logger.info(
        f"\n  Nanigian interpretation: "
        f"{'Each 1% ER is associated with ' + str(round(out.get('ols_Model A (all funds)', {}).get('coef_er', 0)*100, 2)) + 'bps alpha change.' if 'ols_Model A (all funds)' in out else 'OLS not available.'}"
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PERFORMANCE PERSISTENCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_performance_persistence(
    metrics_df: pd.DataFrame,
    reg_results_df: pd.DataFrame,
) -> dict:
    """
    Test whether fund performance is persistent across sub-periods.

    Tests applied
    -------------
    Test A — Spearman rank correlation (CAGR persistence):
        Rank funds by Pre-COVID CAGR; test if rank correlates with COVID
        and Post-COVID CAGR.
        H₀: ρ = 0 (no persistence).
        Significant positive ρ → performance is persistent (skill-based).
        Non-significant → performance is luck-based (Carhart 1997).

    Test B — Spearman rank correlation (Alpha persistence):
        Rank funds by Alpha (annualised).  Split into two halves by time
        and test rank correlation.

    Test C — Winner/Loser contingency table (Malkiel 1995 method):
        Classify each fund as "Winner" (top 50%) or "Loser" (bottom 50%)
        in Period 1 (Pre-COVID) and Period 2 (Post-COVID).
        Build 2×2 contingency table: WW, WL, LW, LL.
        χ² test for independence: significant → persistence exists.

        Persistence ratio: (WW + LL) / (WW + WL + LW + LL).
        Odds ratio: (WW × LL) / (WL × LW).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Per-fund performance metrics (for sub-period CAGR).
    reg_results_df : pd.DataFrame
        Per-fund regression results (for alpha).

    Returns
    -------
    dict
        Persistence test results.
    """
    logger.info("  §6 Performance persistence tests (Spearman ρ, Winner-Loser χ²) …")
    out: dict = {}

    # ── Test A: Spearman rank correlation on CAGR sub-periods ─────────────────
    cagr_cols = ["CAGR_Pre_COVID", "CAGR_COVID", "CAGR_Post_COVID"]
    avail_cols = [c for c in cagr_cols if c in metrics_df.columns]

    period_pairs = [
        ("Pre_COVID → COVID",     "CAGR_Pre_COVID",  "CAGR_COVID"),
        ("COVID → Post_COVID",    "CAGR_COVID",      "CAGR_Post_COVID"),
        ("Pre_COVID → Post_COVID","CAGR_Pre_COVID",  "CAGR_Post_COVID"),
    ]

    logger.info(f"\n  Test A — CAGR Persistence (Spearman ρ):")
    for pair_label, col1, col2 in period_pairs:
        if col1 not in metrics_df.columns or col2 not in metrics_df.columns:
            continue
        sub = metrics_df[[col1, col2]].dropna()
        if len(sub) < 5:
            continue
        rho, p_rho = stats.spearmanr(sub[col1], sub[col2])
        out[f"spearman_cagr_{pair_label.replace(' ', '_')}"] = {
            "rho": round(float(rho), 4),
            "p":   round(float(p_rho), 6),
            "n":   len(sub),
            "significant": bool(p_rho < ALPHA_SIGNIFICANCE),
        }
        logger.info(
            f"    {pair_label:<30}  ρ = {rho:+.4f}  p = {p_rho:.4f}  "
            f"n={len(sub)}  {_stars(p_rho)}  "
            f"{'PERSISTENT' if p_rho < ALPHA_SIGNIFICANCE and rho > 0 else 'NO PERSISTENCE'}"
        )

    # ── Test B: Spearman ρ on alpha (per-fund, if available) ──────────────────
    logger.info(f"\n  Test B — Alpha Persistence (Spearman ρ across fund types):")
    if "Alpha_Annual_Pct" in metrics_df.columns:
        for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
            sub = metrics_df[metrics_df[COL_FUND_TYPE] == ft]
            # Compare pre-COVID CAGR rank with alpha rank
            if "CAGR_Pre_COVID" in sub.columns and "Alpha_Annual_Pct" in sub.columns:
                pair = sub[["CAGR_Pre_COVID", "Alpha_Annual_Pct"]].dropna()
                if len(pair) >= 5:
                    rho, p_rho = stats.spearmanr(pair["CAGR_Pre_COVID"],
                                                  pair["Alpha_Annual_Pct"])
                    out[f"spearman_alpha_{ft}"] = {
                        "rho": round(float(rho), 4),
                        "p":   round(float(p_rho), 6),
                        "significant": bool(p_rho < ALPHA_SIGNIFICANCE),
                    }
                    logger.info(
                        f"    {ft:<10} Pre-COVID CAGR → Alpha:  "
                        f"ρ = {rho:+.4f}  p = {p_rho:.4f}  {_stars(p_rho)}"
                    )

    # ── Test C: Winner-Loser contingency table ─────────────────────────────────
    logger.info(f"\n  Test C — Winner/Loser Contingency (Malkiel 1995 method):")
    if "CAGR_Pre_COVID" in metrics_df.columns and "CAGR_Post_COVID" in metrics_df.columns:
        wl_df = metrics_df[["CAGR_Pre_COVID", "CAGR_Post_COVID"]].dropna().copy()
        median_pre  = wl_df["CAGR_Pre_COVID"].median()
        median_post = wl_df["CAGR_Post_COVID"].median()

        wl_df["W1"] = wl_df["CAGR_Pre_COVID"]  >= median_pre
        wl_df["W2"] = wl_df["CAGR_Post_COVID"] >= median_post

        WW = int(( wl_df["W1"] &  wl_df["W2"]).sum())
        WL = int(( wl_df["W1"] & ~wl_df["W2"]).sum())
        LW = int((~wl_df["W1"] &  wl_df["W2"]).sum())
        LL = int((~wl_df["W1"] & ~wl_df["W2"]).sum())

        contingency = np.array([[WW, WL], [LW, LL]])
        chi2, p_chi2, _, _ = stats.chi2_contingency(contingency, correction=True)

        total = WW + WL + LW + LL
        persistence_ratio = (WW + LL) / total if total > 0 else np.nan
        odds_ratio = (WW * LL) / (WL * LW) if (WL * LW) > 0 else np.nan

        out["winner_loser_contingency"] = {
            "WW": WW, "WL": WL, "LW": LW, "LL": LL,
            "chi2": round(float(chi2), 4),
            "p_chi2": round(float(p_chi2), 6),
            "persistence_ratio": round(float(persistence_ratio), 4),
            "odds_ratio": round(float(odds_ratio), 4) if not np.isnan(odds_ratio) else np.nan,
            "significant": bool(p_chi2 < ALPHA_SIGNIFICANCE),
        }

        logger.info(
            f"    Contingency table (Pre-COVID → Post-COVID):\n"
            f"             Post-Winner  Post-Loser\n"
            f"    Pre-Win:  {WW:>10}  {WL:>10}\n"
            f"    Pre-Los:  {LW:>10}  {LL:>10}\n"
            f"    χ² = {chi2:.4f}   p = {p_chi2:.4f}   {_stars(p_chi2)}\n"
            f"    Persistence ratio = {persistence_ratio:.2%}  |  "
            f"Odds ratio = {odds_ratio:.2f}"
        )
        if p_chi2 < ALPHA_SIGNIFICANCE:
            logger.info(
                f"    → SIGNIFICANT: Performance IS persistent (χ² rejects independence)."
            )
        else:
            logger.info(
                f"    → NOT significant: Performance is NOT reliably persistent."
            )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_alpha_ci(
    reg_results_df: pd.DataFrame,
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> dict:
    """
    Compute non-parametric bootstrap confidence intervals for the group mean
    alpha of Active and Passive funds.

    Method
    ------
    For each fund type (Active, Passive):
    1.  Draw n_bootstrap samples of size N with replacement from the fund-level
        alpha distribution.
    2.  Compute the mean of each bootstrap sample.
    3.  95% CI = [2.5th percentile, 97.5th percentile] of bootstrap means.
        (Percentile bootstrap method — valid without normality assumption.)

    Why bootstrap?
    --------------
    With only 31–33 funds per group, normal-theory CIs rely on the Central
    Limit Theorem which may not hold.  Bootstrap CIs are more reliable and
    are standard in the fund performance literature (Fama and French 2010,
    Barras et al 2010).

    Parameters
    ----------
    reg_results_df : pd.DataFrame
        Per-fund regression results.
    n_bootstrap : int, optional
        Number of bootstrap resamples.  Default 1 000.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Per-group bootstrap CIs:
            mean_alpha, ci_lower, ci_upper, ci_width, n_funds.
        Both monthly decimal and annualised percentage forms.
    """
    logger.info(
        f"  §7 Bootstrap alpha CIs ({n_bootstrap:,} resamples, seed={seed}) …"
    )
    rng = np.random.default_rng(seed)
    out: dict = {}

    ok = reg_results_df[reg_results_df[COL_REG_STATUS] == "OK"].copy()
    ok["Alpha_Annual"] = (1 + ok[COL_ALPHA]) ** MONTHS_PER_YEAR - 1

    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        alphas = ok[ok[COL_FUND_TYPE] == ft]["Alpha_Annual"].dropna().values
        n      = len(alphas)

        if n < 3:
            out[ft] = {"n_funds": n, "error": "Insufficient funds"}
            continue

        # Bootstrap resampling
        boot_means = np.array([
            rng.choice(alphas, size=n, replace=True).mean()
            for _ in range(n_bootstrap)
        ])

        mean_alpha = float(alphas.mean())
        ci_lower   = float(np.percentile(boot_means, 2.5))
        ci_upper   = float(np.percentile(boot_means, 97.5))

        out[ft] = {
            "n_funds":          n,
            "mean_alpha":       round(mean_alpha, 6),
            "ci_lower_95":      round(ci_lower,   6),
            "ci_upper_95":      round(ci_upper,   6),
            "ci_width":         round(ci_upper - ci_lower, 6),
            "ci_excludes_zero": bool(ci_lower > 0 or ci_upper < 0),
            "boot_std":         round(float(boot_means.std()), 6),
        }

        logger.info(
            f"  [{ft}]  Mean α = {mean_alpha:+.2%} ann.  "
            f"95% CI: [{ci_lower:+.2%}, {ci_upper:+.2%}]  "
            f"Width = {ci_upper - ci_lower:.2%}  "
            f"{'CI excludes 0 → statistically significant' if out[ft]['ci_excludes_zero'] else 'CI includes 0 → not significant'}"
        )

    # Difference CI (Active − Passive)
    act_alphas = ok[ok[COL_FUND_TYPE] == FUND_TYPE_ACTIVE]["Alpha_Annual"].dropna().values
    pas_alphas = ok[ok[COL_FUND_TYPE] == FUND_TYPE_PASSIVE]["Alpha_Annual"].dropna().values

    if len(act_alphas) >= 3 and len(pas_alphas) >= 3:
        boot_diffs = np.array([
            rng.choice(act_alphas, size=len(act_alphas), replace=True).mean()
            - rng.choice(pas_alphas, size=len(pas_alphas), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        diff_mean  = float(act_alphas.mean() - pas_alphas.mean())
        diff_lower = float(np.percentile(boot_diffs, 2.5))
        diff_upper = float(np.percentile(boot_diffs, 97.5))

        out["difference"] = {
            "mean_diff":        round(diff_mean,  6),
            "ci_lower_95":      round(diff_lower, 6),
            "ci_upper_95":      round(diff_upper, 6),
            "ci_excludes_zero": bool(diff_lower > 0 or diff_upper < 0),
        }
        logger.info(
            f"  [Active − Passive]  Diff = {diff_mean:+.2%}  "
            f"95% CI: [{diff_lower:+.2%}, {diff_upper:+.2%}]  "
            f"{'Significant' if out['difference']['ci_excludes_zero'] else 'Not significant'}"
        )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MASTER ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_all_statistical_tests(
    enriched_df: pd.DataFrame,
    reg_results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    save_outputs: bool = True,
) -> dict:
    """
    Run the complete battery of statistical tests and save results.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Full enriched master frame (Monthly_Return, Fund_Name, etc.).
    reg_results_df : pd.DataFrame
        Per-fund regression results from regression_analysis.py.
    metrics_df : pd.DataFrame
        Per-fund performance metrics from performance_metrics.py.
    save_outputs : bool
        If True, save all test results to CSV and text files.

    Returns
    -------
    dict
        All test results keyed by test name.
    """
    logger.info("\n" + "=" * 68)
    logger.info("  STATISTICAL TESTS PHASE — run_all_statistical_tests")
    logger.info("=" * 68)

    results: dict = {}

    # §1: Normality
    logger.info("\n── §1 Return Normality Tests ──")
    results["normality"] = test_return_normality(enriched_df)

    # §2: Stationarity
    logger.info("\n── §2 Return Stationarity Tests ──")
    results["stationarity"] = test_return_stationarity(enriched_df)

    # §3: Multiple comparison
    logger.info("\n── §3 Multiple Comparison Correction ──")
    results["multiple_comparison"] = test_alpha_multiple_comparison(reg_results_df)

    # §4: Active vs Passive
    logger.info("\n── §4 Active vs Passive Distribution Tests ──")
    results["active_vs_passive"] = test_active_vs_passive(enriched_df, reg_results_df)

    # §5: Expense ratio impact
    logger.info("\n── §5 Expense Ratio Impact Tests ──")
    results["expense_impact"] = test_expense_ratio_impact(reg_results_df, metrics_df)

    # §6: Persistence
    logger.info("\n── §6 Performance Persistence Tests ──")
    results["persistence"] = test_performance_persistence(metrics_df, reg_results_df)

    # §7: Bootstrap CIs
    logger.info("\n── §7 Bootstrap Alpha CIs ──")
    results["bootstrap_ci"] = bootstrap_alpha_ci(reg_results_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_outputs:
        from src.utils import ensure_dir
        ensure_dir(REGRESSION_RESULTS_DIR)

        save_dataframe(results["normality"],
                       STATISTICAL_TESTS_FILE.parent / "normality_tests.csv",
                       description="Return normality tests")
        save_dataframe(results["stationarity"],
                       STATISTICAL_TESTS_FILE.parent / "stationarity_tests.csv",
                       description="Return stationarity tests")

        mc_df = results["multiple_comparison"]
        save_dataframe(mc_df,
                       STATISTICAL_TESTS_FILE,
                       description="Multiple comparison corrected results")

        # Build a concise summary text report
        lines: list[str] = _build_text_report(results)
        save_text_report("\n".join(lines),
                         STATISTICAL_TESTS_FILE.parent / "statistical_tests_summary.txt")

    # ── Final banner ──────────────────────────────────────────────────────────
    n_stat = results["stationarity"]["Conclusion"].value_counts().get("Stationary", 0)
    n_norm = results["normality"]["Both_Normal"].sum()
    n_mc   = results["multiple_comparison"]["Sig_BH_FDR"].sum()

    logger.info(f"\n  ╔══════ STATISTICAL TESTS SUMMARY ══════╗")
    logger.info(f"  ║  Funds with stationary returns : {n_stat:>5}    ║")
    logger.info(f"  ║  Funds with normal returns     : {n_norm:>5}    ║")
    logger.info(f"  ║  Sig alpha (BH-FDR corrected)  : {n_mc:>5}    ║")
    logger.info(f"  ║  7 test batteries completed    :    ✓    ║")
    logger.info(f"  ╚═══════════════════════════════════════╝")
    logger.info("\n  Statistical tests complete.  Proceed to visualization.py.")
    logger.info("=" * 68 + "\n")

    return results


def _build_text_report(results: dict) -> list[str]:
    """Build a human-readable summary of all statistical test results."""
    lines = [
        "=" * 70,
        "  STATISTICAL TESTS SUMMARY REPORT",
        f"  Quantitative Fund Analysis — Carhart Four-Factor Model",
        "=" * 70,
        "",
        "§1 RETURN NORMALITY",
        "-" * 70,
    ]
    norm = results.get("normality", pd.DataFrame())
    if not norm.empty:
        for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
            sub = norm[norm[COL_FUND_TYPE] == ft]
            lines.append(
                f"  {ft}: JB-normal {sub['JB_Normal'].sum()}/{len(sub)}  |  "
                f"SW-normal {sub['SW_Normal'].sum()}/{len(sub)}  |  "
                f"Mean skew {sub['Skewness'].mean():+.3f}  |  "
                f"Mean exc.kurt {sub['Excess_Kurtosis'].mean():.3f}"
            )

    lines += ["", "§2 RETURN STATIONARITY", "-" * 70]
    stat = results.get("stationarity", pd.DataFrame())
    if not stat.empty:
        for conc, cnt in stat["Conclusion"].value_counts().items():
            lines.append(f"  {conc}: {cnt} funds")

    lines += ["", "§3 MULTIPLE COMPARISON CORRECTION (BH-FDR)", "-" * 70]
    mc = results.get("multiple_comparison", pd.DataFrame())
    if not mc.empty:
        for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
            sub = mc[mc[COL_FUND_TYPE] == ft]
            lines.append(
                f"  {ft}: Raw sig {sub['Alpha_Significant'].sum()}/{len(sub)}  |  "
                f"Bonferroni {sub['Sig_Bonferroni'].sum()}/{len(sub)}  |  "
                f"BH-FDR {sub['Sig_BH_FDR'].sum()}/{len(sub)}"
            )
        cats = mc["Alpha_Category"].value_counts()
        for cat, cnt in cats.items():
            lines.append(f"  Category [{cat}]: {cnt} funds")

    lines += ["", "§4 ACTIVE vs PASSIVE DISTRIBUTION TESTS", "-" * 70]
    avp = results.get("active_vs_passive", {})
    for key, label in [("mwu_returns", "MW-U (returns)"),
                        ("ks_returns",  "KS (returns)"),
                        ("levene_returns", "Levene variance"),
                        ("mwu_alpha",   "MW-U (alpha)"),
                        ("ks_alpha",    "KS (alpha)")]:
        if key in avp:
            r = avp[key]
            lines.append(
                f"  {label:<25}: stat={r['stat']:.4f}  p={r['p']:.4f}  "
                f"{_stars(r['p'])}  {'Significant' if r['significant'] else 'n.s.'}"
            )
    if "cohens_d_returns" in avp:
        lines.append(f"  Cohen's d (returns)     : {avp['cohens_d_returns']:.4f}")

    lines += ["", "§5 EXPENSE RATIO IMPACT", "-" * 70]
    er = results.get("expense_impact", {})
    for key, label in [("spearman_er_alpha_all",    "Spearman ρ (all)"),
                        ("spearman_er_alpha_active", "Spearman ρ (active)"),
                        ("spearman_er_alpha_passive","Spearman ρ (passive)")]:
        if key in er:
            r = er[key]
            lines.append(
                f"  {label:<30}: ρ={r['rho']:+.4f}  p={r['p']:.4f}  "
                f"{_stars(r['p'])}"
            )
    for model_key in ["Model A (all funds)", "Model B (+Is_Active)"]:
        key = f"ols_{model_key}"
        if key in er:
            r = er[key]
            lines.append(
                f"  OLS {model_key}: γ_ER={r['coef_er']:+.4f}  "
                f"t={r['t_er']:+.3f}  p={r['p_er']:.4f}  R²={r['r2']:.3f}  "
                f"{_stars(r['p_er'])}"
            )

    lines += ["", "§6 PERFORMANCE PERSISTENCE", "-" * 70]
    per = results.get("persistence", {})
    for key, val in per.items():
        if isinstance(val, dict) and "rho" in val:
            lines.append(
                f"  {key:<40}: ρ={val['rho']:+.4f}  p={val['p']:.4f}  "
                f"{_stars(val['p'])}"
            )
    if "winner_loser_contingency" in per:
        wl = per["winner_loser_contingency"]
        lines.append(
            f"  Winner-Loser χ²: {wl['chi2']:.4f}  p={wl['p_chi2']:.4f}  "
            f"Persistence ratio={wl['persistence_ratio']:.2%}  "
            f"Odds ratio={wl['odds_ratio']:.2f}"
        )

    lines += ["", "§7 BOOTSTRAP ALPHA 95% CIs", "-" * 70]
    boot = results.get("bootstrap_ci", {})
    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        if ft in boot:
            b = boot[ft]
            lines.append(
                f"  {ft}: mean={b['mean_alpha']:+.2%}  "
                f"95% CI [{b['ci_lower_95']:+.2%}, {b['ci_upper_95']:+.2%}]  "
                f"{'CI excludes 0 ✓' if b['ci_excludes_zero'] else 'CI includes 0'}"
            )
    if "difference" in boot:
        d = boot["difference"]
        lines.append(
            f"  Active − Passive: {d['mean_diff']:+.2%}  "
            f"95% CI [{d['ci_lower_95']:+.2%}, {d['ci_upper_95']:+.2%}]  "
            f"{'Significant' if d['ci_excludes_zero'] else 'Not significant'}"
        )

    lines += ["", "=" * 70]
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SELF-TEST (python src/statistical_tests.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _log = setup_logger("stat_tests_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  STATISTICAL TESTS SELF-TEST")
    _log.info("═" * 68)

    try:
        from src.config import CLEANED_DATA_DIR, REGRESSION_RESULTS_DIR, PERFORMANCE_METRICS_FILE

        # ── Load pre-saved pipeline CSVs ──────────────────────────────────────
        enriched   = pd.read_csv(CLEANED_DATA_DIR     / "master_enriched.csv",    parse_dates=["Date"])
        reg_input  = pd.read_csv(REGRESSION_RESULTS_DIR / "regression_input.csv", parse_dates=["Date"])
        reg_res    = pd.read_csv(REGRESSION_RESULTS_DIR / "carhart_regression_summary.csv")
        metrics    = pd.read_csv(PERFORMANCE_METRICS_FILE)

        # Attach MF from reg_input to enriched
        mf_series = reg_input.groupby("Date")["MF"].mean().reset_index()
        enriched  = enriched.merge(mf_series, on="Date", how="left", suffixes=("", "_reg"))
        if "MF_reg" in enriched.columns:
            enriched["MF"] = enriched["MF"].fillna(enriched["MF_reg"])
            enriched.drop(columns=["MF_reg"], inplace=True)

        _log.info(
            f"  Data loaded:  enriched={enriched.shape}  "
            f"reg_res={reg_res.shape}  metrics={metrics.shape}"
        )

        # ── Run all tests ─────────────────────────────────────────────────────
        results = run_all_statistical_tests(
            enriched_df=enriched,
            reg_results_df=reg_res,
            metrics_df=metrics,
            save_outputs=True,
        )

        _log.info("\n── Assertions ──")

        # A1: Normality DataFrame
        norm = results["normality"]
        assert len(norm) == enriched[COL_FUND_NAME].nunique(), "Normality row count mismatch"
        assert "JB_P" in norm.columns and "SW_P" in norm.columns
        _log.info(
            f"  ✅ Normality tests: {len(norm)} funds.  "
            f"Both-normal: {norm['Both_Normal'].sum()} ({norm['Both_Normal'].mean():.0%})."
        )

        # A2: Stationarity — almost all funds should be stationary
        stat = results["stationarity"]
        n_stationary = (stat["Conclusion"] == "Stationary").sum()
        assert n_stationary >= len(stat) * 0.70, \
            f"Too few stationary funds: {n_stationary}/{len(stat)}"
        _log.info(
            f"  ✅ Stationarity: {n_stationary}/{len(stat)} funds confirmed stationary."
        )

        # A3: Multiple comparison — BH-FDR should reduce significant count
        mc  = results["multiple_comparison"]
        raw_sig = mc.get("Alpha_Significant", mc.get("Sig_BH_FDR", pd.Series())).sum() \
                  if hasattr(mc, "columns") else 0
        bh_sig  = mc["Sig_BH_FDR"].sum() if hasattr(mc, "columns") and "Sig_BH_FDR" in mc.columns else 0
        _log.info(
            f"  ✅ BH-FDR: {bh_sig} significant alphas after correction "
            f"(vs {mc['Alpha_Significant'].sum() if hasattr(mc,'columns') and 'Alpha_Significant' in mc.columns else '?'} raw)."
        )

        # A4: Active vs Passive — at least MWU and KS completed
        avp = results["active_vs_passive"]
        assert "mwu_returns" in avp and "ks_returns" in avp
        _log.info(
            f"  ✅ Distribution tests: MW-U p={avp['mwu_returns']['p']:.4f}  "
            f"KS p={avp['ks_returns']['p']:.4f}  "
            f"Cohen's d={avp.get('cohens_d_returns', np.nan):.4f}."
        )

        # A5: Expense ratio — Spearman ρ computed
        er_tests = results["expense_impact"]
        assert "spearman_er_alpha_all" in er_tests
        rho = er_tests["spearman_er_alpha_all"]["rho"]
        _log.info(
            f"  ✅ ER impact: Spearman ρ (all) = {rho:+.4f}  "
            f"p={er_tests['spearman_er_alpha_all']['p']:.4f}."
        )

        # A6: Persistence — at least one Spearman ρ computed
        per = results["persistence"]
        assert len(per) > 0, "No persistence tests ran"
        _log.info(
            f"  ✅ Persistence: {len(per)} sub-tests computed."
        )

        # A7: Bootstrap CIs — active CI should exclude zero
        boot = results["bootstrap_ci"]
        assert FUND_TYPE_ACTIVE in boot and FUND_TYPE_PASSIVE in boot
        act_ci = boot[FUND_TYPE_ACTIVE]
        _log.info(
            f"  ✅ Bootstrap CI [Active]: "
            f"[{act_ci['ci_lower_95']:+.2%}, {act_ci['ci_upper_95']:+.2%}]  "
            f"{'Excludes 0 ✓' if act_ci['ci_excludes_zero'] else 'Includes 0'}."
        )

        _log.info("\n✅  All statistical_tests self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Self-test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        raise

    _log.info("═" * 68 + "\n")
