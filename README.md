# Carhart Four-Factor Fund Analysis

> **Research-grade quantitative finance pipeline** comparing Indian active
> and passive large-cap mutual funds (2013–2026) using the Carhart (1997)
> Four-Factor Model.
>
> Inspired by: *"The Historical Record on Active vs Passive Mutual Fund
> Performance"* — David Nanigian (2019).

---

## Overview

This project investigates whether Indian active large-cap direct-plan mutual
funds generate statistically significant risk-adjusted excess returns
(Jensen's alpha) relative to passive index funds, and whether the
performance gap is explained by expense ratio differences (the Nanigian
hypothesis).

**Key findings from the live data (2013–2026, 66 funds):**

| Metric | Active (33 funds) | Passive (33 funds) |
|---|---|---|
| Mean CAGR | **13.49%** | 7.92% |
| Mean Sharpe Ratio | **0.502** | 0.315 |
| Mean Alpha (ann.) | **+3.26%** | +0.12% |
| Mean Max Drawdown | **−24.4%** | −39.1% |
| Up-Capture vs MF | **121%** | 113% |
| Down-Capture vs MF | **73%** ← lower = better | 84% |
| Mean Expense Ratio | 1.21% | 0.13% |
| Gross Alpha (α + ER) | **+4.45%** | +0.25% |

**Welch t-test** on alpha: `t = 5.20, p < 0.001` → Active alpha significantly
higher. **Bootstrap 95% CI** for Active − Passive alpha: `[+2.12%, +4.32%]`
(excludes zero). **BH-FDR corrected**: 29/64 funds have significant positive
alpha after multiple-comparison correction.

> ⚠️ **Caveat**: Unlike the US evidence (Nanigian 2019), Indian active
> large-cap managers appear to generate genuine gross alpha over this period.
> This likely reflects India's market-inefficiency premium and the relatively
> nascent passive fund universe. Results are sensitive to sample period and
> fund universe construction.

---

## Project Structure

```
quant_fund_analysis/
├── data/                             # Raw input datasets (not tracked in git)
│   ├── monthly_nav_with_returns.csv  # 33 passive funds — NAV history
│   ├── all_active_large_cap_funds_with_returns.csv  # 33 active funds
│   ├── fund_expense_ratios.xlsx      # Annual expense ratios
│   └── factor_data.csv               # Fama-French-style factors (sparse)
│
├── src/                              # Source modules (clean pipeline)
│   ├── config.py                     # All constants, paths, column names
│   ├── utils.py                      # Shared helpers: logging, math, I/O
│   ├── data_loader.py                # Typed CSV/XLSX loaders + schema validation
│   ├── preprocessing.py              # Merge, clean, winsorise, date normalise
│   ├── return_calculations.py        # Simple/log/excess/cumulative/rolling returns
│   ├── factor_merge.py               # Synthetic factor construction + merge
│   ├── regression_analysis.py        # Carhart OLS + HAC + pooled + diagnostics
│   ├── performance_metrics.py        # 41 metrics: Sharpe/Sortino/Calmar/VaR/etc.
│   ├── statistical_tests.py          # 7 test batteries (BH-FDR, bootstrap, etc.)
│   └── visualization.py              # 12 publication-quality charts
│
├── outputs/
│   ├── cleaned_data/                 # Intermediate DataFrames
│   ├── regression_results/           # All CSVs: regression, metrics, stat tests
│   ├── figures/                      # PNG charts (300 dpi)
│   └── reports/                      # Text summary reports
│
├── main.py                           # 8-stage pipeline orchestrator
├── run_project.py                    # Preflight checks + launch
└── requirements.txt                  # Python dependencies
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_project.py
```

This runs all 8 stages and saves outputs to `outputs/`.

### 3. Fast-resume (skip already-completed stages)

```bash
# Run only from Stage 6 (metrics) onward — uses saved CSVs from Stages 1–5
python main.py --stage 6

# Skip visualization (faster for stat analysis only)
python main.py --no-plots

# Validate data without writing any output
python main.py --dry-run
```

### 4. Run individual modules (self-tests)

Each module has a built-in `__main__` self-test with assertions:

```bash
PYTHONPATH="." python3 src/data_loader.py
PYTHONPATH="." python3 src/preprocessing.py
PYTHONPATH="." python3 src/return_calculations.py
PYTHONPATH="." python3 src/factor_merge.py
PYTHONPATH="." python3 src/regression_analysis.py
PYTHONPATH="." python3 src/performance_metrics.py
PYTHONPATH="." python3 src/statistical_tests.py
PYTHONPATH="." python3 src/visualization.py
```

---

## Pipeline Stages

| Stage | Module | Key Outputs |
|---|---|---|
| **1** | `data_loader.py` | Raw DataFrames (4 datasets) |
| **2** | `preprocessing.py` | `master_fund_data.csv`, expense-ratio merged |
| **3** | `return_calculations.py` | `master_enriched.csv`, rolling returns, fund summary |
| **4** | `factor_merge.py` | `synthetic_factors.csv`, `regression_input.csv` |
| **5** | `regression_analysis.py` | `carhart_regression_summary.csv`, pooled OLS, diagnostics |
| **6** | `performance_metrics.py` | `performance_metrics.csv` (41 metrics × 66 funds) |
| **7** | `statistical_tests.py` | `statistical_tests.csv`, BH-FDR, bootstrap CIs |
| **8** | `visualization.py` | 12 × 300 dpi PNG charts in `outputs/figures/` |

---

## Methodology

### Data

- **Passive funds**: 33 Nifty 50 / BSE Sensex ETFs and index funds
  (2013–2026, direct plans, monthly NAV)
- **Active funds**: 33 large-cap active direct-plan mutual funds
  (2013–2026, monthly NAV)
- **Risk-free rate**: Derived from RBI repo rate history (time-varying)
- **Market factor**: Equal-weighted average monthly return of passive funds
  (MF = Rm − Rf)
- **Momentum factor**: Cross-sectional WML — winner minus loser passive funds

> **Note on factors**: The provided `factor_data.csv` covers only
> Oct–Dec 1993 (no overlap with the 2013–2026 NAV data). All four
> factors are therefore **synthetic**. SMB and HML are zero (constant)
> in this large-cap universe — they are automatically excluded from
> the OLS design matrix.

### Carhart (1997) Four-Factor Model

```
R_it − RF_t = α_i + β₁·MF_t + β₂·SMB_t + β₃·HML_t + β₄·WML_t + ε_it
```

Estimated fund-by-fund with **Newey-West HAC standard errors** (4 lags)
to correct for heteroskedasticity and autocorrelation in monthly panels.

### Multiple Comparison Correction

With 64 simultaneous alpha tests, raw p-values overstate significance.
Two corrections are applied:

| Method | Controls | Conservative? |
|---|---|---|
| Bonferroni | FWER (≤1 false discovery) | Very |
| **BH-FDR** (used for main inference) | Expected false discovery rate | Moderate |

### Expense Ratio Impact (Nanigian 2019)

Cross-sectional OLS:  `α_i = γ₀ + γ₁·ER_i + ε_i`

If γ₁ ≈ −1, each 1% expense ratio costs 1% alpha (a one-for-one drag).

### Performance Persistence

- **Spearman ρ** across sub-periods (Pre-COVID → COVID → Post-COVID)
- **Malkiel (1995) Winner-Loser contingency table** with χ² test

---

## Output Files

### CSVs (`outputs/regression_results/`)

| File | Description |
|---|---|
| `carhart_regression_summary.csv` | Per-fund OLS: α, t, β, R², HAC SEs |
| `regression_results_formatted.csv` | Publication table with *** significance |
| `alpha_comparison.csv` | Welch t-test: Active vs Passive alpha |
| `pooled_regression_summaries.txt` | Nanigian panel OLS (Models A & B) |
| `performance_metrics.csv` | 41 metrics × 66 funds |
| `group_performance_summary.csv` | Active vs Passive mean/median/std |
| `ranking_by_sharpe.csv` | Top-20 funds by Sharpe |
| `ranking_by_alpha.csv` | Top-20 funds by alpha |
| `statistical_tests.csv` | BH-FDR / Bonferroni corrected p-values |
| `normality_tests.csv` | JB + Shapiro-Wilk per fund |
| `stationarity_tests.csv` | ADF + KPSS per fund |
| `statistical_tests_summary.txt` | Human-readable summary of all 7 test batteries |

### Charts (`outputs/figures/`)

| File | Description |
|---|---|
| `01_wealth_index_comparison.png` | NAV wealth index curves (Active vs Passive) |
| `02_return_distributions.png` | Violin + box plots of monthly returns |
| `03_alpha_distribution.png` | Alpha histograms with BH-FDR colouring |
| `04_factor_time_series.png` | MF, WML, RF time series |
| `05_sharpe_ratio_heatmap.png` | Sharpe ratio bar chart (all 66 funds) |
| `06_er_vs_alpha_scatter.png` | Expense ratio vs alpha scatter (Nanigian test) |
| `07_risk_return_scatter.png` | Risk-return bubble chart |
| `08_capture_ratios.png` | Up/Down capture vs market |
| `09_sub_period_performance.png` | Pre-COVID / COVID / Post-COVID CAGR bars |
| `10_drawdown_comparison.png` | Max drawdown (all funds) |
| `11_rolling_sharpe.png` | 24-month rolling Sharpe ratio |
| `12_alpha_significance_grid.png` | Forest plot: alpha + 95% CI, BH-FDR |

---

## Design Principles

- **Modularity**: Each module has a single responsibility and a self-contained `__main__` test.
- **No magic strings**: All column names and file paths are centralised in `config.py`.
- **Reproducibility**: Bootstrap uses a fixed seed (42). All outputs are deterministic.
- **Robustness**: HAC SEs, BH-FDR correction, bootstrap CIs — not just raw OLS p-values.
- **No leakage**: All sub-period analyses use only data from within that period.

---

## References

1. Carhart, M. M. (1997). *On persistence in mutual fund performance.* Journal of Finance, 52(1), 57–82.
2. Nanigian, D. (2019). *The historical record on active vs passive mutual fund performance.* Financial Analysts Journal, 75(3), 15–30.
3. Fama, E. F., & French, K. R. (2010). *Luck versus skill in the cross-section of mutual fund returns.* Journal of Finance, 65(5), 1915–1947.
4. Barras, L., Scaillet, O., & Wermers, R. (2010). *False discoveries in mutual fund performance.* Journal of Finance, 65(1), 179–216.
5. Malkiel, B. G. (1995). *Returns from investing in equity mutual funds 1971 to 1991.* Journal of Finance, 50(2), 549–572.
6. Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery rate.* Journal of the Royal Statistical Society, Series B, 57(1), 289–300.
