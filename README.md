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


> ⚠️ **Caveat**: Unlike the US evidence (Nanigian 2019), Indian active
> large-cap managers appear to generate genuine gross alpha over this period.
> This likely reflects India's market-inefficiency premium and the relatively
> nascent passive fund universe. Results are sensitive to sample period and
> fund universe construction.

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
---

## References

1. Carhart, M. M. (1997). *On persistence in mutual fund performance.* Journal of Finance, 52(1), 57–82.
2. Nanigian, D. (2019). *The historical record on active vs passive mutual fund performance.* Financial Analysts Journal, 75(3), 15–30.
3. Fama, E. F., & French, K. R. (2010). *Luck versus skill in the cross-section of mutual fund returns.* Journal of Finance, 65(5), 1915–1947.
4. Barras, L., Scaillet, O., & Wermers, R. (2010). *False discoveries in mutual fund performance.* Journal of Finance, 65(1), 179–216.
5. Malkiel, B. G. (1995). *Returns from investing in equity mutual funds 1971 to 1991.* Journal of Finance, 50(2), 549–572.
6. Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery rate.* Journal of the Royal Statistical Society, Series B, 57(1), 289–300.
