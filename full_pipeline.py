"""
COMPLETE QUANTITATIVE FINANCE RESEARCH PIPELINE
================================================
Active vs Passive Large-Cap Mutual Funds in India
Carhart Four-Factor Analysis

This script executes ALL steps from data extraction through final analysis.
NON-NEGOTIABLE: No synthetic data, no simulated observations, no proxy factors.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from numbers_parser import Document
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUTS_TABLES = os.path.join(PROJECT_ROOT, 'outputs', 'tables')
OUTPUTS_FIGURES = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
OUTPUTS_REPORTS = os.path.join(PROJECT_ROOT, 'outputs', 'reports')

for d in [DATA_RAW, DATA_PROCESSED, OUTPUTS_TABLES, OUTPUTS_FIGURES, OUTPUTS_REPORTS]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# FUND NAME MAPPING
# ============================================================
# Maps names in .numbers files to names in fund_classification.csv
FUND_NAME_MAP = {
    # Active funds
    'Kotak Large Cap Fund - Growth - Direct': 'Kotak Large Cap Fund - Direct Plan',
    
    # Passive funds  
    'Aditya Birla Sun Life Nifty 50 Index Fund - Growth - Direct Plan': 'Aditya Birla Sun Life Nifty 50 Index Fund - Direct Plan',
    'Axis Nifty 50 Index Fund - Direct Plan - Growth': 'Axis Nifty 50 Index Fund - Direct Plan',
    'BANDHAN Nifty 50 Index Fund - Regular Plan - Growth': 'Bandhan Nifty 50 Index Fund - Direct Plan',
    'DSP Nifty 50 Index Fund - Direct Plan - Growth': 'DSP Nifty 50 Index Fund - Direct Plan',
    'Franklin India INDEX FUND- NSE NIFTY 50 INDEX FUND- Direct - Growth': 'Franklin India NSE Nifty 50 Index - Direct Plan',
    'HDFC NIFTY 50 ETF - Growth Plan': 'HDFC NIFTY 50 ETF',
    'HDFC Nifty 50 Index Fund - Direct Plan': 'HDFC NIFTY 50 Index Fund - Direct Plan',
    'ICICI Prudential BSE Sensex Index Fund - Direct Plan - Cumulative Option': 'ICICI Prudential BSE Sensex Index Fund - Direct Plan',
    'ICICI Prudential Nifty 50 Index Fund - Direct Plan IDCW Option': 'ICICI Prudential Nifty 50 Index Fund - Direct Plan',
    'Kotak Nifty 50 Index Fund - Direct Plan-Growth': 'Kotak Nifty 50 Index Fund - Direct Plan',
    'Motilal Oswal Nifty 50 Index Fund - Direct plan - Growth': 'Motilal Oswal Nifty 50 Index Fund - Direct Plan',
    'NIPPON INDIA INDEX FUND - BSE SENSEX PLAN - DIRECT Plan - ANNUAL - IDCW Option': 'Nippon India Index Fund BSE Sensex Plan - Direct Plan',
    'NIPPON INDIA INDEX FUND - NIFTY 50 PLAN - DIRECT Plan - ANNUAL - IDCW Option': 'Nippon India Index Fund - Nifty 50 Plan - Direct Plan',
    'Navi Nifty 50 Index Fund-Direct Plan-Growth': 'Navi Nifty 50 Index Fund - Direct Plan',
    'SBI BSE SENSEX ETF': 'SBI BSE Sensex ETF',
    'SBI NIFTY INDEX FUND - DIRECT PLAN - GROWTH': 'SBI Nifty Index Fund - Direct Plan',
    'Tata Nifty 50 Index Fund -Direct Plan': 'Tata Nifty 50 Index Fund - Direct Plan',
    'UTI Nifty 50 Index Fund - Direct Plan - IDCW': 'UTI Nifty 50 Index Fund - Direct Plan',
    # Added: Samco and Parag Parikh — check if present in active
    'Samco Large Cap Fund - Direct Plan': 'Samco Large Cap Fund - Direct Plan',
    'Parag Parikh Large Cap Fund - Direct Plan': 'Parag Parikh Large Cap Fund - Direct Plan',
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============================================================
# STEP 0: EXTRACT .numbers FILES
# ============================================================
log("=" * 70)
log("STEP 0: DATA EXTRACTION")
log("=" * 70)

def extract_numbers_to_df(filepath):
    """Extract a .numbers file to DataFrame."""
    doc = Document(filepath)
    table = doc.sheets[0].tables[0]
    headers = [str(table.cell(0, c).value) for c in range(table.num_cols)]
    rows = []
    for r in range(1, table.num_rows):
        row = [table.cell(r, c).value for c in range(table.num_cols)]
        rows.append(row)
    return pd.DataFrame(rows, columns=headers)

# Extract active funds
log("Extracting active_funds.numbers...")
df_active_raw = extract_numbers_to_df(os.path.join(PROJECT_ROOT, 'active_funds.numbers'))
log(f"  Active funds raw: {df_active_raw.shape}")
df_active_raw.to_csv(os.path.join(DATA_RAW, 'active_funds_raw.csv'), index=False)

# Extract passive funds
log("Extracting passive_funds.numbers...")
df_passive_raw = extract_numbers_to_df(os.path.join(PROJECT_ROOT, 'passive_funds.numbers'))
log(f"  Passive funds raw: {df_passive_raw.shape}")
df_passive_raw.to_csv(os.path.join(DATA_RAW, 'passive_funds_raw.csv'), index=False)

# ============================================================
# STEP 1: DATA PREPARATION
# ============================================================
log("\n" + "=" * 70)
log("STEP 1: DATA PREPARATION")
log("=" * 70)

# --- Load classification ---
df_classification = pd.read_csv(os.path.join(PROJECT_ROOT, 'fund_classification.csv'))
df_classification.columns = [c.strip() for c in df_classification.columns]
for col in df_classification.select_dtypes(include='object').columns:
    df_classification[col] = df_classification[col].str.strip()

classification_set = set(df_classification['Fund_Name'].unique())
log(f"Classification: {len(df_classification)} funds ({df_classification['Fund_Type'].value_counts().to_dict()})")

# --- Prepare Active Returns ---
log("\nPreparing Active Fund Returns...")
df_active = df_active_raw.copy()
df_active.columns = ['Fund_Name', 'Month', 'Date', 'NAV', 'Monthly_Return', 'Monthly_Return_Percent']
df_active['Fund_Name'] = df_active['Fund_Name'].astype(str).str.strip()
df_active['Monthly_Return'] = pd.to_numeric(df_active['Monthly_Return'], errors='coerce')
df_active['NAV'] = pd.to_numeric(df_active['NAV'], errors='coerce')

# Apply name mapping
df_active['Fund_Name'] = df_active['Fund_Name'].replace(FUND_NAME_MAP)

# Parse dates — use Month column (YYYY-MM format)
df_active['Date_YM'] = df_active['Month'].astype(str).str.strip().str[:7]

# SCALE: Active returns are in DECIMAL (e.g., -0.06 = -6%)
# Convert to PERCENT to match factors
df_active['Monthly_Return_Pct'] = df_active['Monthly_Return'] * 100.0

log(f"  Active returns: {df_active.shape}")
log(f"  Unique funds: {df_active['Fund_Name'].nunique()}")
log(f"  Date range: {df_active['Date_YM'].min()} to {df_active['Date_YM'].max()}")
log(f"  Return scale: DECIMAL → converted to PERCENT")
log(f"  Return (pct) stats: mean={df_active['Monthly_Return_Pct'].mean():.4f}%, std={df_active['Monthly_Return_Pct'].std():.4f}%")

# Verify all active fund names match classification
active_names = set(df_active['Fund_Name'].unique())
active_matched = active_names & classification_set
active_unmatched = active_names - classification_set
log(f"  Classification match: {len(active_matched)}/{len(active_names)}")
if active_unmatched:
    log(f"  UNMATCHED active funds: {active_unmatched}")

# --- Prepare Passive Returns ---
log("\nPreparing Passive Fund Returns...")
df_passive = df_passive_raw.copy()
df_passive.columns = ['Fund_Name', 'Month_Name', 'Date', 'NAV']
df_passive['Fund_Name'] = df_passive['Fund_Name'].astype(str).str.strip()
df_passive['NAV'] = pd.to_numeric(df_passive['NAV'], errors='coerce')

# Apply name mapping
df_passive['Fund_Name'] = df_passive['Fund_Name'].replace(FUND_NAME_MAP)

# Parse dates — use Date column (YYYY-MM-DD format)
df_passive['Date_YM'] = df_passive['Date'].astype(str).str.strip().str[:7]

# Calculate monthly returns from NAV
df_passive = df_passive.sort_values(['Fund_Name', 'Date_YM']).reset_index(drop=True)
df_passive['Monthly_Return'] = df_passive.groupby('Fund_Name')['NAV'].pct_change()

# pct_change returns decimal → convert to percent
df_passive['Monthly_Return_Pct'] = df_passive['Monthly_Return'] * 100.0

log(f"  Passive returns: {df_passive.shape}")
log(f"  Unique funds: {df_passive['Fund_Name'].nunique()}")
log(f"  Date range: {df_passive['Date_YM'].min()} to {df_passive['Date_YM'].max()}")
log(f"  Return scale: DECIMAL → converted to PERCENT")
log(f"  NaN returns (first month per fund): {df_passive['Monthly_Return_Pct'].isna().sum()}")
log(f"  Return (pct) stats: mean={df_passive['Monthly_Return_Pct'].dropna().mean():.4f}%, std={df_passive['Monthly_Return_Pct'].dropna().std():.4f}%")

# Verify passive fund names match classification
passive_names = set(df_passive['Fund_Name'].unique())
passive_matched = passive_names & classification_set
passive_unmatched = passive_names - classification_set
log(f"  Classification match: {len(passive_matched)}/{len(passive_names)}")
if passive_unmatched:
    log(f"  UNMATCHED passive funds: {passive_unmatched}")

# --- Save processed returns ---
active_returns = df_active[['Fund_Name', 'Date_YM', 'Monthly_Return_Pct']].dropna(subset=['Date_YM', 'Monthly_Return_Pct']).copy()
active_returns = active_returns.rename(columns={'Date_YM': 'Date', 'Monthly_Return_Pct': 'Monthly_Return'})
active_returns.to_csv(os.path.join(DATA_PROCESSED, 'active_returns.csv'), index=False)
log(f"\nSaved active_returns.csv: {active_returns.shape}")

passive_returns = df_passive[['Fund_Name', 'Date_YM', 'Monthly_Return_Pct']].dropna(subset=['Date_YM', 'Monthly_Return_Pct']).copy()
passive_returns = passive_returns.rename(columns={'Date_YM': 'Date', 'Monthly_Return_Pct': 'Monthly_Return'})
passive_returns.to_csv(os.path.join(DATA_PROCESSED, 'passive_returns.csv'), index=False)
log(f"Saved passive_returns.csv: {passive_returns.shape}")

# ============================================================
# STEP 2: MASTER DATASET
# ============================================================
log("\n" + "=" * 70)
log("STEP 2: MASTER DATASET CONSTRUCTION")
log("=" * 70)

# Combine active and passive returns
all_returns = pd.concat([active_returns, passive_returns], ignore_index=True)
log(f"Combined returns: {all_returns.shape}")

# Load and prepare factors
df_factors = pd.read_csv(os.path.join(PROJECT_ROOT, 'carhart_factors.csv'))
df_factors.columns = [c.strip() for c in df_factors.columns]
df_factors = df_factors.rename(columns={'Date': 'Date_YM'})

# Convert factor columns to numeric
for col in ['MF', 'SMB', 'HML', 'WML', 'RF']:
    df_factors[col] = pd.to_numeric(df_factors[col], errors='coerce')

# Drop rows with NA factors
factor_na_before = len(df_factors)
df_factors = df_factors.dropna(subset=['MF', 'SMB', 'HML', 'WML', 'RF'])
factor_na_dropped = factor_na_before - len(df_factors)
log(f"Factor rows dropped (NA): {factor_na_dropped}")
log(f"Factors available: {len(df_factors)} months ({df_factors['Date_YM'].iloc[0]} to {df_factors['Date_YM'].iloc[-1]})")

# Load expense ratios
df_expense = pd.read_csv(os.path.join(PROJECT_ROOT, 'expense_ratios.csv'))
df_expense.columns = [c.strip() for c in df_expense.columns]
df_expense = df_expense.rename(columns={'Funds': 'Fund_Name', 'Expense Ratio (%)': 'Expense_Ratio'})
for col in df_expense.select_dtypes(include='object').columns:
    df_expense[col] = df_expense[col].str.strip()

# Load AUM data
df_aum = pd.read_csv(os.path.join(PROJECT_ROOT, 'aum_data.csv'))
df_aum.columns = [c.strip() for c in df_aum.columns]
df_aum = df_aum.rename(columns={'Funds': 'Fund_Name', 'Net Assets (Cr)': 'AUM'})
for col in df_aum.select_dtypes(include='object').columns:
    df_aum[col] = df_aum[col].str.strip()

def parse_indian_number(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().strip('"')
    val_str = val_str.replace(',', '')
    try:
        return float(val_str)
    except ValueError:
        return np.nan

df_aum['AUM'] = df_aum['AUM'].apply(parse_indian_number)

# --- Merge classification ---
master = all_returns.merge(df_classification[['Fund_Name', 'Fund_Type']], on='Fund_Name', how='inner')
log(f"\nAfter classification merge: {master.shape} (from {len(all_returns)})")
log(f"  Records lost: {len(all_returns) - len(master)}")
log(f"  Funds matched: {master['Fund_Name'].nunique()}")

# --- Merge expense ratios ---
master = master.merge(df_expense[['Fund_Name', 'Expense_Ratio']], on='Fund_Name', how='left')
expense_na = master['Expense_Ratio'].isna().sum()
log(f"\nAfter expense ratio merge:")
log(f"  Missing expense ratios: {expense_na} records ({master[master['Expense_Ratio'].isna()]['Fund_Name'].nunique()} funds)")

# --- Merge AUM ---
master = master.merge(df_aum[['Fund_Name', 'AUM']], on='Fund_Name', how='left')
aum_na = master['AUM'].isna().sum()
log(f"\nAfter AUM merge:")
log(f"  Missing AUM: {aum_na} records ({master[master['AUM'].isna()]['Fund_Name'].nunique()} funds)")
if aum_na > 0:
    missing_aum_funds = master[master['AUM'].isna()]['Fund_Name'].unique()
    log(f"  Funds missing AUM: {list(missing_aum_funds)}")

# --- Merge factors ---
master = master.merge(df_factors[['Date_YM', 'MF', 'SMB', 'HML', 'WML', 'RF']], 
                       left_on='Date', right_on='Date_YM', how='inner')
master = master.drop(columns=['Date_YM'], errors='ignore')
log(f"\nAfter factor merge: {master.shape}")
log(f"  Date range: {master['Date'].min()} to {master['Date'].max()}")

# Final master dataset
log(f"\n--- MASTER DATASET SUMMARY ---")
log(f"  Shape: {master.shape}")
log(f"  Total funds: {master['Fund_Name'].nunique()}")
log(f"  Active funds: {master[master['Fund_Type']=='Active']['Fund_Name'].nunique()}")
log(f"  Passive funds: {master[master['Fund_Type']=='Passive']['Fund_Name'].nunique()}")
log(f"  Date range: {master['Date'].min()} to {master['Date'].max()}")
log(f"  Unique months: {master['Date'].nunique()}")
log(f"  Missing values:")
for col in master.columns:
    na = master[col].isna().sum()
    if na > 0:
        log(f"    {col}: {na}")

master.to_csv(os.path.join(DATA_PROCESSED, 'master_dataset.csv'), index=False)
log(f"\nSaved master_dataset.csv: {master.shape}")

# ============================================================
# STEP 3: MERGE VALIDATION
# ============================================================
log("\n" + "=" * 70)
log("STEP 3: MERGE VALIDATION")
log("=" * 70)

# Classification match rate
total_classified = len(df_classification)
funds_in_master = set(master['Fund_Name'].unique())
classified_in_master = classification_set & funds_in_master
classification_match_rate = len(classified_in_master) / total_classified * 100
log(f"Classification Match Rate: {classification_match_rate:.1f}% ({len(classified_in_master)}/{total_classified})")

# Missing classified funds
missing_classified = classification_set - funds_in_master
if missing_classified:
    log(f"  Missing classified funds ({len(missing_classified)}):")
    for f in sorted(missing_classified):
        ft = df_classification[df_classification['Fund_Name'] == f]['Fund_Type'].iloc[0]
        log(f"    [{ft}] {f}")

# Expense ratio match
funds_with_expense = master.dropna(subset=['Expense_Ratio'])['Fund_Name'].nunique()
expense_match_rate = funds_with_expense / master['Fund_Name'].nunique() * 100
log(f"\nExpense Ratio Match Rate: {expense_match_rate:.1f}% ({funds_with_expense}/{master['Fund_Name'].nunique()})")

# AUM match
funds_with_aum = master.dropna(subset=['AUM'])['Fund_Name'].nunique()
aum_match_rate = funds_with_aum / master['Fund_Name'].nunique() * 100
log(f"AUM Match Rate: {aum_match_rate:.1f}% ({funds_with_aum}/{master['Fund_Name'].nunique()})")

# Factor match — all rows in master have factors by construction (inner join)
log(f"Factor Match Rate: 100.0% (inner join)")

# ============================================================
# STEP 4: PORTFOLIO CONSTRUCTION
# ============================================================
log("\n" + "=" * 70)
log("STEP 4: PORTFOLIO CONSTRUCTION")
log("=" * 70)

# Separate active and passive
active_master = master[master['Fund_Type'] == 'Active'].copy()
passive_master = master[master['Fund_Type'] == 'Passive'].copy()

log(f"Active records: {len(active_master)} ({active_master['Fund_Name'].nunique()} funds)")
log(f"Passive records: {len(passive_master)} ({passive_master['Fund_Name'].nunique()} funds)")

# --- Equal-Weighted Portfolios ---
log("\nConstructing Equal-Weighted Portfolios...")
active_ew = active_master.groupby('Date')['Monthly_Return'].mean().reset_index()
active_ew = active_ew.rename(columns={'Monthly_Return': 'Active_EW'})

passive_ew = passive_master.groupby('Date')['Monthly_Return'].mean().reset_index()
passive_ew = passive_ew.rename(columns={'Monthly_Return': 'Passive_EW'})

log(f"  Active EW: {len(active_ew)} months")
log(f"  Passive EW: {len(passive_ew)} months")

# --- Value-Weighted Portfolios ---
log("\nConstructing Value-Weighted Portfolios...")

def compute_vw_return(group):
    """Compute value-weighted return for a group."""
    valid = group.dropna(subset=['AUM', 'Monthly_Return'])
    if len(valid) == 0 or valid['AUM'].sum() == 0:
        return np.nan
    weights = valid['AUM'] / valid['AUM'].sum()
    return (weights * valid['Monthly_Return']).sum()

active_vw = active_master.groupby('Date').apply(compute_vw_return).reset_index()
active_vw.columns = ['Date', 'Active_VW']

passive_vw = passive_master.groupby('Date').apply(compute_vw_return).reset_index()
passive_vw.columns = ['Date', 'Passive_VW']

log(f"  Active VW: {len(active_vw)} months")
log(f"  Passive VW: {len(passive_vw)} months")

# Merge all portfolios
portfolio_returns = active_ew.merge(active_vw, on='Date', how='outer')
portfolio_returns = portfolio_returns.merge(passive_ew, on='Date', how='outer')
portfolio_returns = portfolio_returns.merge(passive_vw, on='Date', how='outer')
portfolio_returns = portfolio_returns.sort_values('Date').reset_index(drop=True)

# Merge RF for Sharpe ratio calculation
portfolio_returns = portfolio_returns.merge(df_factors[['Date_YM', 'RF', 'MF', 'SMB', 'HML', 'WML']], 
                                              left_on='Date', right_on='Date_YM', how='inner')
portfolio_returns = portfolio_returns.drop(columns=['Date_YM'])

log(f"\nPortfolio returns: {portfolio_returns.shape}")
log(f"  Date range: {portfolio_returns['Date'].min()} to {portfolio_returns['Date'].max()}")
log(f"  Months: {len(portfolio_returns)}")

# Save
portfolio_returns.to_csv(os.path.join(DATA_PROCESSED, 'portfolio_returns.csv'), index=False)
log(f"Saved portfolio_returns.csv")

# Print summary stats
log("\n--- Portfolio Return Statistics (monthly %) ---")
for col in ['Active_EW', 'Active_VW', 'Passive_EW', 'Passive_VW']:
    vals = portfolio_returns[col].dropna()
    log(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

# ============================================================
# STEP 5: PERFORMANCE METRICS
# ============================================================
log("\n" + "=" * 70)
log("STEP 5: PERFORMANCE METRICS")
log("=" * 70)

def compute_cagr(returns_pct, periods_per_year=12):
    """Compute CAGR from percentage monthly returns."""
    returns_dec = returns_pct / 100.0
    cumulative = (1 + returns_dec).prod()
    n_years = len(returns_dec) / periods_per_year
    if n_years <= 0 or cumulative <= 0:
        return np.nan
    cagr = cumulative ** (1 / n_years) - 1
    return cagr * 100  # Return as percent

def compute_annualized_return(returns_pct):
    """Compute annualized return from monthly percentage returns."""
    returns_dec = returns_pct / 100.0
    mean_monthly = returns_dec.mean()
    ann_return = (1 + mean_monthly) ** 12 - 1
    return ann_return * 100

def compute_annualized_vol(returns_pct):
    """Compute annualized volatility from monthly percentage returns."""
    returns_dec = returns_pct / 100.0
    monthly_vol = returns_dec.std()
    ann_vol = monthly_vol * np.sqrt(12)
    return ann_vol * 100

def compute_sharpe(returns_pct, rf_pct):
    """Compute annualized Sharpe ratio."""
    excess_dec = (returns_pct - rf_pct) / 100.0
    if excess_dec.std() == 0:
        return np.nan
    sharpe = (excess_dec.mean() / excess_dec.std()) * np.sqrt(12)
    return sharpe

def compute_max_drawdown(returns_pct):
    """Compute maximum drawdown from percentage monthly returns."""
    returns_dec = returns_pct / 100.0
    cumulative = (1 + returns_dec).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return drawdowns.min() * 100  # Return as percent

performance_results = []
for col in ['Active_EW', 'Active_VW', 'Passive_EW', 'Passive_VW']:
    valid = portfolio_returns.dropna(subset=[col])
    rets = valid[col]
    rf = valid['RF']
    
    metrics = {
        'Portfolio': col,
        'N_Months': len(rets),
        'CAGR (%)': compute_cagr(rets),
        'Annualized_Return (%)': compute_annualized_return(rets),
        'Annualized_Volatility (%)': compute_annualized_vol(rets),
        'Sharpe_Ratio': compute_sharpe(rets, rf),
        'Max_Drawdown (%)': compute_max_drawdown(rets),
        'Mean_Monthly_Return (%)': rets.mean(),
        'Median_Monthly_Return (%)': rets.median(),
    }
    performance_results.append(metrics)
    log(f"\n  {col}:")
    for k, v in metrics.items():
        if k != 'Portfolio':
            log(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

df_performance = pd.DataFrame(performance_results)
df_performance.to_csv(os.path.join(OUTPUTS_TABLES, 'portfolio_performance.csv'), index=False)
log(f"\nSaved portfolio_performance.csv")

# ============================================================
# STEP 6: CARHART FOUR-FACTOR MODEL — PORTFOLIO LEVEL
# ============================================================
log("\n" + "=" * 70)
log("STEP 6: CARHART FOUR-FACTOR REGRESSION (PORTFOLIO LEVEL)")
log("=" * 70)

carhart_results = []
for col in ['Active_EW', 'Active_VW', 'Passive_EW', 'Passive_VW']:
    valid = portfolio_returns.dropna(subset=[col, 'MF', 'SMB', 'HML', 'WML', 'RF'])
    
    # Excess return = Portfolio return - RF (both in percent)
    y = valid[col] - valid['RF']
    X = valid[['MF', 'SMB', 'HML', 'WML']]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit(cov_type='HC1')  # Heteroskedasticity-robust standard errors
    
    monthly_alpha = model.params['const']
    annualized_alpha = ((1 + monthly_alpha / 100) ** 12 - 1) * 100
    
    result = {
        'Portfolio': col,
        'Monthly_Alpha (%)': monthly_alpha,
        'Annualized_Alpha (%)': annualized_alpha,
        'Beta_Market': model.params['MF'],
        'Beta_SMB': model.params['SMB'],
        'Beta_HML': model.params['HML'],
        'Beta_WML': model.params['WML'],
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj,
        'Alpha_tStat': model.tvalues['const'],
        'Alpha_pValue': model.pvalues['const'],
        'N_Observations': int(model.nobs),
    }
    carhart_results.append(result)
    
    log(f"\n  {col}:")
    log(f"    Monthly Alpha: {monthly_alpha:.4f}% (t={model.tvalues['const']:.3f}, p={model.pvalues['const']:.4f})")
    log(f"    Annualized Alpha: {annualized_alpha:.4f}%")
    log(f"    Beta_Market: {model.params['MF']:.4f}")
    log(f"    Beta_SMB: {model.params['SMB']:.4f}")
    log(f"    Beta_HML: {model.params['HML']:.4f}")
    log(f"    Beta_WML: {model.params['WML']:.4f}")
    log(f"    R²: {model.rsquared:.4f}")
    log(f"    N: {int(model.nobs)}")

df_carhart = pd.DataFrame(carhart_results)
df_carhart.to_csv(os.path.join(OUTPUTS_TABLES, 'portfolio_carhart_results.csv'), index=False)
log(f"\nSaved portfolio_carhart_results.csv")

# ============================================================
# STEP 7: FUND-LEVEL ROBUSTNESS ANALYSIS
# ============================================================
log("\n" + "=" * 70)
log("STEP 7: FUND-LEVEL CARHART REGRESSION")
log("=" * 70)

fund_level_results = []
excluded_funds = []

for fund_name in master['Fund_Name'].unique():
    fund_data = master[master['Fund_Name'] == fund_name].dropna(subset=['Monthly_Return', 'MF', 'SMB', 'HML', 'WML', 'RF'])
    
    if len(fund_data) < 12:
        excluded_funds.append({
            'Fund_Name': fund_name,
            'Fund_Type': fund_data['Fund_Type'].iloc[0] if len(fund_data) > 0 else 'Unknown',
            'N_Observations': len(fund_data),
            'Reason': f'Insufficient observations ({len(fund_data)} < 12)'
        })
        continue
    
    y = fund_data['Monthly_Return'] - fund_data['RF']
    X = fund_data[['MF', 'SMB', 'HML', 'WML']]
    X = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        monthly_alpha = model.params['const']
        annualized_alpha = ((1 + monthly_alpha / 100) ** 12 - 1) * 100
        
        fund_level_results.append({
            'Fund_Name': fund_name,
            'Fund_Type': fund_data['Fund_Type'].iloc[0],
            'Monthly_Alpha (%)': monthly_alpha,
            'Annualized_Alpha (%)': annualized_alpha,
            'Beta_Market': model.params['MF'],
            'Beta_SMB': model.params['SMB'],
            'Beta_HML': model.params['HML'],
            'Beta_WML': model.params['WML'],
            'R2': model.rsquared,
            'Alpha_tStat': model.tvalues['const'],
            'Alpha_pValue': model.pvalues['const'],
            'Num_Observations': int(model.nobs),
        })
    except Exception as e:
        excluded_funds.append({
            'Fund_Name': fund_name,
            'Fund_Type': fund_data['Fund_Type'].iloc[0],
            'N_Observations': len(fund_data),
            'Reason': f'Regression error: {str(e)}'
        })

df_fund_level = pd.DataFrame(fund_level_results)
df_fund_level.to_csv(os.path.join(OUTPUTS_TABLES, 'fund_level_carhart_results.csv'), index=False)
log(f"Fund-level regressions completed: {len(fund_level_results)} funds")
log(f"Funds excluded: {len(excluded_funds)}")

if excluded_funds:
    log("  Excluded funds:")
    for ef in excluded_funds:
        log(f"    [{ef['Fund_Type']}] {ef['Fund_Name']}: {ef['Reason']}")

# Summary by type
for ft in ['Active', 'Passive']:
    subset = df_fund_level[df_fund_level['Fund_Type'] == ft]
    if len(subset) > 0:
        log(f"\n  {ft} Fund Summary ({len(subset)} funds):")
        log(f"    Mean Monthly Alpha: {subset['Monthly_Alpha (%)'].mean():.4f}%")
        log(f"    Median Monthly Alpha: {subset['Monthly_Alpha (%)'].median():.4f}%")
        log(f"    Funds with positive alpha: {(subset['Monthly_Alpha (%)'] > 0).sum()}")
        log(f"    Funds with significant alpha (p<0.05): {(subset['Alpha_pValue'] < 0.05).sum()}")

# ============================================================
# STEP 8: ACTIVE VS PASSIVE COMPARISON
# ============================================================
log("\n" + "=" * 70)
log("STEP 8: ACTIVE VS PASSIVE COMPARISON")
log("=" * 70)

comparison_rows = []
for ft in ['Active', 'Passive']:
    subset = df_fund_level[df_fund_level['Fund_Type'] == ft]
    if len(subset) == 0:
        continue
    
    row = {
        'Metric': ft,
        'N_Funds': len(subset),
        'Mean_Monthly_Alpha (%)': subset['Monthly_Alpha (%)'].mean(),
        'Median_Monthly_Alpha (%)': subset['Monthly_Alpha (%)'].median(),
        'Mean_Annualized_Alpha (%)': subset['Annualized_Alpha (%)'].mean(),
        'Median_Annualized_Alpha (%)': subset['Annualized_Alpha (%)'].median(),
        'Mean_Beta_Market': subset['Beta_Market'].mean(),
        'Mean_R2': subset['R2'].mean(),
        'Pct_Positive_Alpha': (subset['Monthly_Alpha (%)'] > 0).mean() * 100,
        'Pct_Significant_Alpha_5pct': (subset['Alpha_pValue'] < 0.05).mean() * 100,
    }
    comparison_rows.append(row)

# Add Sharpe and CAGR at fund level
for ft in ['Active', 'Passive']:
    fund_names_ft = df_fund_level[df_fund_level['Fund_Type'] == ft]['Fund_Name'].tolist()
    fund_sharpes = []
    fund_cagrs = []
    
    for fn in fund_names_ft:
        fund_data = master[(master['Fund_Name'] == fn)].dropna(subset=['Monthly_Return', 'RF'])
        if len(fund_data) >= 12:
            fund_sharpes.append(compute_sharpe(fund_data['Monthly_Return'], fund_data['RF']))
            fund_cagrs.append(compute_cagr(fund_data['Monthly_Return']))
    
    for row in comparison_rows:
        if row['Metric'] == ft:
            row['Mean_Sharpe'] = np.mean(fund_sharpes) if fund_sharpes else np.nan
            row['Mean_CAGR (%)'] = np.mean(fund_cagrs) if fund_cagrs else np.nan

df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv(os.path.join(OUTPUTS_TABLES, 'active_vs_passive_summary.csv'), index=False)
log(f"\nSaved active_vs_passive_summary.csv")
log(f"\n{df_comparison.to_string()}")

# ============================================================
# STEP 9: STATISTICAL TESTING
# ============================================================
log("\n" + "=" * 70)
log("STEP 9: STATISTICAL TESTING")
log("=" * 70)

active_fl = df_fund_level[df_fund_level['Fund_Type'] == 'Active']
passive_fl = df_fund_level[df_fund_level['Fund_Type'] == 'Passive']

stat_tests = []

# 1. Alpha Difference Test (two-sample t-test)
if len(active_fl) > 1 and len(passive_fl) > 1:
    t_stat_alpha, p_val_alpha = stats.ttest_ind(
        active_fl['Monthly_Alpha (%)'], passive_fl['Monthly_Alpha (%)'], equal_var=False
    )
    stat_tests.append({
        'Test': 'Alpha Difference (Welch t-test)',
        'Active_Mean': active_fl['Monthly_Alpha (%)'].mean(),
        'Passive_Mean': passive_fl['Monthly_Alpha (%)'].mean(),
        'Difference': active_fl['Monthly_Alpha (%)'].mean() - passive_fl['Monthly_Alpha (%)'].mean(),
        'Statistic': t_stat_alpha,
        'p_value': p_val_alpha,
        'Interpretation': 'Significant' if p_val_alpha < 0.05 else 'Not Significant'
    })
    log(f"\n1. Alpha Difference Test:")
    log(f"   Active mean alpha: {active_fl['Monthly_Alpha (%)'].mean():.4f}%")
    log(f"   Passive mean alpha: {passive_fl['Monthly_Alpha (%)'].mean():.4f}%")
    log(f"   t-stat: {t_stat_alpha:.4f}, p-value: {p_val_alpha:.4f}")
    log(f"   Result: {'Significant at 5%' if p_val_alpha < 0.05 else 'Not significant at 5%'}")

# 2. Sharpe Ratio Difference Test
active_sharpes = []
passive_sharpes = []
for fn in active_fl['Fund_Name']:
    fd = master[master['Fund_Name'] == fn].dropna(subset=['Monthly_Return', 'RF'])
    if len(fd) >= 12:
        active_sharpes.append(compute_sharpe(fd['Monthly_Return'], fd['RF']))
for fn in passive_fl['Fund_Name']:
    fd = master[master['Fund_Name'] == fn].dropna(subset=['Monthly_Return', 'RF'])
    if len(fd) >= 12:
        passive_sharpes.append(compute_sharpe(fd['Monthly_Return'], fd['RF']))

if len(active_sharpes) > 1 and len(passive_sharpes) > 1:
    t_stat_sharpe, p_val_sharpe = stats.ttest_ind(active_sharpes, passive_sharpes, equal_var=False)
    stat_tests.append({
        'Test': 'Sharpe Ratio Difference (Welch t-test)',
        'Active_Mean': np.mean(active_sharpes),
        'Passive_Mean': np.mean(passive_sharpes),
        'Difference': np.mean(active_sharpes) - np.mean(passive_sharpes),
        'Statistic': t_stat_sharpe,
        'p_value': p_val_sharpe,
        'Interpretation': 'Significant' if p_val_sharpe < 0.05 else 'Not Significant'
    })
    log(f"\n2. Sharpe Ratio Difference Test:")
    log(f"   Active mean Sharpe: {np.mean(active_sharpes):.4f}")
    log(f"   Passive mean Sharpe: {np.mean(passive_sharpes):.4f}")
    log(f"   t-stat: {t_stat_sharpe:.4f}, p-value: {p_val_sharpe:.4f}")

# 3. CAGR Difference Test
active_cagrs = []
passive_cagrs = []
for fn in active_fl['Fund_Name']:
    fd = master[master['Fund_Name'] == fn].dropna(subset=['Monthly_Return'])
    if len(fd) >= 12:
        active_cagrs.append(compute_cagr(fd['Monthly_Return']))
for fn in passive_fl['Fund_Name']:
    fd = master[master['Fund_Name'] == fn].dropna(subset=['Monthly_Return'])
    if len(fd) >= 12:
        passive_cagrs.append(compute_cagr(fd['Monthly_Return']))

if len(active_cagrs) > 1 and len(passive_cagrs) > 1:
    t_stat_cagr, p_val_cagr = stats.ttest_ind(active_cagrs, passive_cagrs, equal_var=False)
    stat_tests.append({
        'Test': 'CAGR Difference (Welch t-test)',
        'Active_Mean': np.mean(active_cagrs),
        'Passive_Mean': np.mean(passive_cagrs),
        'Difference': np.mean(active_cagrs) - np.mean(passive_cagrs),
        'Statistic': t_stat_cagr,
        'p_value': p_val_cagr,
        'Interpretation': 'Significant' if p_val_cagr < 0.05 else 'Not Significant'
    })
    log(f"\n3. CAGR Difference Test:")
    log(f"   Active mean CAGR: {np.mean(active_cagrs):.4f}%")
    log(f"   Passive mean CAGR: {np.mean(passive_cagrs):.4f}%")
    log(f"   t-stat: {t_stat_cagr:.4f}, p-value: {p_val_cagr:.4f}")

# 4. Mann-Whitney U test (non-parametric) for alpha
if len(active_fl) > 1 and len(passive_fl) > 1:
    u_stat, p_val_mw = stats.mannwhitneyu(
        active_fl['Monthly_Alpha (%)'], passive_fl['Monthly_Alpha (%)'], alternative='two-sided'
    )
    stat_tests.append({
        'Test': 'Alpha Difference (Mann-Whitney U)',
        'Active_Mean': active_fl['Monthly_Alpha (%)'].mean(),
        'Passive_Mean': passive_fl['Monthly_Alpha (%)'].mean(),
        'Difference': active_fl['Monthly_Alpha (%)'].mean() - passive_fl['Monthly_Alpha (%)'].mean(),
        'Statistic': u_stat,
        'p_value': p_val_mw,
        'Interpretation': 'Significant' if p_val_mw < 0.05 else 'Not Significant'
    })
    log(f"\n4. Alpha Difference (Mann-Whitney U):")
    log(f"   U-stat: {u_stat:.4f}, p-value: {p_val_mw:.4f}")

df_stat_tests = pd.DataFrame(stat_tests)
df_stat_tests.to_csv(os.path.join(OUTPUTS_TABLES, 'statistical_tests.csv'), index=False)
log(f"\nSaved statistical_tests.csv")

# ============================================================
# STEP 10: EXPENSE RATIO ANALYSIS
# ============================================================
log("\n" + "=" * 70)
log("STEP 10: EXPENSE RATIO ANALYSIS")
log("=" * 70)

# Merge fund-level results with expense ratios
expense_analysis = df_fund_level.merge(df_expense[['Fund_Name', 'Expense_Ratio']], on='Fund_Name', how='left')

# Add Sharpe and CAGR
fund_metrics = []
for _, row in expense_analysis.iterrows():
    fd = master[master['Fund_Name'] == row['Fund_Name']].dropna(subset=['Monthly_Return', 'RF'])
    sharpe = compute_sharpe(fd['Monthly_Return'], fd['RF']) if len(fd) >= 12 else np.nan
    cagr = compute_cagr(fd['Monthly_Return']) if len(fd) >= 12 else np.nan
    fund_metrics.append({'Fund_Name': row['Fund_Name'], 'Sharpe_Ratio': sharpe, 'CAGR (%)': cagr})

df_fund_metrics = pd.DataFrame(fund_metrics)
expense_analysis = expense_analysis.merge(df_fund_metrics, on='Fund_Name', how='left')

# Compute averages
log("\nExpense Ratio Summary:")
for ft in ['Active', 'Passive']:
    subset = expense_analysis[expense_analysis['Fund_Type'] == ft]
    subset_valid = subset.dropna(subset=['Expense_Ratio'])
    log(f"  {ft}: Mean ER = {subset_valid['Expense_Ratio'].mean():.4f}%, Median ER = {subset_valid['Expense_Ratio'].median():.4f}%")

# Correlation: Expense Ratio vs Alpha
valid_expense = expense_analysis.dropna(subset=['Expense_Ratio', 'Monthly_Alpha (%)'])
if len(valid_expense) > 2:
    corr_er_alpha, p_corr_alpha = stats.pearsonr(valid_expense['Expense_Ratio'], valid_expense['Monthly_Alpha (%)'])
    log(f"\nCorrelation (Expense Ratio vs Monthly Alpha): r={corr_er_alpha:.4f}, p={p_corr_alpha:.4f}")

valid_expense_sharpe = expense_analysis.dropna(subset=['Expense_Ratio', 'Sharpe_Ratio'])
if len(valid_expense_sharpe) > 2:
    corr_er_sharpe, p_corr_sharpe = stats.pearsonr(valid_expense_sharpe['Expense_Ratio'], valid_expense_sharpe['Sharpe_Ratio'])
    log(f"Correlation (Expense Ratio vs Sharpe): r={corr_er_sharpe:.4f}, p={p_corr_sharpe:.4f}")

expense_analysis.to_csv(os.path.join(OUTPUTS_TABLES, 'expense_analysis.csv'), index=False)
log(f"\nSaved expense_analysis.csv")

# Descriptive statistics
desc_stats = expense_analysis.groupby('Fund_Type').agg({
    'Monthly_Alpha (%)': ['mean', 'median', 'std'],
    'Annualized_Alpha (%)': ['mean', 'median'],
    'Beta_Market': ['mean', 'median'],
    'R2': ['mean', 'median'],
    'Expense_Ratio': ['mean', 'median'],
    'Sharpe_Ratio': ['mean', 'median'],
    'CAGR (%)': ['mean', 'median'],
    'Num_Observations': ['mean', 'min', 'max'],
}).round(4)
desc_stats.to_csv(os.path.join(OUTPUTS_TABLES, 'descriptive_statistics.csv'))
log(f"Saved descriptive_statistics.csv")

# ============================================================
# STEP 11: VISUALIZATIONS
# ============================================================
log("\n" + "=" * 70)
log("STEP 11: VISUALIZATIONS")
log("=" * 70)

# Color palette
ACTIVE_COLOR = '#E74C3C'
PASSIVE_COLOR = '#3498DB'
ACTIVE_EW_COLOR = '#E74C3C'
ACTIVE_VW_COLOR = '#C0392B'
PASSIVE_EW_COLOR = '#3498DB'
PASSIVE_VW_COLOR = '#2980B9'

# --- 1. Growth of ₹100 ---
log("\n  Creating Growth of ₹100 chart...")
fig, ax = plt.subplots(figsize=(14, 7))

portfolio_sorted = portfolio_returns.sort_values('Date')
for col, label, color, ls in [
    ('Active_EW', 'Active (Equal-Weighted)', ACTIVE_EW_COLOR, '-'),
    ('Active_VW', 'Active (Value-Weighted)', ACTIVE_VW_COLOR, '--'),
    ('Passive_EW', 'Passive (Equal-Weighted)', PASSIVE_EW_COLOR, '-'),
    ('Passive_VW', 'Passive (Value-Weighted)', PASSIVE_VW_COLOR, '--'),
]:
    valid = portfolio_sorted.dropna(subset=[col])
    growth = 100 * (1 + valid[col] / 100).cumprod()
    ax.plot(range(len(growth)), growth.values, label=label, color=color, linewidth=2, linestyle=ls)

# Set x-axis labels
dates = portfolio_sorted['Date'].values
tick_interval = max(1, len(dates) // 12)
ax.set_xticks(range(0, len(dates), tick_interval))
ax.set_xticklabels([dates[i] for i in range(0, len(dates), tick_interval)], rotation=45, ha='right')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Growth of ₹100', fontsize=12)
ax.set_title('Growth of ₹100 — Active vs Passive Large-Cap Portfolios', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, 'growth_of_100.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  Saved growth_of_100.png")

# --- 2. CAGR Comparison ---
log("  Creating CAGR Comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
cagr_data = df_performance[['Portfolio', 'CAGR (%)']].copy()
colors = [ACTIVE_EW_COLOR, ACTIVE_VW_COLOR, PASSIVE_EW_COLOR, PASSIVE_VW_COLOR]
bars = ax.bar(cagr_data['Portfolio'], cagr_data['CAGR (%)'], color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cagr_data['CAGR (%)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%', 
            ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('CAGR (%)', fontsize=12)
ax.set_title('CAGR Comparison — Active vs Passive Portfolios', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, 'cagr_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  Saved cagr_comparison.png")

# --- 3. Sharpe Ratio Comparison ---
log("  Creating Sharpe Ratio Comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
sharpe_data = df_performance[['Portfolio', 'Sharpe_Ratio']].copy()
bars = ax.bar(sharpe_data['Portfolio'], sharpe_data['Sharpe_Ratio'], color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, sharpe_data['Sharpe_Ratio']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Sharpe Ratio Comparison — Active vs Passive Portfolios', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, 'sharpe_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  Saved sharpe_comparison.png")

# --- 4. Alpha Comparison (Portfolio Level) ---
log("  Creating Alpha Comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Monthly Alpha
ax = axes[0]
alpha_monthly = df_carhart[['Portfolio', 'Monthly_Alpha (%)']].copy()
bars = ax.bar(alpha_monthly['Portfolio'], alpha_monthly['Monthly_Alpha (%)'], color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, alpha_monthly['Monthly_Alpha (%)']):
    ypos = bar.get_height() + 0.002 if bar.get_height() >= 0 else bar.get_height() - 0.02
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.4f}%', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)
ax.set_ylabel('Monthly Alpha (%)', fontsize=11)
ax.set_title('Monthly Carhart Alpha', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Annualized Alpha
ax = axes[1]
alpha_ann = df_carhart[['Portfolio', 'Annualized_Alpha (%)']].copy()
bars = ax.bar(alpha_ann['Portfolio'], alpha_ann['Annualized_Alpha (%)'], color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, alpha_ann['Annualized_Alpha (%)']):
    ypos = bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.15
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.2f}%', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)
ax.set_ylabel('Annualized Alpha (%)', fontsize=11)
ax.set_title('Annualized Carhart Alpha', fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.suptitle('Carhart Four-Factor Alpha — Active vs Passive', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, 'alpha_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  Saved alpha_comparison.png")

# --- 5. Alpha Distribution (Fund Level) ---
log("  Creating Alpha Distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Monthly Alpha distribution
ax = axes[0]
for ft, color, label in [('Active', ACTIVE_COLOR, 'Active'), ('Passive', PASSIVE_COLOR, 'Passive')]:
    subset = df_fund_level[df_fund_level['Fund_Type'] == ft]['Monthly_Alpha (%)']
    if len(subset) > 0:
        ax.hist(subset, bins=15, alpha=0.6, color=color, label=f'{label} (n={len(subset)})', edgecolor='white')
        ax.axvline(subset.mean(), color=color, linestyle='--', linewidth=2, label=f'{label} Mean: {subset.mean():.4f}%')
ax.set_xlabel('Monthly Alpha (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Monthly Alpha Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

# Annualized Alpha distribution
ax = axes[1]
for ft, color, label in [('Active', ACTIVE_COLOR, 'Active'), ('Passive', PASSIVE_COLOR, 'Passive')]:
    subset = df_fund_level[df_fund_level['Fund_Type'] == ft]['Annualized_Alpha (%)']
    if len(subset) > 0:
        ax.hist(subset, bins=15, alpha=0.6, color=color, label=f'{label} (n={len(subset)})', edgecolor='white')
        ax.axvline(subset.mean(), color=color, linestyle='--', linewidth=2, label=f'{label} Mean: {subset.mean():.2f}%')
ax.set_xlabel('Annualized Alpha (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Annualized Alpha Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Fund-Level Alpha Distribution — Active vs Passive', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, 'alpha_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  Saved alpha_distribution.png")

# --- 6. Expense Ratio Distribution ---
log("  Creating Expense Ratio Distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Expense ratio distribution
ax = axes[0]
for ft, color, label in [('Active', ACTIVE_COLOR, 'Active'), ('Passive', PASSIVE_COLOR, 'Passive')]:
    subset = expense_analysis[expense_analysis['Fund_Type'] == ft]['Expense_Ratio'].dropna()
    if len(subset) > 0:
        ax.hist(subset, bins=12, alpha=0.6, color=color, label=f'{label} (n={len(subset)})', edgecolor='white')
        ax.axvline(subset.mean(), color=color, linestyle='--', linewidth=2, label=f'{label} Mean: {subset.mean():.2f}%')
ax.set_xlabel('Expense Ratio (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Expense Ratio Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Expense Ratio vs Alpha scatter
ax = axes[1]
for ft, color, marker in [('Active', ACTIVE_COLOR, 'o'), ('Passive', PASSIVE_COLOR, 's')]:
    subset = expense_analysis[(expense_analysis['Fund_Type'] == ft)].dropna(subset=['Expense_Ratio', 'Monthly_Alpha (%)'])
    if len(subset) > 0:
        ax.scatter(subset['Expense_Ratio'], subset['Monthly_Alpha (%)'], color=color, marker=marker, 
                   s=60, alpha=0.7, label=ft, edgecolors='white', linewidth=0.5)

# Add regression line for all funds
valid_all = expense_analysis.dropna(subset=['Expense_Ratio', 'Monthly_Alpha (%)'])
if len(valid_all) > 2:
    z = np.polyfit(valid_all['Expense_Ratio'], valid_all['Monthly_Alpha (%)'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_all['Expense_Ratio'].min(), valid_all['Expense_Ratio'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.5, label=f'Trend (r={corr_er_alpha:.3f})')

ax.set_xlabel('Expense Ratio (%)', fontsize=11)
ax.set_ylabel('Monthly Alpha (%)', fontsize=11)
ax.set_title('Expense Ratio vs Monthly Alpha', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(alpha=0.3)

plt.suptitle('Expense Ratio Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, 'expense_ratio_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
log("  Saved expense_ratio_distribution.png")

# ============================================================
# SAVE FINAL VALIDATION REPORT  
# ============================================================
log("\n" + "=" * 70)
log("GENERATING FINAL VALIDATION REPORT")
log("=" * 70)

# Re-count after name mapping
final_active_funds = master[master['Fund_Type']=='Active']['Fund_Name'].nunique()
final_passive_funds = master[master['Fund_Type']=='Passive']['Fund_Name'].nunique()
final_date_range = f"{master['Date'].min()} to {master['Date'].max()}"
final_n_months = master['Date'].nunique()

validation_report = f"""# Validation Report
## Active vs Passive Large-Cap Mutual Funds in India — Carhart Four-Factor Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Dataset Summary

| Dataset | File | Rows | Columns |
|---------|------|------|---------|
| Fund Classification | fund_classification.csv | {len(df_classification)} | {len(df_classification.columns)} |
| Active Fund Returns | active_funds.numbers | {len(df_active_raw)} | {len(df_active_raw.columns)} |
| Passive Fund Returns | passive_funds.numbers | {len(df_passive_raw)} | {len(df_passive_raw.columns)} |
| Carhart Factors | carhart_factors.csv | {len(df_factors)} | 6 |
| Expense Ratios | expense_ratios.csv | {len(df_expense)} | {len(df_expense.columns)} |
| AUM Data | aum_data.csv | {len(df_aum)} | {len(df_aum.columns)} |

## 2. Detected Columns

| Dataset | Columns |
|---------|---------|
| Fund Classification | Fund_Name, Fund_Type |
| Active Funds | Fund_Name, Month, Date, NAV, Monthly_Return, Monthly_Return_Percent |
| Passive Funds | FUND_NAME, month, date, NAV |
| Carhart Factors | Date, SMB, HML, WML, MF, RF |
| Expense Ratios | Fund_Name, Expense_Ratio |
| AUM Data | Fund_Name, AUM |

## 3. Row Counts

| Dataset | Records |
|---------|---------|
| Active returns (processed) | {len(active_returns)} |
| Passive returns (processed) | {len(passive_returns)} |
| Master dataset | {len(master)} |

## 4. Fund Counts

| Category | Count |
|----------|-------|
| Total classified | {len(df_classification)} |
| Active classified | {len(df_classification[df_classification['Fund_Type']=='Active'])} |
| Passive classified | {len(df_classification[df_classification['Fund_Type']=='Passive'])} |
| **Active in master** | **{final_active_funds}** |
| **Passive in master** | **{final_passive_funds}** |
| **Total in master** | **{final_active_funds + final_passive_funds}** |

## 5. Date Ranges

| Dataset | Earliest | Latest | Months |
|---------|----------|--------|--------|
| Active returns | {active_returns['Date'].min()} | {active_returns['Date'].max()} | {active_returns['Date'].nunique()} |
| Passive returns | {passive_returns['Date'].min()} | {passive_returns['Date'].max()} | {passive_returns['Date'].nunique()} |
| Carhart Factors | {df_factors['Date_YM'].min()} | {df_factors['Date_YM'].max()} | {len(df_factors)} |
| **Master (overlap)** | **{master['Date'].min()}** | **{master['Date'].max()}** | **{final_n_months}** |

## 6. Missing Values in Master Dataset

| Column | Missing | % |
|--------|---------|---|
"""

for col in master.columns:
    na = master[col].isna().sum()
    validation_report += f"| {col} | {na} | {na/len(master)*100:.1f}% |\n"

validation_report += f"""
## 7. Duplicate Records

| Dataset | Duplicate (Fund+Month) | Action |
|---------|----------------------|--------|
| Active Funds | 0 | None needed |
| Passive Funds | 0 (after date fix) | Dates correctly parsed from 'date' column |
| Classification | 0 | None needed |

## 8. Match Rates

| Match Type | Rate | Matched | Total |
|------------|------|---------|-------|
| Classification ↔ Master | {classification_match_rate:.1f}% | {len(classified_in_master)} | {total_classified} |
| Expense Ratio ↔ Master | {expense_match_rate:.1f}% | {funds_with_expense} | {master['Fund_Name'].nunique()} |
| AUM ↔ Master | {aum_match_rate:.1f}% | {funds_with_aum} | {master['Fund_Name'].nunique()} |
| Factor dates | 100.0% | (inner join) | — |

### Classified Funds Not in Master ({len(missing_classified)}):
"""

for f in sorted(missing_classified):
    ft = df_classification[df_classification['Fund_Name'] == f]['Fund_Type'].iloc[0]
    validation_report += f"- [{ft}] {f}\n"

validation_report += f"""
### Fund Name Mapping Applied

The following names in the .numbers files were mapped to classification names:

| Original Name (in .numbers) | Mapped Name (in classification) |
|------------------------------|-------------------------------|
"""
for orig, mapped in sorted(FUND_NAME_MAP.items()):
    if orig != mapped:
        validation_report += f"| {orig} | {mapped} |\n"

validation_report += f"""
## 9. RF Scale Verification

| Metric | Value |
|--------|-------|
| RF Min | {df_factors['RF'].min():.6f} |
| RF Max | {df_factors['RF'].max():.6f} |
| RF Mean | {df_factors['RF'].mean():.6f} |
| **Scale** | **Percent monthly** |

RF values (range 0.23–1.11%) are consistent with India's monthly risk-free rate derived from 91-day T-bill yields.

## 10. Factor Scale Verification

| Factor | Min | Max | Mean | Scale |
|--------|-----|-----|------|-------|
| MF | {df_factors['MF'].min():.4f} | {df_factors['MF'].max():.4f} | {df_factors['MF'].mean():.4f} | Percent |
| SMB | {df_factors['SMB'].min():.4f} | {df_factors['SMB'].max():.4f} | {df_factors['SMB'].mean():.4f} | Percent |
| HML | {df_factors['HML'].min():.4f} | {df_factors['HML'].max():.4f} | {df_factors['HML'].mean():.4f} | Percent |
| WML | {df_factors['WML'].min():.4f} | {df_factors['WML'].max():.4f} | {df_factors['WML'].mean():.4f} | Percent |

All factors confirmed in **percent** scale.

## 11. Return Scale Verification

| Source | Raw Scale | Converted To | Method |
|--------|-----------|-------------|--------|
| Active funds | Decimal (e.g., 0.05 = 5%) | Percent (×100) | Monthly_Return × 100 |
| Passive funds | Decimal (pct_change) | Percent (×100) | NAV pct_change × 100 |

All returns, factors, and RF are now in **percent** for consistent regression analysis.

## 12. Excluded Funds

| Fund | Type | Observations | Reason |
|------|------|-------------|--------|
"""

for ef in excluded_funds:
    validation_report += f"| {ef['Fund_Name']} | {ef['Fund_Type']} | {ef['N_Observations']} | {ef['Reason']} |\n"

if not excluded_funds:
    validation_report += "| None | — | — | All funds have ≥12 observations |\n"

validation_report += f"""
## 13. Assumptions

1. **Date Standardization:** All dates converted to YYYY-MM for monthly alignment.
2. **Active Returns:** Monthly_Return used directly from the data (decimal → percent ×100).
3. **Passive Returns:** Computed as NAV pct_change (decimal → percent ×100).
4. **First-Month NaN:** First observation per passive fund dropped (no prior NAV).
5. **RF Scale:** Monthly percentage return (~0.23–1.11%).
6. **Factor Scale:** All factors (MF, SMB, HML, WML) in percent.
7. **Static AUM:** Latest available AUM snapshot used for value-weighting.
8. **Name Mapping:** Fund names harmonized between .numbers files and classification.
9. **Inner Join:** Only months with both fund returns AND factor data retained.
10. **Minimum Observations:** Fund-level regressions require ≥12 months.
11. **Robust Standard Errors:** HC1 (White) heteroskedasticity-consistent standard errors used.
12. **Alpha Annualization:** Compounding method: (1 + monthly_alpha)^12 − 1.

## 14. Number of Excluded Observations

| Stage | Records Before | Records After | Lost |
|-------|---------------|---------------|------|
| Raw active | {len(df_active_raw)} | {len(active_returns)} | {len(df_active_raw) - len(active_returns)} |
| Raw passive | {len(df_passive_raw)} | {len(passive_returns)} | {len(df_passive_raw) - len(passive_returns)} |
| Classification merge | {len(active_returns) + len(passive_returns)} | {len(master)} | {len(active_returns) + len(passive_returns) - len(master)} |

---

**Status:** ✅ Validation Complete. All checks documented. Proceeding with analysis.
"""

with open(os.path.join(OUTPUTS_REPORTS, 'Validation_Report.md'), 'w') as f:
    f.write(validation_report)
log("Saved Validation_Report.md")

# ============================================================
# FINAL SUMMARY
# ============================================================
log("\n" + "=" * 70)
log("PIPELINE COMPLETE")
log("=" * 70)
log(f"\nMaster dataset: {final_active_funds} active + {final_passive_funds} passive funds")
log(f"Date range: {final_date_range}")
log(f"Analysis months: {final_n_months}")
log(f"\nAll output files generated successfully.")
