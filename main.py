"""
main.py
=======
Top-level pipeline orchestrator for the Carhart Four-Factor Fund Analysis.

Executes the complete research pipeline in strict stage order:

  Stage 1 — Data Loading      (data_loader.py)
  Stage 2 — Pre-processing     (preprocessing.py)
  Stage 3 — Return Engineering (return_calculations.py)
  Stage 4 — Factor Merge       (factor_merge.py)
  Stage 5 — Regression         (regression_analysis.py)
  Stage 6 — Performance Metrics(performance_metrics.py)
  Stage 7 — Statistical Tests  (statistical_tests.py)
  Stage 8 — Visualization      (visualization.py)

Usage
-----
    python main.py                   # full pipeline, save all outputs
    python main.py --no-plots        # skip visualization (faster)
    python main.py --stage 5         # run from stage 5 onward
    python main.py --dry-run         # validate data without writing output
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# ── Path fix so this script works from any working directory ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CLEANED_DATA_DIR,
    PERFORMANCE_METRICS_FILE,
    REGRESSION_RESULTS_DIR,
    STATISTICAL_TESTS_FILE,
)
from src.utils import ensure_dir, setup_logger

logger = setup_logger("main", level="INFO")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carhart Four-Factor Fund Analysis Pipeline"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip the visualization stage (Stage 8)."
    )
    parser.add_argument(
        "--stage", type=int, default=1, choices=range(1, 9),
        metavar="N",
        help=(
            "Start pipeline from stage N (1–8).  "
            "Stages 1–4 must have been run before using --stage ≥ 5 "
            "because they produce saved CSVs consumed by later stages."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and validate data without saving any outputs."
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _banner(stage: int, name: str) -> None:
    logger.info(f"\n{'═' * 68}")
    logger.info(f"  STAGE {stage}: {name}")
    logger.info(f"{'═' * 68}")


def _stage_time(t0: float) -> str:
    elapsed = time.perf_counter() - t0
    return f"{elapsed:.1f}s"


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════

def stage1_load(dry_run: bool) -> dict:
    from src.data_loader import load_all_data
    t0 = time.perf_counter()
    _banner(1, "Data Loading")
    raw = load_all_data()
    logger.info(
        f"  ✓ Stage 1 complete ({_stage_time(t0)}).  "
        f"Datasets: {list(raw.keys())}"
    )
    return raw


def stage2_preprocess(raw: dict, dry_run: bool):
    from src.preprocessing import build_master_frame
    t0 = time.perf_counter()
    _banner(2, "Pre-processing")
    master = build_master_frame(
        passive_df=raw["passive_nav"],
        active_df=raw["active_nav"],
        er_df=raw["expense_ratios"],
        save_intermediates=not dry_run,
    )
    logger.info(
        f"  ✓ Stage 2 complete ({_stage_time(t0)}).  "
        f"Master frame: {master.shape}"
    )
    return master


def stage3_returns(master, dry_run: bool):
    from src.return_calculations import enrich_master_frame
    t0 = time.perf_counter()
    _banner(3, "Return Engineering")
    enriched, fund_summary, reg_df = enrich_master_frame(
        master_df=master, save_outputs=not dry_run
    )
    logger.info(
        f"  ✓ Stage 3 complete ({_stage_time(t0)}).  "
        f"Enriched: {enriched.shape}  |  Reg-eligible: {len(reg_df)} rows"
    )
    return enriched, fund_summary, reg_df


def stage4_factors(enriched, reg_df, raw: dict, dry_run: bool):
    from src.factor_merge import build_and_merge_factors
    t0 = time.perf_counter()
    _banner(4, "Factor Construction & Merge")
    factor_table, merged_df = build_and_merge_factors(
        master_df=enriched,
        reg_df=reg_df,
        raw_factor_df=raw["factor_data"],
        save_outputs=not dry_run,
    )
    logger.info(
        f"  ✓ Stage 4 complete ({_stage_time(t0)}).  "
        f"Merged frame: {merged_df.shape}"
    )
    return factor_table, merged_df


def stage5_regression(merged_df, dry_run: bool) -> dict:
    from src.regression_analysis import run_full_regression_analysis
    t0 = time.perf_counter()
    _banner(5, "Carhart Regression Analysis")
    reg_out = run_full_regression_analysis(merged_df, save_outputs=not dry_run)
    logger.info(
        f"  ✓ Stage 5 complete ({_stage_time(t0)}).  "
        f"Funds regressed: {len(reg_out['fund_results'])}"
    )
    return reg_out


def stage6_metrics(enriched, reg_results, dry_run: bool) -> dict:
    import pandas as pd
    from src.performance_metrics import build_full_performance_report
    t0 = time.perf_counter()
    _banner(6, "Performance Metrics")
    report = build_full_performance_report(
        enriched_df=enriched,
        reg_results_df=reg_results,
        save_outputs=not dry_run,
    )
    logger.info(
        f"  ✓ Stage 6 complete ({_stage_time(t0)}).  "
        f"Metrics table: {report['metrics_table'].shape}"
    )
    return report


def stage7_stats(enriched, reg_results, metrics_df, dry_run: bool) -> dict:
    from src.statistical_tests import run_all_statistical_tests
    t0 = time.perf_counter()
    _banner(7, "Statistical Tests")
    results = run_all_statistical_tests(
        enriched_df=enriched,
        reg_results_df=reg_results,
        metrics_df=metrics_df,
        save_outputs=not dry_run,
    )
    logger.info(
        f"  ✓ Stage 7 complete ({_stage_time(t0)}).  "
        f"Test batteries: 7."
    )
    return results


def stage8_viz(enriched, reg_results, metrics_df, reg_input_df,
               stat_tests_df, dry_run: bool) -> dict:
    from src.visualization import generate_all_charts
    t0 = time.perf_counter()
    _banner(8, "Visualization")
    if dry_run:
        logger.info("  --dry-run: skipping chart generation.")
        return {}
    saved = generate_all_charts(
        enriched_df=enriched,
        reg_results_df=reg_results,
        metrics_df=metrics_df,
        reg_input_df=reg_input_df,
        stat_tests_df=stat_tests_df,
    )
    logger.info(
        f"  ✓ Stage 8 complete ({_stage_time(t0)}).  "
        f"Charts saved: {len(saved)}"
    )
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# FAST-RESUME: load pre-saved CSVs when starting from a mid-stage
# ══════════════════════════════════════════════════════════════════════════════

def _load_saved_csvs(start_stage: int):
    """Load already-saved CSV files when --stage N skips early stages."""
    import pandas as pd
    logger.info(
        f"  Fast-resume: loading pre-saved CSVs for stages < {start_stage} …"
    )

    enriched = reg_input = reg_results = metrics_df = stat_tests_df = None

    if start_stage >= 3:
        csv = CLEANED_DATA_DIR / "master_enriched.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"Cannot fast-resume at stage {start_stage}: "
                f"master_enriched.csv not found.  Run from stage 1."
            )
        enriched = pd.read_csv(csv, parse_dates=["Date"])
        logger.info(f"    Loaded master_enriched.csv  {enriched.shape}")

    if start_stage >= 5:
        csv = REGRESSION_RESULTS_DIR / "regression_input.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"regression_input.csv missing — run from stage 4."
            )
        reg_input = pd.read_csv(csv, parse_dates=["Date"])

        # Merge MF into enriched
        if enriched is not None and "MF" not in enriched.columns:
            mf = reg_input.groupby("Date")["MF"].mean().reset_index()
            enriched = enriched.merge(mf, on="Date", how="left")
        logger.info(f"    Loaded regression_input.csv  {reg_input.shape}")

    if start_stage >= 6:
        csv = REGRESSION_RESULTS_DIR / "carhart_regression_summary.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"carhart_regression_summary.csv missing — run from stage 5."
            )
        reg_results = pd.read_csv(csv)
        logger.info(f"    Loaded carhart_regression_summary.csv  {reg_results.shape}")

    if start_stage >= 7:
        if not PERFORMANCE_METRICS_FILE.exists():
            raise FileNotFoundError(
                f"performance_metrics.csv missing — run from stage 6."
            )
        metrics_df = pd.read_csv(PERFORMANCE_METRICS_FILE)
        logger.info(f"    Loaded performance_metrics.csv  {metrics_df.shape}")

    if start_stage >= 8:
        if STATISTICAL_TESTS_FILE.exists():
            stat_tests_df = pd.read_csv(STATISTICAL_TESTS_FILE)
            logger.info(f"    Loaded statistical_tests.csv  {stat_tests_df.shape}")

    return enriched, reg_input, reg_results, metrics_df, stat_tests_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()
    t_total = time.perf_counter()

    logger.info("\n" + "█" * 68)
    logger.info("  CARHART FOUR-FACTOR FUND ANALYSIS PIPELINE")
    logger.info("  Inspired by: Nanigian (2019) — Active vs Passive MF Performance")
    logger.info("█" * 68)

    if args.dry_run:
        logger.info("  ⚠  DRY-RUN MODE: no output files will be written.")
    if args.stage > 1:
        logger.info(f"  ⚡ FAST-RESUME: starting from Stage {args.stage}.")
    if args.no_plots:
        logger.info("  ⚡ --no-plots: Stage 8 (Visualization) will be skipped.")

    # Ensure output directories exist
    for d in [CLEANED_DATA_DIR, REGRESSION_RESULTS_DIR]:
        ensure_dir(d)

    # ── Stage state ─────────────────────────────────────────────────────────
    raw         = None
    master      = None
    enriched    = None
    fund_summary= None
    reg_df      = None
    factor_table= None
    merged_df   = None
    reg_out     = None
    reg_results = None
    metrics_rpt = None
    metrics_df  = None
    stat_results= None
    stat_df     = None
    reg_input   = None

    try:
        # ── Fast-resume: pre-load saved CSVs ──────────────────────────────────
        if args.stage > 1:
            (enriched, reg_input, reg_results, metrics_df, stat_df
             ) = _load_saved_csvs(args.stage)

        # ── Stage 1: Data Loading ─────────────────────────────────────────────
        if args.stage <= 1:
            raw = stage1_load(args.dry_run)

        # ── Stage 2: Pre-processing ───────────────────────────────────────────
        if args.stage <= 2:
            master = stage2_preprocess(raw, args.dry_run)

        # ── Stage 3: Return Engineering ───────────────────────────────────────
        if args.stage <= 3:
            enriched, fund_summary, reg_df = stage3_returns(master, args.dry_run)

        # ── Stage 4: Factor Merge ─────────────────────────────────────────────
        if args.stage <= 4:
            factor_table, merged_df = stage4_factors(
                enriched, reg_df, raw, args.dry_run
            )
        else:
            merged_df = reg_input   # pre-loaded

        # ── Stage 5: Regression ───────────────────────────────────────────────
        if args.stage <= 5:
            reg_out     = stage5_regression(merged_df, args.dry_run)
            reg_results = reg_out["fund_results"]
        # (if start >= 6, reg_results already loaded by _load_saved_csvs)

        # ── Stage 6: Performance Metrics ──────────────────────────────────────
        if args.stage <= 6:
            # Ensure MF is in enriched (needed by performance_metrics)
            if enriched is not None and "MF" not in enriched.columns and merged_df is not None:
                import pandas as pd
                mf = merged_df.groupby("Date")["MF"].mean().reset_index()
                enriched = enriched.merge(mf, on="Date", how="left")

            metrics_rpt = stage6_metrics(enriched, reg_results, args.dry_run)
            metrics_df  = metrics_rpt["metrics_table"]

        # ── Stage 7: Statistical Tests ────────────────────────────────────────
        if args.stage <= 7:
            stat_results = stage7_stats(
                enriched, reg_results, metrics_df, args.dry_run
            )
            stat_df = stat_results.get("multiple_comparison")

        # ── Stage 8: Visualization ────────────────────────────────────────────
        if args.stage <= 8 and not args.no_plots:
            stage8_viz(
                enriched=enriched,
                reg_results=reg_results,
                metrics_df=metrics_df,
                reg_input_df=merged_df if merged_df is not None else reg_input,
                stat_tests_df=stat_df,
                dry_run=args.dry_run,
            )

    except KeyboardInterrupt:
        logger.warning("\n  ⚠  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception:
        logger.error("\n  ❌  Pipeline failed with an unhandled exception:")
        traceback.print_exc()
        sys.exit(2)

    # ── Final summary ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    logger.info(f"\n{'█' * 68}")
    logger.info(f"  PIPELINE COMPLETE  —  Total time: {total_time:.1f}s")
    logger.info(f"{'█' * 68}\n")


if __name__ == "__main__":
    main()
