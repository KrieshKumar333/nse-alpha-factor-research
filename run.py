"""
run.py — Batch runner for Alpha Factor Research.
Usage: python run.py [--mock]
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run")


def main(use_mock: bool = False) -> None:
    import os; os.makedirs("outputs", exist_ok=True)
    from src.data_fetcher import UniverseDataFetcher
    from src.factor_engine import FactorEngine
    from src.config import FACTOR_NAMES, IC_SIGNIFICANCE_THRESHOLD

    logger.info("=" * 65)
    logger.info("Alpha Factor Research — NSE 500 Universe")
    logger.info("=" * 65)

    # 1. Load data
    fetcher = UniverseDataFetcher(use_mock=use_mock)
    mp  = fetcher.get_monthly_prices()
    fd  = fetcher.get_fundamentals()
    bm  = fetcher.get_benchmark_returns()
    logger.info(f"Universe: {mp.shape[1]} stocks | {mp.shape[0]} months")

    # 2. Build + score factors
    eng = FactorEngine(mp, fd, bm)
    scored = eng.score_factors()
    logger.info(f"Factors built: {list(scored.keys())}")

    # 3. IC summary
    ic_df = eng.ic_summary()
    logger.info("\n" + "─" * 65)
    logger.info(f"{'Factor':<20} {'MeanIC':>8} {'ICIR':>8} {'t-stat':>8} {'Sig':>6}")
    logger.info("─" * 65)
    for factor, row in ic_df.iterrows():
        sig = "✓" if row.get("Significant", False) else " "
        logger.info(f"  {factor:<18} {row['Mean IC']:>8.4f} "
                    f"{row['ICIR'] if not (isinstance(row['ICIR'], float) and row['ICIR'] != row['ICIR']) else float('nan'):>8.3f} "
                    f"{row['t-statistic'] if not (isinstance(row['t-statistic'], float) and row['t-statistic'] != row['t-statistic']) else float('nan'):>8.2f}  {sig}")
    logger.info("─" * 65)
    n_sig = int(ic_df["Significant"].sum())
    logger.info(f"Significant factors (|IC| > {IC_SIGNIFICANCE_THRESHOLD}): {n_sig} / {len(ic_df)}")

    # 4. L/S stats for each significant factor
    logger.info("\nLong-Short Portfolio Performance:")
    logger.info(f"  {'Factor':<20} {'Ann.Ret':>9} {'Ann.Vol':>9} {'Sharpe':>9} {'MaxDD':>9}")
    for factor_name in ic_df[ic_df["Significant"] == True].index:
        try:
            ls, ls_cum = eng.long_short_returns(factor_name)
            ls = ls.dropna()
            if len(ls) < 6:
                continue
            ann_ret = ls.mean() * 12
            ann_vol = ls.std() * (12 ** 0.5)
            sharpe  = ann_ret / ann_vol if ann_vol > 0 else float('nan')
            max_dd  = ((1 + ls.cumsum()).cummax() - (1 + ls.cumsum())).max()
            logger.info(f"  {factor_name:<20} {ann_ret:>8.1%} {ann_vol:>9.1%} "
                        f"{sharpe:>9.2f} {max_dd:>9.1%}")
        except Exception as e:
            logger.warning(f"  {factor_name}: L/S calc error — {e}")

    # 5. Factor correlation
    corr = eng.factor_correlation_matrix()
    avg_corr = corr.abs().values.copy()
    import numpy as np; np.fill_diagonal(avg_corr, np.nan)
    logger.info(f"\nAverage inter-factor |IC| correlation: {np.nanmean(avg_corr):.3f}")
    logger.info("(Lower = better diversification when combining factors into multi-factor score)")

    # 6. Save IC summary to Excel
    try:
        ic_df.to_excel("outputs/Alpha_Factor_Research.xlsx")
        logger.info("\nIC summary saved → outputs/Alpha_Factor_Research.xlsx")
    except Exception as e:
        logger.warning(f"Excel save failed: {e}")

    logger.info("\nRun 'streamlit run dashboard.py' for the interactive dashboard.")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    main(use_mock=args.mock)
