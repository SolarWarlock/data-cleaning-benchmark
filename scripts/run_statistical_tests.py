"""
CLI entrypoint to calculate non-parametric statistical tests (Friedman and Wilcoxon with Holm-Bonferroni correction),
verify break-even index (BEI), and display summary metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats
from src.analysis.stats import BenchmarkAnalyzer

def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "experiments_results.parquet"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "experiments_results.csv"

    if not data_path.exists():
        print(f"Error: experimental results not found in {data_path}")
        sys.exit(1)

    print("=" * 75)
    print("STATISTICAL VALIDATION & COST-BENEFIT ANALYSIS OF DATA CLEANING STRATEGIES")
    print(f"Results Source: {data_path.name}")
    print("=" * 75)

    analyzer = BenchmarkAnalyzer(data_path=data_path, output_dir=PROJECT_ROOT / "figures")

    # 1. Friedman Test
    friedman_stat, friedman_p, ranks = analyzer.run_friedman_test()
    print("\n1. Non-parametric Friedman Test across 240 paired conditions (N = 240, k = 4):")
    print(f"   Chi-squared statistic (Q): {friedman_stat:.4f}")
    print(f"   p-value:                   {friedman_p:.4e}")
    print("   Mean Ranks (lower is better, ranking order):")
    for strat, rank in sorted(ranks.items(), key=lambda x: x[1]):
        print(f"     - {strat:<20}: {rank:.3f}")

    # 2. Wilcoxon Paired Tests with Holm-Bonferroni Correction
    print("\n2. Pairwise Wilcoxon Signed-Rank Tests (vs Baseline C_0, with Holm-Bonferroni):")
    wilcox_df = analyzer.run_wilcoxon_tests()
    print(wilcox_df.to_string(index=False))

    # 3. Break-even index summary for LightGBM
    df = analyzer.df
    lgb = df[df["model"] == "lightgbm"]
    print("\n3. Break-even Index (BEI = Delta_F1 / T_clean_sec) for LightGBM across noise ratios:")
    bei_summary = lgb.groupby(["strategy", "noise_ratio"])["break_even_ratio"].mean().unstack(level=1)
    print(bei_summary.round(4).to_string())

    print("\n" + "=" * 75)
    print("ALL STATISTICAL CHECKS COMPLETED SUCCESSFULLY.")
    print("=" * 75)

if __name__ == "__main__":
    main()
