"""
CLI entrypoint to run the benchmark suite or inspect benchmark status.
Wraps src.pipeline.runner.BenchmarkRunner.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.runner import BenchmarkRunner, parse_args, NoiseType, CleaningStrategyType, ModelType
import pandas as pd

if __name__ == "__main__":
    args = parse_args()
    runner = BenchmarkRunner()

    if args.status:
        completed = runner.get_completed_keys()
        print("=" * 60)
        print(f"BENCHMARK STATUS: {len(completed)} experimental runs recorded on disk.")
        print("=" * 60)
        if runner.parquet_path.exists():
            df = pd.read_parquet(runner.parquet_path)
            print("\nBreakdown by Dataset:")
            print(df["dataset"].value_counts().to_string())
            print("\nBreakdown by Cleaning Strategy:")
            print(df["strategy"].value_counts().to_string())
            print("\nBreakdown by Defect Type:")
            print(df["noise_type"].value_counts().to_string())
            print("\nBreakdown by Classifier Model:")
            print(df["model"].value_counts().to_string())
        sys.exit(0)

    if args.pilot:
        print("=" * 60)
        print("STARTING PILOT BENCHMARK RUN (Adult & HELOC, 2 noise levels, 4 strategies, 2 models)")
        print("=" * 60)
        datasets = ["adult", "heloc"]
        noise_types = [NoiseType.MCAR, NoiseType.OUTLIERS, NoiseType.LABEL_SYMMETRIC]
        ratios = [0.10, 0.20]
        strategies = [
            CleaningStrategyType.BASELINE,
            CleaningStrategyType.SIMPLE,
            CleaningStrategyType.ISOLATION_FOREST,
            CleaningStrategyType.CLEANLAB,
        ]
        models = [ModelType.LIGHTGBM, ModelType.LOGISTIC_REGRESSION]
        seeds = [42]
    else:
        print("=" * 60)
        print("STARTING BENCHMARK SUITE")
        print("=" * 60)
        datasets = args.datasets
        noise_types = [NoiseType(nt) for nt in args.noise_types]
        ratios = args.ratios
        strategies = [CleaningStrategyType(s) for s in args.strategies]
        models = [ModelType(m) for m in args.models]
        seeds = args.seeds

    runner.run_suite(
        datasets=datasets,
        noise_types=noise_types,
        noise_ratios=ratios,
        strategies=strategies,
        models=models,
        seeds=seeds,
        chunk_size=args.chunk_size,
        force_rerun=args.force_rerun,
        save_every=args.save_every,
    )
