"""Experiment runner and pipeline orchestrator with checkpointing and chunked execution."""

import argparse
import itertools
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cleaning import CleaningStrategyType, get_cleaner
from src.data.loader import load_dataset
from src.models import ModelType, get_model
from src.noise import NoiseInjector, NoiseType
from src.profiling import compute_break_even_ratio, track_resources


@dataclass
class ExperimentRecord:
    """Record of a single experimental run."""

    dataset: str
    noise_type: str
    noise_ratio: float
    pipeline_mode: str  # 'ETL' or 'ELT'
    strategy: str
    model: str
    seed: int
    train_size: int
    test_size: int
    cleaning_time_sec: float
    cleaning_cpu_sec: float
    cleaning_peak_ram_mb: float
    train_rows_removed: int
    model_train_time_sec: float
    model_eval_time_sec: float
    clean_f1: float
    clean_f1_macro: float
    clean_roc_auc: float
    clean_accuracy: float
    dirty_f1: Optional[float] = None
    dirty_f1_macro: Optional[float] = None
    dirty_roc_auc: Optional[float] = None
    dirty_accuracy: Optional[float] = None
    delta_f1: float = 0.0
    delta_auc: float = 0.0
    break_even_ratio: float = 0.0
    is_pareto: bool = False


class BenchmarkRunner:
    """Orchestrates controlled cost-benefit experiments across datasets and models.

    Supports chunked execution, idempotent restarts, and atomic per-run persistence.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        random_seeds: Optional[List[int]] = None,
    ) -> None:
        self.output_dir = output_dir or Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seeds = random_seeds or [42, 43, 44]
        self.parquet_path = self.output_dir / "experiments_results.parquet"
        self.csv_path = self.output_dir / "experiments_results.csv"

    def get_completed_keys(self) -> Set[Tuple[str, str, float, str, str, str, int]]:
        """Return set of completed experiment keys to avoid redundant computation."""
        if not self.parquet_path.exists():
            return set()

        try:
            df = pd.read_parquet(self.parquet_path)
            keys = set(
                zip(
                    df["dataset"],
                    df["noise_type"],
                    df["noise_ratio"],
                    df["strategy"],
                    df["model"],
                    df["pipeline_mode"],
                    df["seed"],
                )
            )
            return keys
        except Exception:
            return set()

    def run_single_experiment(
        self,
        dataset_name: str,
        noise_type: NoiseType,
        noise_ratio: float,
        strategy_type: CleaningStrategyType,
        model_type: ModelType,
        pipeline_mode: str = "ELT",
        seed: int = 42,
    ) -> ExperimentRecord:
        """Run an isolated pipeline combination."""
        # 1. Load dataset
        dataset = load_dataset(dataset_name)

        # 2. Train-test split (80/20 stratified)
        X_train, X_test, y_train, y_test = dataset.split(
            test_size=0.2, random_state=seed, stratify=True
        )

        injector = NoiseInjector(random_state=seed)

        # 3. Inject noise
        if dataset_name == "hospital":
            if noise_ratio > 0:
                artifact_train = injector.inject(
                    X_train,
                    y_train,
                    noise_type,
                    ratio=noise_ratio,
                    numerical_cols=dataset.numerical_cols,
                    categorical_cols=dataset.categorical_cols,
                )
                noisy_X_train = artifact_train.corrupted_X
                noisy_y_train = artifact_train.corrupted_y
            else:
                noisy_X_train = X_train
                noisy_y_train = y_train

            artifact_test = injector.inject(
                X_test,
                y_test,
                noise_type,
                ratio=noise_ratio,
                numerical_cols=dataset.numerical_cols,
                categorical_cols=dataset.categorical_cols,
            )
            dirty_X_test = artifact_test.corrupted_X
            dirty_y_test = artifact_test.corrupted_y
        else:
            artifact_train = injector.inject(
                X_train,
                y_train,
                noise_type,
                ratio=noise_ratio,
                numerical_cols=dataset.numerical_cols,
                categorical_cols=dataset.categorical_cols,
            )
            noisy_X_train = artifact_train.corrupted_X
            noisy_y_train = artifact_train.corrupted_y

            artifact_test = injector.inject(
                X_test,
                y_test,
                noise_type,
                ratio=noise_ratio,
                numerical_cols=dataset.numerical_cols,
                categorical_cols=dataset.categorical_cols,
            )
            dirty_X_test = artifact_test.corrupted_X
            dirty_y_test = artifact_test.corrupted_y

        clean_X_test = X_test.copy()
        clean_y_test = y_test.copy()

        # 4. Clean data with resource tracking
        cleaner = get_cleaner(strategy_type, random_state=seed)

        with track_resources() as prof:
            cleaning_result = cleaner.fit_transform(
                X_train=noisy_X_train,
                y_train=noisy_y_train,
                X_test=clean_X_test,
                y_test=clean_y_test,
                numerical_cols=dataset.numerical_cols,
                categorical_cols=dataset.categorical_cols,
            )

        # 5. Fit downstream model and evaluate
        model = get_model(model_type, random_state=seed)
        model_res = model.fit_and_evaluate(
            X_train=cleaning_result.cleaned_X_train,
            y_train=cleaning_result.cleaned_y_train,
            X_test_clean=cleaning_result.cleaned_X_test,
            y_test_clean=cleaning_result.cleaned_y_test,
            X_test_dirty=dirty_X_test,
            y_test_dirty=dirty_y_test,
            numerical_cols=dataset.numerical_cols,
            categorical_cols=dataset.categorical_cols,
        )

        return ExperimentRecord(
            dataset=dataset_name,
            noise_type=noise_type.value,
            noise_ratio=noise_ratio,
            pipeline_mode=pipeline_mode,
            strategy=strategy_type.value,
            model=model_type.value,
            seed=seed,
            train_size=len(cleaning_result.cleaned_X_train),
            test_size=len(cleaning_result.cleaned_X_test),
            cleaning_time_sec=prof.duration_sec,
            cleaning_cpu_sec=prof.cpu_time_sec,
            cleaning_peak_ram_mb=prof.peak_memory_mb,
            train_rows_removed=cleaning_result.train_rows_removed,
            model_train_time_sec=model_res.train_time_sec,
            model_eval_time_sec=model_res.eval_time_sec,
            clean_f1=model_res.clean_test_metrics.f1_score,
            clean_f1_macro=model_res.clean_test_metrics.f1_macro,
            clean_roc_auc=model_res.clean_test_metrics.roc_auc,
            clean_accuracy=model_res.clean_test_metrics.accuracy,
            dirty_f1=(
                model_res.dirty_test_metrics.f1_score
                if model_res.dirty_test_metrics
                else None
            ),
            dirty_f1_macro=(
                model_res.dirty_test_metrics.f1_macro
                if model_res.dirty_test_metrics
                else None
            ),
            dirty_roc_auc=(
                model_res.dirty_test_metrics.roc_auc
                if model_res.dirty_test_metrics
                else None
            ),
            dirty_accuracy=(
                model_res.dirty_test_metrics.accuracy
                if model_res.dirty_test_metrics
                else None
            ),
        )

    def run_suite(
        self,
        datasets: List[str],
        noise_types: List[NoiseType],
        noise_ratios: List[float],
        strategies: List[CleaningStrategyType],
        models: List[ModelType],
        pipeline_modes: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        chunk_size: Optional[int] = None,
        force_rerun: bool = False,
        save_every: int = 1,
    ) -> pd.DataFrame:
        """Run experimental suite with checkpointing and chunked batching.

        Parameters
        ----------
        datasets : List[str]
            Datasets to process.
        noise_types : List[NoiseType]
            Noise types to evaluate.
        noise_ratios : List[float]
            Corruption ratios.
        strategies : List[CleaningStrategyType]
            Strategies to run.
        models : List[ModelType]
            Classifiers to evaluate.
        pipeline_modes : Optional[List[str]]
            Pipeline architectures ('ELT' or 'ETL').
        seeds : Optional[List[int]]
            Seeds for replication.
        chunk_size : Optional[int]
            If specified, run at most this number of experiments in this session.
        force_rerun : bool
            If True, recomputes experiments already in disk cache.
        save_every : int
            Save results after every N experiments (default: 1 for atomic safety).
        """
        modes = pipeline_modes or ["ELT"]
        run_seeds = seeds or self.random_seeds

        all_combinations = list(
            itertools.product(
                datasets,
                noise_types,
                noise_ratios,
                strategies,
                models,
                modes,
                run_seeds,
            )
        )

        completed_keys = set() if force_rerun else self.get_completed_keys()
        existing_records: List[Dict[str, Any]] = []

        if self.parquet_path.exists() and not force_rerun:
            try:
                df_existing = pd.read_parquet(self.parquet_path)
                existing_records = df_existing.to_dict(orient="records")
            except Exception:
                existing_records = []

        # Filter pending combinations
        pending_combinations = []
        for c in all_combinations:
            ds, nt, nr, strat, m, pmode, s = c
            key = (ds, nt.value, float(nr), strat.value, m.value, pmode, int(s))
            if key not in completed_keys:
                pending_combinations.append(c)

        total_grid = len(all_combinations)
        already_done = total_grid - len(pending_combinations)
        print(f"Benchmark status: {already_done}/{total_grid} already completed on disk.")

        if not pending_combinations:
            print("All requested experiments are already completed. Nothing to run.")
            return pd.DataFrame(existing_records)

        # Apply chunk size limit if specified
        if chunk_size and chunk_size > 0:
            to_execute = pending_combinations[:chunk_size]
            print(f"Chunked batch execution: processing {len(to_execute)} of {len(pending_combinations)} remaining tasks.")
        else:
            to_execute = pending_combinations
            print(f"Executing full remaining batch: {len(to_execute)} tasks.")

        current_records = list(existing_records)
        pbar = tqdm(to_execute, desc="Chunk progress")

        for idx, (ds, nt, nr, strat, m, pmode, s) in enumerate(pbar):
            pbar.set_postfix(
                dataset=ds, noise=nt.value, ratio=f"{nr:.2f}", strat=strat.value, model=m.value
            )
            try:
                rec = self.run_single_experiment(
                    dataset_name=ds,
                    noise_type=nt,
                    noise_ratio=nr,
                    strategy_type=strat,
                    model_type=m,
                    pipeline_mode=pmode,
                    seed=s,
                )
                current_records.append(asdict(rec))
            except Exception as e:
                print(f"\n[ERROR] Combination ({ds}, {nt}, {nr}, {strat}, {m}, seed {s}) failed: {e}")
                continue

            # Save immediately on interval or at chunk end
            if (idx + 1) % save_every == 0 or (idx + 1) == len(to_execute):
                df_temp = pd.DataFrame(current_records)
                df_temp = self._calculate_derived_metrics(df_temp)
                df_temp.to_parquet(self.parquet_path, index=False)
                df_temp.to_csv(self.csv_path, index=False)

        print(f"\nBatch completed. Total records saved: {len(current_records)} in {self.parquet_path}")
        return pd.DataFrame(current_records)

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate delta F1, delta AUC, break-even ratio, and Pareto optimality."""
        if df.empty:
            return df

        # Baseline lookup dictionary: (dataset, noise_type, noise_ratio, model, pipeline_mode, seed) -> (f1, auc)
        baseline_mask = df["strategy"] == CleaningStrategyType.BASELINE.value
        baselines = (
            df[baseline_mask]
            .set_index(["dataset", "noise_type", "noise_ratio", "model", "pipeline_mode", "seed"])[
                ["clean_f1", "clean_roc_auc"]
            ]
            .to_dict(orient="index")
        )

        delta_f1_list = []
        delta_auc_list = []
        break_even_list = []

        for _, row in df.iterrows():
            key = (
                row["dataset"],
                row["noise_type"],
                float(row["noise_ratio"]),
                row["model"],
                row["pipeline_mode"],
                int(row["seed"]),
            )
            base_vals = baselines.get(key)
            if base_vals:
                base_f1 = base_vals["clean_f1"]
                base_auc = base_vals["clean_roc_auc"]
                d_f1 = row["clean_f1"] - base_f1
                d_auc = row["clean_roc_auc"] - base_auc
            else:
                d_f1 = 0.0
                d_auc = 0.0

            be_ratio = compute_break_even_ratio(
                row["clean_f1"],
                row["clean_f1"] - d_f1,
                row["cleaning_cpu_sec"],
            )
            delta_f1_list.append(d_f1)
            delta_auc_list.append(d_auc)
            break_even_list.append(be_ratio)

        df["delta_f1"] = delta_f1_list
        df["delta_auc"] = delta_auc_list
        df["break_even_ratio"] = break_even_list

        return df


def parse_args() -> argparse.Namespace:
    """Parse command line parameters."""
    parser = argparse.ArgumentParser(description="Run benchmark suite in controlled chunks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "credit_fraud", "heloc", "hospital"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=["mcar", "mnar", "outliers", "label_symmetric", "label_asymmetric"],
        help="Noise types to inject",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.40],
        help="Noise ratios",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["baseline", "simple", "isolation_forest", "cleanlab"],
        help="Cleaning strategies",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "mlp", "logistic_regression"],
        help="Downstream models",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for replication",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Execute at most N experiments in this run",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached results and rerun all combinations",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Persist results after every N experiments (default: 1)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot configuration",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show benchmark progress status on disk and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    runner = BenchmarkRunner()

    if args.status:
        completed = runner.get_completed_keys()
        print(f"Total experiments recorded on disk: {len(completed)}")
        if runner.parquet_path.exists():
            df = pd.read_parquet(runner.parquet_path)
            print("\nBreakdown by dataset:")
            print(df["dataset"].value_counts())
            print("\nBreakdown by strategy:")
            print(df["strategy"].value_counts())
            print("\nBreakdown by noise type:")
            print(df["noise_type"].value_counts())
        exit(0)

    if args.pilot:
        print("Configuring PILOT parameters...")
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
