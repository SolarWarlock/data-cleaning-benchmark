"""Credit Card Fraud Detection dataset loader (OpenML DID 1597)."""

from pathlib import Path
from typing import Optional

import openml
import pandas as pd

from src.data.base import TabularDataset

OPENML_CREDIT_ID = 1597


def load_credit_fraud(
    data_dir: Optional[Path] = None,
    sample_size: int = 50000,
    random_state: int = 42,
    force_download: bool = False,
) -> TabularDataset:
    """Load Credit Card Fraud benchmark dataset with controlled stratification.

    Retains all minority positive class instances (fraud) and samples non-fraud
    cases up to `sample_size` to maintain realistic imbalance under hardware limits.

    Parameters
    ----------
    data_dir : Optional[Path]
        Directory where raw parquet files are cached.
    sample_size : int
        Target sample size (default: 50000).
    random_state : int
        Seed for reproducibility.
    force_download : bool
        If True, re-download from OpenML even if cache exists.

    Returns
    -------
    TabularDataset
        Standardized dataset object.
    """
    cache_path = (
        (data_dir / f"credit_fraud_{sample_size}.parquet")
        if data_dir
        else Path(f"data/raw/credit_fraud_{sample_size}.parquet")
    )

    if cache_path.exists() and not force_download:
        df = pd.read_parquet(cache_path)
    else:
        dataset = openml.datasets.get_dataset(OPENML_CREDIT_ID)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        full_df = X.copy()
        target_name = dataset.default_target_attribute or "Class"
        full_df["target"] = y.astype(str).str.strip().replace({"\'1\'": 1, "\'0\'": 0, "1": 1, "0": 0}).astype(int)

        # Separate positive (fraud) and negative (normal)
        fraud = full_df[full_df["target"] == 1]
        normal = full_df[full_df["target"] == 0]

        n_fraud = len(fraud)
        n_normal_target = max(1, sample_size - n_fraud)
        normal_sampled = normal.sample(n=n_normal_target, random_state=random_state)

        df = pd.concat([fraud, normal_sampled], ignore_index=True)
        # Shuffle to break ordering
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    numerical_cols = [c for c in df.columns if c != "target"]
    categorical_cols: list[str] = []

    return TabularDataset(
        name="credit_fraud",
        df=df,
        target_col="target",
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
