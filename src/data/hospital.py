"""Hospital Readmissions benchmark dataset loader (OpenML DID 43874)."""

from pathlib import Path
from typing import Optional

import numpy as np
import openml
import pandas as pd

from src.data.base import TabularDataset

OPENML_HOSPITAL_ID = 43874


def load_hospital(
    data_dir: Optional[Path] = None,
    sample_size: Optional[int] = 50000,
    random_state: int = 42,
    force_download: bool = False,
) -> TabularDataset:
    """Load Diabetes-130-Hospitals dataset for real-world error evaluation.

    Contains real-world medical records with intrinsic missing values,
    disparate categories, and imbalanced readmission targets.

    Parameters
    ----------
    data_dir : Optional[Path]
        Directory where raw parquet files are cached.
    sample_size : Optional[int]
        Target sample size (default: 50000). If None, keeps all 101,766 rows.
    random_state : int
        Seed for sampling.
    force_download : bool
        If True, re-download from OpenML even if cache exists.

    Returns
    -------
    TabularDataset
        Standardized dataset object.
    """
    suffix = f"_{sample_size}" if sample_size else "_full"
    cache_path = (
        (data_dir / f"hospital{suffix}.parquet")
        if data_dir
        else Path(f"data/raw/hospital{suffix}.parquet")
    )

    if cache_path.exists() and not force_download:
        df = pd.read_parquet(cache_path)
        # Check if cache contains leaked columns and fix if needed
        if "readmit_binary" in df.columns or "readmitted" in df.columns:
            if "readmit_binary" in df.columns:
                df["target"] = df["readmit_binary"].astype(int)
            leakage_cols = ["readmitted", "readmit_binary", "readmit_30_days"]
            df = df.drop(columns=[c for c in leakage_cols if c in df.columns and c != "target"])
            df.to_parquet(cache_path, index=False)
    else:
        dataset = openml.datasets.get_dataset(OPENML_HOSPITAL_ID)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        full_df = X.copy()
        # Set target to readmit_binary (clinical readmission)
        if "readmit_binary" in full_df.columns:
            full_df["target"] = (
                full_df["readmit_binary"]
                .astype(str)
                .str.strip()
                .map(lambda v: 1 if v in {"1", "True", "true", "Yes", "yes", "1.0"} else 0)
            )
        else:
            full_df["target"] = (
                y.astype(str)
                .str.strip()
                .map(lambda v: 1 if v in {"1", "True", "true", "Yes", "yes", "1.0"} else 0)
            )

        # Drop leakage columns from features
        leakage_cols = ["readmitted", "readmit_binary", "readmit_30_days"]
        full_df = full_df.drop(
            columns=[c for c in leakage_cols if c in full_df.columns and c != "target"]
        )

        # Standardize missing indicator symbols ('?', 'Unknown', 'Missing')
        full_df = full_df.replace(["?", "Unknown", "Missing", "None"], np.nan)

        if sample_size and sample_size < len(full_df):
            # Stratified subsampling
            pos = full_df[full_df["target"] == 1]
            neg = full_df[full_df["target"] == 0]
            pos_ratio = len(pos) / len(full_df)
            n_pos = int(sample_size * pos_ratio)
            n_neg = sample_size - n_pos

            pos_sample = pos.sample(n=min(n_pos, len(pos)), random_state=random_state)
            neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)
            df = (
                pd.concat([pos_sample, neg_sample], ignore_index=True)
                .sample(frac=1.0, random_state=random_state)
                .reset_index(drop=True)
            )
        else:
            df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    categorical_cols = [
        col
        for col in df.columns
        if col != "target"
        and (
            isinstance(df[col].dtype, pd.CategoricalDtype)
            or df[col].dtype == object
            or pd.api.types.is_string_dtype(df[col])
        )
    ]
    numerical_cols = [
        col for col in df.columns if col != "target" and col not in categorical_cols
    ]

    return TabularDataset(
        name="hospital",
        df=df,
        target_col="target",
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
