"""Adult Income dataset loader (OpenML DID 1590)."""

from pathlib import Path
from typing import Optional

import numpy as np
import openml
import pandas as pd

from src.data.base import TabularDataset

OPENML_ADULT_ID = 1590


def load_adult(
    data_dir: Optional[Path] = None, force_download: bool = False
) -> TabularDataset:
    """Load Adult Income benchmark dataset.

    Parameters
    ----------
    data_dir : Optional[Path]
        Directory where raw parquet files are cached.
    force_download : bool
        If True, re-download from OpenML even if cache exists.

    Returns
    -------
    TabularDataset
        Standardized dataset object.
    """
    cache_path = (
        (data_dir / "adult.parquet")
        if data_dir
        else Path("data/raw/adult.parquet")
    )

    if cache_path.exists() and not force_download:
        df = pd.read_parquet(cache_path)
    else:
        dataset = openml.datasets.get_dataset(OPENML_ADULT_ID)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        df = X.copy()
        df["target"] = y

        # Standardize missing value symbols ('?')
        df = df.replace("?", np.nan)

        # Standardize column types: categorical columns as string or category
        for col in df.columns:
            if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype("category")

        # Map binary target: >50K -> 1, <=50K -> 0
        df["target"] = df["target"].apply(
            lambda val: 1 if ">50K" in str(val) else 0
        )

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
        name="adult",
        df=df,
        target_col="target",
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
