"""HELOC (Home Equity Line of Credit) dataset loader (OpenML DID 45023)."""

from pathlib import Path
from typing import Optional

import numpy as np
import openml
import pandas as pd

from src.data.base import TabularDataset

OPENML_HELOC_ID = 45023


def load_heloc(
    data_dir: Optional[Path] = None,
    convert_negative_to_nan: bool = True,
    force_download: bool = False,
) -> TabularDataset:
    """Load FICO HELOC benchmark dataset.

    FICO encodes special missing values as negative integers (-7, -8, -9):
    - -9: No Bureau Record
    - -8: No Usable/Established Accounts
    - -7: Condition Not Met / No Inquiries
    Setting `convert_negative_to_nan=True` exposes these natural missing values.

    Parameters
    ----------
    data_dir : Optional[Path]
        Directory where raw parquet files are cached.
    convert_negative_to_nan : bool
        Whether to map negative special codes (-7, -8, -9) to np.nan.
    force_download : bool
        If True, re-download from OpenML even if cache exists.

    Returns
    -------
    TabularDataset
        Standardized dataset object.
    """
    cache_path = (
        (data_dir / "heloc.parquet")
        if data_dir
        else Path("data/raw/heloc.parquet")
    )

    if cache_path.exists() and not force_download:
        df = pd.read_parquet(cache_path)
    else:
        dataset = openml.datasets.get_dataset(OPENML_HELOC_ID)
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        df = X.copy()

        if convert_negative_to_nan:
            # Convert FICO special missing values (-7, -8, -9) to NaN
            for col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

        # Target: RiskPerformance (0 vs 1)
        df["target"] = y.astype(str).str.strip().replace({"1": 1, "0": 0}).astype(int)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    numerical_cols = [c for c in df.columns if c != "target"]
    categorical_cols: list[str] = []

    return TabularDataset(
        name="heloc",
        df=df,
        target_col="target",
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
