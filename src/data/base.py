"""Base data structures and classes for tabular benchmark datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetMetadata:
    """Metadata container for tabular datasets."""

    name: str
    n_rows: int
    n_features: int
    target_column: str
    categorical_columns: List[str]
    numerical_columns: List[str]
    class_distribution: Dict[int, int]
    missing_value_ratio: float

    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "target_column": self.target_column,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "class_distribution": self.class_distribution,
            "missing_value_ratio": self.missing_value_ratio,
        }


class TabularDataset:
    """Standardized representation of a tabular dataset for data cleaning experiments."""

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        target_col: str,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.df = df.copy()
        self.target_col = target_col

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        # Identify feature columns
        feature_cols = [c for c in self.df.columns if c != target_col]

        if categorical_cols is None or numerical_cols is None:
            inferred_cat = []
            inferred_num = []
            for col in feature_cols:
                if (
                    pd.api.types.is_object_dtype(self.df[col])
                    or pd.api.types.is_categorical_dtype(self.df[col])
                    or pd.api.types.is_string_dtype(self.df[col])
                ):
                    inferred_cat.append(col)
                else:
                    inferred_num.append(col)
            self.categorical_cols = categorical_cols or inferred_cat
            self.numerical_cols = numerical_cols or inferred_num
        else:
            self.categorical_cols = categorical_cols
            self.numerical_cols = numerical_cols

        # Ensure binary target encoded as 0 and 1
        self._standardize_target()

        # Compute metadata
        self.metadata = self._compute_metadata()

    def _standardize_target(self) -> None:
        """Ensure binary classification target is mapped to {0, 1}."""
        target_series = self.df[self.target_col]
        unique_vals = target_series.dropna().unique()

        if len(unique_vals) != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 classes, got {len(unique_vals)}: {unique_vals}"
            )

        if set(unique_vals) != {0, 1}:
            val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            self.df[self.target_col] = self.df[self.target_col].map(val_map).astype(int)
        else:
            self.df[self.target_col] = self.df[self.target_col].astype(int)

    def _compute_metadata(self) -> DatasetMetadata:
        """Compute dataset summary statistics."""
        class_counts = self.df[self.target_col].value_counts().to_dict()
        feature_df = self.df[[c for c in self.df.columns if c != self.target_col]]
        total_cells = feature_df.shape[0] * feature_df.shape[1]
        missing_cells = int(feature_df.isnull().sum().sum())
        missing_ratio = float(missing_cells / total_cells) if total_cells > 0 else 0.0

        return DatasetMetadata(
            name=self.name,
            n_rows=len(self.df),
            n_features=len(self.df.columns) - 1,
            target_column=self.target_col,
            categorical_columns=list(self.categorical_cols),
            numerical_columns=list(self.numerical_cols),
            class_distribution=class_counts,
            missing_value_ratio=missing_ratio,
        )

    @property
    def X(self) -> pd.DataFrame:
        """Feature matrix without target column."""
        return self.df[[c for c in self.df.columns if c != self.target_col]]

    @property
    def y(self) -> pd.Series:
        """Target label series."""
        return self.df[self.target_col]

    def split(
        self, test_size: float = 0.2, random_state: int = 42, stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified train-test split."""
        strat = self.y if stratify else None
        return train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

    def save_parquet(self, filepath: Path) -> Path:
        """Save dataset to Parquet format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(filepath, index=False)
        return filepath

    @classmethod
    def from_parquet(
        cls,
        filepath: Path,
        name: str,
        target_col: str,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ) -> "TabularDataset":
        """Load dataset from Parquet format."""
        df = pd.read_parquet(filepath)
        return cls(
            name=name,
            df=df,
            target_col=target_col,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
        )

    def summary(self) -> str:
        """Formatted scientific summary of the dataset."""
        meta = self.metadata
        pos_ratio = meta.class_distribution.get(1, 0) / meta.n_rows * 100
        return (
            f"Dataset: {meta.name}\n"
            f"  Observations: {meta.n_rows:,}\n"
            f"  Features: {meta.n_features} (numerical: {len(meta.numerical_columns)}, categorical: {len(meta.categorical_columns)})\n"
            f"  Target: '{meta.target_column}' (Class 1 ratio: {pos_ratio:.2f}%)\n"
            f"  Missing value ratio: {meta.missing_value_ratio * 100:.2f}%"
        )
