"""Synthetic noise injection module with precise ground-truth tracking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class NoiseType(str, Enum):
    """Enumeration of supported noise and error types."""

    MCAR = "mcar"
    MNAR = "mnar"
    OUTLIERS = "outliers"
    LABEL_SYMMETRIC = "label_symmetric"
    LABEL_ASYMMETRIC = "label_asymmetric"


@dataclass
class NoiseArtifact:
    """Container for corrupted dataset and exact ground-truth mask."""

    corrupted_X: pd.DataFrame
    corrupted_y: pd.Series
    clean_X: pd.DataFrame
    clean_y: pd.Series
    noise_type: NoiseType
    noise_ratio: float
    feature_corruption_mask: Optional[pd.DataFrame] = None
    label_corruption_mask: Optional[pd.Series] = None
    metadata: Dict[str, Union[int, float, str]] = field(default_factory=dict)

    @property
    def corrupted_feature_cells(self) -> int:
        """Count total corrupted cells in features."""
        if self.feature_corruption_mask is not None:
            return int(self.feature_corruption_mask.values.sum())
        return 0

    @property
    def corrupted_label_count(self) -> int:
        """Count total corrupted labels."""
        if self.label_corruption_mask is not None:
            return int(self.label_corruption_mask.sum())
        return 0


class NoiseInjector:
    """Injects controlled synthetic noise into tabular data with exact ground truth."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def inject(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        noise_type: Union[NoiseType, str],
        ratio: float,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> NoiseArtifact:
        """Inject specified error type into features or labels.

        Parameters
        ----------
        X : pd.DataFrame
            Clean feature dataframe.
        y : pd.Series
            Clean target series (binary {0, 1}).
        noise_type : Union[NoiseType, str]
            Type of error to inject.
        ratio : float
            Proportion of corruption (e.g. 0.05, 0.10, 0.20, 0.40).
        numerical_cols : Optional[List[str]]
            List of numerical column names.
        categorical_cols : Optional[List[str]]
            List of categorical column names.

        Returns
        -------
        NoiseArtifact
            Container with corrupted data and ground truth boolean masks.
        """
        if isinstance(noise_type, str):
            noise_type = NoiseType(noise_type)

        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Noise ratio must be in [0, 1], got {ratio}")

        if numerical_cols is None:
            numerical_cols = [
                c
                for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c])
            ]
        if categorical_cols is None:
            categorical_cols = [c for c in X.columns if c not in numerical_cols]

        corrupted_X = X.copy()
        if numerical_cols:
            for col in numerical_cols:
                if col in corrupted_X.columns and not pd.api.types.is_float_dtype(corrupted_X[col]):
                    corrupted_X[col] = corrupted_X[col].astype(float)

        if noise_type == NoiseType.MCAR:
            return self._inject_mcar(corrupted_X, y, ratio)
        elif noise_type == NoiseType.MNAR:
            return self._inject_mnar(corrupted_X, y, ratio, numerical_cols)
        elif noise_type == NoiseType.OUTLIERS:
            return self._inject_outliers(corrupted_X, y, ratio, numerical_cols)
        elif noise_type == NoiseType.LABEL_SYMMETRIC:
            return self._inject_label_symmetric(X, y, ratio)
        elif noise_type == NoiseType.LABEL_ASYMMETRIC:
            return self._inject_label_asymmetric(X, y, ratio)
        else:
            raise NotImplementedError(f"Unsupported noise type: {noise_type}")

    def _inject_mcar(
        self, X: pd.DataFrame, y: pd.Series, ratio: float
    ) -> NoiseArtifact:
        """Inject Missing Completely At Random (MCAR) across all feature cells."""
        corrupted_X = X.copy()
        n_rows, n_cols = corrupted_X.shape
        total_cells = n_rows * n_cols

        # Mask of originally missing cells
        orig_missing = corrupted_X.isna().values

        # Sample cells from currently non-missing cells
        valid_indices = np.where(~orig_missing)
        n_valid = len(valid_indices[0])
        n_to_corrupt = int(n_valid * ratio)

        chosen_idx = self.rng.choice(n_valid, size=n_to_corrupt, replace=False)
        target_rows = valid_indices[0][chosen_idx]
        target_cols = valid_indices[1][chosen_idx]

        corruption_matrix = np.zeros((n_rows, n_cols), dtype=bool)
        corruption_matrix[target_rows, target_cols] = True

        for r, c in zip(target_rows, target_cols):
            corrupted_X.iat[r, c] = np.nan

        mask_df = pd.DataFrame(
            corruption_matrix, index=corrupted_X.index, columns=corrupted_X.columns
        )

        return NoiseArtifact(
            corrupted_X=corrupted_X,
            corrupted_y=y.copy(),
            clean_X=X.copy(),
            clean_y=y.copy(),
            noise_type=NoiseType.MCAR,
            noise_ratio=ratio,
            feature_corruption_mask=mask_df,
            metadata={
                "injected_missing_cells": n_to_corrupt,
                "total_cells": total_cells,
                "effective_ratio": n_to_corrupt / total_cells if total_cells else 0.0,
            },
        )

    def _inject_mnar(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ratio: float,
        numerical_cols: List[str],
    ) -> NoiseArtifact:
        """Inject Missing Not At Random (MNAR) conditioned on feature magnitude."""
        corrupted_X = X.copy()
        n_rows, n_cols = corrupted_X.shape
        corruption_matrix = np.zeros((n_rows, n_cols), dtype=bool)

        target_cols = numerical_cols if len(numerical_cols) > 0 else list(X.columns)

        for col_name in target_cols:
            col_idx = corrupted_X.columns.get_loc(col_name)
            series = corrupted_X[col_name]
            valid_mask = ~series.isna()
            valid_vals = series[valid_mask]

            if len(valid_vals) == 0:
                continue

            # In MNAR, higher probability of missingness for values above median
            try:
                median_val = float(valid_vals.median())
                probs = np.where(valid_vals >= median_val, 0.8, 0.2)
                probs = probs / probs.sum()

                n_to_corrupt = int(len(valid_vals) * ratio)
                if n_to_corrupt > 0:
                    chosen_pos = self.rng.choice(
                        len(valid_vals), size=n_to_corrupt, replace=False, p=probs
                    )
                    chosen_orig_indices = valid_vals.index[chosen_pos]

                    corrupted_X.loc[chosen_orig_indices, col_name] = np.nan
                    row_positions = [
                        corrupted_X.index.get_loc(idx) for idx in chosen_orig_indices
                    ]
                    corruption_matrix[row_positions, col_idx] = True
            except Exception:
                continue

        mask_df = pd.DataFrame(
            corruption_matrix, index=corrupted_X.index, columns=corrupted_X.columns
        )

        return NoiseArtifact(
            corrupted_X=corrupted_X,
            corrupted_y=y.copy(),
            clean_X=X.copy(),
            clean_y=y.copy(),
            noise_type=NoiseType.MNAR,
            noise_ratio=ratio,
            feature_corruption_mask=mask_df,
            metadata={"target_columns": len(target_cols)},
        )

    def _inject_outliers(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ratio: float,
        numerical_cols: List[str],
    ) -> NoiseArtifact:
        """Inject heavy-tailed extreme outliers into numerical features."""
        corrupted_X = X.copy()
        n_rows, n_cols = corrupted_X.shape
        corruption_matrix = np.zeros((n_rows, n_cols), dtype=bool)

        if not numerical_cols:
            return NoiseArtifact(
                corrupted_X=corrupted_X,
                corrupted_y=y.copy(),
                clean_X=X.copy(),
                clean_y=y.copy(),
                noise_type=NoiseType.OUTLIERS,
                noise_ratio=ratio,
                feature_corruption_mask=pd.DataFrame(
                    corruption_matrix, index=corrupted_X.index, columns=corrupted_X.columns
                ),
            )

        n_rows_to_corrupt = int(n_rows * ratio)
        if n_rows_to_corrupt > 0:
            target_rows = self.rng.choice(n_rows, size=n_rows_to_corrupt, replace=False)

            for r in target_rows:
                # Corrupt 1 to 3 random numerical features for the row
                n_features_to_corrupt = self.rng.randint(
                    1, min(4, len(numerical_cols) + 1)
                )
                chosen_cols = self.rng.choice(
                    numerical_cols, size=n_features_to_corrupt, replace=False
                )

                for col in chosen_cols:
                    col_idx = corrupted_X.columns.get_loc(col)
                    val = corrupted_X.iat[r, col_idx]
                    if pd.notna(val) and isinstance(val, (int, float, np.number)):
                        col_series = corrupted_X[col].dropna()
                        std = float(col_series.std()) if col_series.std() > 0 else 1.0
                        mean = float(col_series.mean())

                        # Extreme perturbation: 4 to 8 standard deviations
                        direction = self.rng.choice([-1.0, 1.0])
                        magnitude = self.rng.uniform(4.0, 8.0) * std
                        corrupted_val = mean + direction * magnitude

                        corrupted_X.iat[r, col_idx] = corrupted_val
                        corruption_matrix[r, col_idx] = True

        mask_df = pd.DataFrame(
            corruption_matrix, index=corrupted_X.index, columns=corrupted_X.columns
        )

        return NoiseArtifact(
            corrupted_X=corrupted_X,
            corrupted_y=y.copy(),
            clean_X=X.copy(),
            clean_y=y.copy(),
            noise_type=NoiseType.OUTLIERS,
            noise_ratio=ratio,
            feature_corruption_mask=mask_df,
            metadata={"corrupted_rows": n_rows_to_corrupt},
        )

    def _inject_label_symmetric(
        self, X: pd.DataFrame, y: pd.Series, ratio: float
    ) -> NoiseArtifact:
        """Inject symmetric label noise by flipping random labels."""
        corrupted_y = y.copy().astype(int)
        n = len(corrupted_y)
        n_flips = int(n * ratio)

        flip_indices = self.rng.choice(n, size=n_flips, replace=False)
        label_mask = pd.Series(False, index=corrupted_y.index)

        iloc_indices = corrupted_y.iloc[flip_indices].index
        corrupted_y.loc[iloc_indices] = 1 - corrupted_y.loc[iloc_indices]
        label_mask.loc[iloc_indices] = True

        return NoiseArtifact(
            corrupted_X=X.copy(),
            corrupted_y=corrupted_y,
            clean_X=X.copy(),
            clean_y=y.copy(),
            noise_type=NoiseType.LABEL_SYMMETRIC,
            noise_ratio=ratio,
            label_corruption_mask=label_mask,
            metadata={"flipped_labels": n_flips, "flip_ratio": n_flips / n if n else 0},
        )

    def _inject_label_asymmetric(
        self, X: pd.DataFrame, y: pd.Series, ratio: float
    ) -> NoiseArtifact:
        """Inject class-conditional label noise (class 1 flipped to 0 at higher rate)."""
        corrupted_y = y.copy().astype(int)
        label_mask = pd.Series(False, index=corrupted_y.index)

        pos_indices = corrupted_y[corrupted_y == 1].index
        neg_indices = corrupted_y[corrupted_y == 0].index

        # Positive class flipped with probability `ratio`
        n_pos_flips = int(len(pos_indices) * ratio)
        if n_pos_flips > 0 and len(pos_indices) > 0:
            chosen_pos = self.rng.choice(pos_indices, size=n_pos_flips, replace=False)
            corrupted_y.loc[chosen_pos] = 0
            label_mask.loc[chosen_pos] = True

        # Negative class flipped with lower probability `ratio * 0.2`
        n_neg_flips = int(len(neg_indices) * (ratio * 0.2))
        if n_neg_flips > 0 and len(neg_indices) > 0:
            chosen_neg = self.rng.choice(neg_indices, size=n_neg_flips, replace=False)
            corrupted_y.loc[chosen_neg] = 1
            label_mask.loc[chosen_neg] = True

        total_flips = int(label_mask.sum())

        return NoiseArtifact(
            corrupted_X=X.copy(),
            corrupted_y=corrupted_y,
            clean_X=X.copy(),
            clean_y=y.copy(),
            noise_type=NoiseType.LABEL_ASYMMETRIC,
            noise_ratio=ratio,
            label_corruption_mask=label_mask,
            metadata={
                "flipped_pos": n_pos_flips,
                "flipped_neg": n_neg_flips,
                "total_flips": total_flips,
            },
        )
