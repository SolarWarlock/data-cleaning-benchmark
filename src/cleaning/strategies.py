"""Data cleaning strategies with strict leakage-free fit/transform contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CleaningStrategyType(str, Enum):
    """Enumeration of cleaning strategies."""

    BASELINE = "baseline"
    SIMPLE = "simple"
    ISOLATION_FOREST = "isolation_forest"
    CLEANLAB = "cleanlab"


@dataclass
class CleaningResult:
    """Result of applying a data cleaning strategy."""

    cleaned_X_train: pd.DataFrame
    cleaned_y_train: pd.Series
    cleaned_X_test: pd.DataFrame
    cleaned_y_test: pd.Series
    strategy: CleaningStrategyType
    train_rows_removed: int = 0
    identified_error_indices: Optional[List[int]] = None
    execution_time_sec: float = 0.0
    peak_memory_mb: float = 0.0


class BaseCleaner(ABC):
    """Abstract base class for data cleaning strategies."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @abstractmethod
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> CleaningResult:
        """Fit cleaning strategy on train data and transform train and test splits."""
        pass


class BaselineCleaner(BaseCleaner):
    """Baseline strategy: minimal imputation if necessary, no row removal."""

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> CleaningResult:
        num_cols = (
            numerical_cols
            if numerical_cols is not None
            else [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        )
        cat_cols = (
            categorical_cols
            if categorical_cols is not None
            else [c for c in X_train.columns if c not in num_cols]
        )

        clean_X_train = X_train.copy()
        clean_X_test = X_test.copy()

        # Minimal imputation for numerical (median) and categorical (mode)
        # to ensure compatibility across all downstream classifiers
        if num_cols:
            num_imp = SimpleImputer(strategy="median")
            clean_X_train[num_cols] = num_imp.fit_transform(clean_X_train[num_cols])
            clean_X_test[num_cols] = num_imp.transform(clean_X_test[num_cols])

        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            clean_X_train[cat_cols] = cat_imp.fit_transform(clean_X_train[cat_cols])
            clean_X_test[cat_cols] = cat_imp.transform(clean_X_test[cat_cols])

        return CleaningResult(
            cleaned_X_train=clean_X_train,
            cleaned_y_train=y_train.copy(),
            cleaned_X_test=clean_X_test,
            cleaned_y_test=y_test.copy(),
            strategy=CleaningStrategyType.BASELINE,
            train_rows_removed=0,
        )


class SimpleImputerCleaner(BaseCleaner):
    """Heuristic cleaning: median/mode imputation plus IQR-based numerical clipping."""

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> CleaningResult:
        num_cols = (
            numerical_cols
            if numerical_cols is not None
            else [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        )
        cat_cols = (
            categorical_cols
            if categorical_cols is not None
            else [c for c in X_train.columns if c not in num_cols]
        )

        clean_X_train = X_train.copy()
        clean_X_test = X_test.copy()

        # Step 1: Imputation fitted strictly on train
        if num_cols:
            num_imp = SimpleImputer(strategy="median")
            clean_X_train[num_cols] = num_imp.fit_transform(clean_X_train[num_cols])
            clean_X_test[num_cols] = num_imp.transform(clean_X_test[num_cols])

            # Step 2: IQR bounds computed strictly from train and applied (clipped)
            for col in num_cols:
                q25 = clean_X_train[col].quantile(0.25)
                q75 = clean_X_train[col].quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    clean_X_train[col] = clean_X_train[col].clip(lower, upper)
                    clean_X_test[col] = clean_X_test[col].clip(lower, upper)

        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            clean_X_train[cat_cols] = cat_imp.fit_transform(clean_X_train[cat_cols])
            clean_X_test[cat_cols] = cat_imp.transform(clean_X_test[cat_cols])

        return CleaningResult(
            cleaned_X_train=clean_X_train,
            cleaned_y_train=y_train.copy(),
            cleaned_X_test=clean_X_test,
            cleaned_y_test=y_test.copy(),
            strategy=CleaningStrategyType.SIMPLE,
            train_rows_removed=0,
        )


class IsolationForestCleaner(BaseCleaner):
    """Outlier detection via Isolation Forest; prunes anomalies from train split."""

    def __init__(
        self, contamination: float = 0.05, random_state: int = 42
    ) -> None:
        super().__init__(random_state=random_state)
        self.contamination = contamination

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> CleaningResult:
        # First ensure clean imputations
        base = BaselineCleaner(random_state=self.random_state)
        res = base.fit_transform(
            X_train, y_train, X_test, y_test, numerical_cols, categorical_cols
        )
        clean_X_train = res.cleaned_X_train
        clean_y_train = res.cleaned_y_train
        clean_X_test = res.cleaned_X_test
        clean_y_test = res.cleaned_y_test

        num_cols = (
            numerical_cols
            if numerical_cols is not None
            else [c for c in clean_X_train.columns if pd.api.types.is_numeric_dtype(clean_X_train[c])]
        )

        if not num_cols:
            return CleaningResult(
                cleaned_X_train=clean_X_train,
                cleaned_y_train=clean_y_train,
                cleaned_X_test=clean_X_test,
                cleaned_y_test=clean_y_test,
                strategy=CleaningStrategyType.ISOLATION_FOREST,
                train_rows_removed=0,
            )

        # Scale numerical features for Isolation Forest
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean_X_train[num_cols])

        # Clamp contamination to valid range (0, 0.5]
        effective_contamination = min(max(self.contamination, 0.01), 0.49)
        iso = IsolationForest(
            contamination=effective_contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        preds = iso.fit_predict(X_scaled)  # 1 for inliers, -1 for outliers

        inlier_mask = preds == 1
        outlier_indices = clean_X_train.index[~inlier_mask].tolist()

        pruned_X_train = clean_X_train.iloc[inlier_mask].reset_index(drop=True)
        pruned_y_train = clean_y_train.iloc[inlier_mask].reset_index(drop=True)

        return CleaningResult(
            cleaned_X_train=pruned_X_train,
            cleaned_y_train=pruned_y_train,
            cleaned_X_test=clean_X_test,
            cleaned_y_test=clean_y_test,
            strategy=CleaningStrategyType.ISOLATION_FOREST,
            train_rows_removed=len(outlier_indices),
            identified_error_indices=outlier_indices,
        )


class CleanlabCleaner(BaseCleaner):
    """Confident learning label error detector; prunes label errors from train split."""

    def __init__(self, cv_folds: int = 3, random_state: int = 42) -> None:
        super().__init__(random_state=random_state)
        self.cv_folds = cv_folds

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> CleaningResult:
        from cleanlab.filter import find_label_issues
        from sklearn.model_selection import StratifiedKFold

        base = BaselineCleaner(random_state=self.random_state)
        res = base.fit_transform(
            X_train, y_train, X_test, y_test, numerical_cols, categorical_cols
        )
        clean_X_train = res.cleaned_X_train
        clean_y_train = res.cleaned_y_train
        clean_X_test = res.cleaned_X_test
        clean_y_test = res.cleaned_y_test

        num_cols = (
            numerical_cols
            if numerical_cols is not None
            else [c for c in clean_X_train.columns if pd.api.types.is_numeric_dtype(clean_X_train[c])]
        )

        # Prepare numerical feature matrix for out-of-fold probability estimation
        # If categorical features present, one-hot encode dummy features on train
        X_encoded = pd.get_dummies(clean_X_train, drop_first=True)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_encoded)
        y_arr = clean_y_train.values.astype(int)

        # Out-of-fold predicted probabilities via Logistic Regression
        oof_probs = np.zeros((len(y_arr), 2))
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        for train_idx, val_idx in skf.split(X_norm, y_arr):
            clf = LogisticRegression(
                max_iter=500, random_state=self.random_state, solver="liblinear"
            )
            clf.fit(X_norm[train_idx], y_arr[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_norm[val_idx])

        # Detect label issues via cleanlab
        try:
            issues = find_label_issues(
                labels=y_arr,
                pred_probs=oof_probs,
                return_indices_ranked_by="self_confidence",
            )
            issue_mask = np.zeros(len(y_arr), dtype=bool)
            issue_mask[issues] = True
            identified_indices = clean_X_train.index[issue_mask].tolist()

            # Prune detected corrupted samples from training set
            clean_mask = ~issue_mask
            pruned_X_train = clean_X_train.iloc[clean_mask].reset_index(drop=True)
            pruned_y_train = clean_y_train.iloc[clean_mask].reset_index(drop=True)

            return CleaningResult(
                cleaned_X_train=pruned_X_train,
                cleaned_y_train=pruned_y_train,
                cleaned_X_test=clean_X_test,
                cleaned_y_test=clean_y_test,
                strategy=CleaningStrategyType.CLEANLAB,
                train_rows_removed=len(issues),
                identified_error_indices=identified_indices,
            )
        except Exception:
            # Fallback if cleanlab encountered degenerate probability distribution
            return CleaningResult(
                cleaned_X_train=clean_X_train,
                cleaned_y_train=clean_y_train,
                cleaned_X_test=clean_X_test,
                cleaned_y_test=clean_y_test,
                strategy=CleaningStrategyType.CLEANLAB,
                train_rows_removed=0,
            )


def get_cleaner(
    strategy: Union[CleaningStrategyType, str], random_state: int = 42, **kwargs
) -> BaseCleaner:
    """Factory function for creating cleaning strategy instances."""
    if isinstance(strategy, str):
        strategy = CleaningStrategyType(strategy)

    if strategy == CleaningStrategyType.BASELINE:
        return BaselineCleaner(random_state=random_state)
    elif strategy == CleaningStrategyType.SIMPLE:
        return SimpleImputerCleaner(random_state=random_state)
    elif strategy == CleaningStrategyType.ISOLATION_FOREST:
        contamination = kwargs.get("contamination", 0.05)
        return IsolationForestCleaner(
            contamination=contamination, random_state=random_state
        )
    elif strategy == CleaningStrategyType.CLEANLAB:
        cv_folds = kwargs.get("cv_folds", 3)
        return CleanlabCleaner(cv_folds=cv_folds, random_state=random_state)
    else:
        raise ValueError(f"Unknown cleaning strategy: {strategy}")
