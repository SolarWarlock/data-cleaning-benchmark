"""Model training and dual-evaluation (clean and dirty test) module."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelType(str, Enum):
    """Supported machine learning model architectures."""

    LIGHTGBM = "lightgbm"
    MLP = "mlp"
    LOGISTIC_REGRESSION = "logistic_regression"


@dataclass
class EvaluationMetrics:
    """Downstream classification performance metrics."""

    f1_score: float
    f1_macro: float
    roc_auc: float
    accuracy: float


@dataclass
class ModelExecutionResult:
    """Complete summary of model training and dual-test evaluation."""

    model_type: ModelType
    train_time_sec: float
    eval_time_sec: float
    clean_test_metrics: EvaluationMetrics
    dirty_test_metrics: Optional[EvaluationMetrics] = None


class BaseTabularModel(ABC):
    """Abstract base class for downstream tabular classifiers."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @abstractmethod
    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test_clean: pd.DataFrame,
        y_test_clean: pd.Series,
        X_test_dirty: Optional[pd.DataFrame] = None,
        y_test_dirty: Optional[pd.Series] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> ModelExecutionResult:
        """Fit model on training split and evaluate on clean and dirty test splits."""
        pass


class TabularPreprocessor:
    """Robust preprocessor handling scaling and one-hot encoding without leakage."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.encoder: Optional[OneHotEncoder] = None
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> "TabularPreprocessor":
        if numerical_cols is None:
            self.num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        else:
            self.num_cols = [c for c in numerical_cols if c in X.columns]

        if categorical_cols is None:
            self.cat_cols = [c for c in X.columns if c not in self.num_cols]
        else:
            self.cat_cols = [c for c in categorical_cols if c in X.columns]

        if self.num_cols:
            self.scaler.fit(X[self.num_cols].fillna(0.0))

        if self.cat_cols:
            self.encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
            self.encoder.fit(X[self.cat_cols].astype(str).fillna("missing"))

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []

        if self.num_cols:
            num_scaled = self.scaler.transform(X[self.num_cols].fillna(0.0))
            parts.append(num_scaled)

        if self.cat_cols and self.encoder is not None:
            cat_encoded = self.encoder.transform(
                X[self.cat_cols].astype(str).fillna("missing")
            )
            parts.append(cat_encoded)

        if not parts:
            return np.zeros((len(X), 1))

        return np.hstack(parts)


def _compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> EvaluationMetrics:
    """Safely calculate binary classification performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        # If binary probabilities are available
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        elif y_proba.ndim == 1:
            auc = float(roc_auc_score(y_true, y_proba))
        else:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    except Exception:
        auc = 0.5

    return EvaluationMetrics(
        f1_score=f1, f1_macro=f1_m, roc_auc=auc, accuracy=acc
    )


class LightGBMModel(BaseTabularModel):
    """LightGBM classifier implementation."""

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test_clean: pd.DataFrame,
        y_test_clean: pd.Series,
        X_test_dirty: Optional[pd.DataFrame] = None,
        y_test_dirty: Optional[pd.Series] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> ModelExecutionResult:
        preprocessor = TabularPreprocessor().fit(
            X_train, numerical_cols, categorical_cols
        )
        X_train_proc = preprocessor.transform(X_train)
        X_clean_proc = preprocessor.transform(X_test_clean)

        clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=4,
        )

        t_start = time.perf_counter()
        clf.fit(X_train_proc, y_train.values)
        t_train = max(time.perf_counter() - t_start, 1e-4)

        t_eval_start = time.perf_counter()
        clean_pred = clf.predict(X_clean_proc)
        clean_proba = clf.predict_proba(X_clean_proc)
        t_eval = max(time.perf_counter() - t_eval_start, 1e-4)

        clean_metrics = _compute_metrics(y_test_clean, clean_pred, clean_proba)

        dirty_metrics = None
        if X_test_dirty is not None and y_test_dirty is not None:
            X_dirty_proc = preprocessor.transform(X_test_dirty)
            dirty_pred = clf.predict(X_dirty_proc)
            dirty_proba = clf.predict_proba(X_dirty_proc)
            dirty_metrics = _compute_metrics(y_test_dirty, dirty_pred, dirty_proba)

        return ModelExecutionResult(
            model_type=ModelType.LIGHTGBM,
            train_time_sec=t_train,
            eval_time_sec=t_eval,
            clean_test_metrics=clean_metrics,
            dirty_test_metrics=dirty_metrics,
        )


class MLPModel(BaseTabularModel):
    """Multi-Layer Perceptron classifier implementation."""

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test_clean: pd.DataFrame,
        y_test_clean: pd.Series,
        X_test_dirty: Optional[pd.DataFrame] = None,
        y_test_dirty: Optional[pd.Series] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> ModelExecutionResult:
        preprocessor = TabularPreprocessor().fit(
            X_train, numerical_cols, categorical_cols
        )
        X_train_proc = preprocessor.transform(X_train)
        X_clean_proc = preprocessor.transform(X_test_clean)

        clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=self.random_state,
        )

        t_start = time.perf_counter()
        clf.fit(X_train_proc, y_train.values)
        t_train = max(time.perf_counter() - t_start, 1e-4)

        t_eval_start = time.perf_counter()
        clean_pred = clf.predict(X_clean_proc)
        clean_proba = clf.predict_proba(X_clean_proc)
        t_eval = max(time.perf_counter() - t_eval_start, 1e-4)

        clean_metrics = _compute_metrics(y_test_clean, clean_pred, clean_proba)

        dirty_metrics = None
        if X_test_dirty is not None and y_test_dirty is not None:
            X_dirty_proc = preprocessor.transform(X_test_dirty)
            dirty_pred = clf.predict(X_dirty_proc)
            dirty_proba = clf.predict_proba(X_dirty_proc)
            dirty_metrics = _compute_metrics(y_test_dirty, dirty_pred, dirty_proba)

        return ModelExecutionResult(
            model_type=ModelType.MLP,
            train_time_sec=t_train,
            eval_time_sec=t_eval,
            clean_test_metrics=clean_metrics,
            dirty_test_metrics=dirty_metrics,
        )


class LogisticRegressionModel(BaseTabularModel):
    """Logistic Regression classifier implementation."""

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test_clean: pd.DataFrame,
        y_test_clean: pd.Series,
        X_test_dirty: Optional[pd.DataFrame] = None,
        y_test_dirty: Optional[pd.Series] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> ModelExecutionResult:
        preprocessor = TabularPreprocessor().fit(
            X_train, numerical_cols, categorical_cols
        )
        X_train_proc = preprocessor.transform(X_train)
        X_clean_proc = preprocessor.transform(X_test_clean)

        clf = LogisticRegression(
            max_iter=300,
            random_state=self.random_state,
            solver="liblinear",
        )

        t_start = time.perf_counter()
        clf.fit(X_train_proc, y_train.values)
        t_train = max(time.perf_counter() - t_start, 1e-4)

        t_eval_start = time.perf_counter()
        clean_pred = clf.predict(X_clean_proc)
        clean_proba = clf.predict_proba(X_clean_proc)
        t_eval = max(time.perf_counter() - t_eval_start, 1e-4)

        clean_metrics = _compute_metrics(y_test_clean, clean_pred, clean_proba)

        dirty_metrics = None
        if X_test_dirty is not None and y_test_dirty is not None:
            X_dirty_proc = preprocessor.transform(X_test_dirty)
            dirty_pred = clf.predict(X_dirty_proc)
            dirty_proba = clf.predict_proba(X_dirty_proc)
            dirty_metrics = _compute_metrics(y_test_dirty, dirty_pred, dirty_proba)

        return ModelExecutionResult(
            model_type=ModelType.LOGISTIC_REGRESSION,
            train_time_sec=t_train,
            eval_time_sec=t_eval,
            clean_test_metrics=clean_metrics,
            dirty_test_metrics=dirty_metrics,
        )


def get_model(
    model_type: Union[ModelType, str], random_state: int = 42
) -> BaseTabularModel:
    """Factory function for model creation."""
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    if model_type == ModelType.LIGHTGBM:
        return LightGBMModel(random_state=random_state)
    elif model_type == ModelType.MLP:
        return MLPModel(random_state=random_state)
    elif model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegressionModel(random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
