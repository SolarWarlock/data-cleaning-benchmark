"""Data loading and preprocessing module for research datasets."""

from src.data.adult import load_adult
from src.data.base import DatasetMetadata, TabularDataset
from src.data.credit_fraud import load_credit_fraud
from src.data.heloc import load_heloc
from src.data.hospital import load_hospital
from src.data.loader import DATASET_REGISTRY, load_all_datasets, load_dataset

__all__ = [
    "TabularDataset",
    "DatasetMetadata",
    "load_adult",
    "load_credit_fraud",
    "load_heloc",
    "load_hospital",
    "load_dataset",
    "load_all_datasets",
    "DATASET_REGISTRY",
]
