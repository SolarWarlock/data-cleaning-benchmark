"""Central registry and dispatch loader for all benchmark datasets."""

from pathlib import Path
from typing import Callable, Dict, Optional

from src.data.adult import load_adult
from src.data.base import TabularDataset
from src.data.credit_fraud import load_credit_fraud
from src.data.heloc import load_heloc
from src.data.hospital import load_hospital

DATASET_REGISTRY: Dict[str, Callable[..., TabularDataset]] = {
    "adult": load_adult,
    "credit_fraud": load_credit_fraud,
    "heloc": load_heloc,
    "hospital": load_hospital,
}


def load_dataset(
    name: str, data_dir: Optional[Path] = None, force_download: bool = False, **kwargs
) -> TabularDataset:
    """Load a tabular benchmark dataset by identifier.

    Parameters
    ----------
    name : str
        Dataset identifier: 'adult', 'credit_fraud', 'heloc', or 'hospital'.
    data_dir : Optional[Path]
        Target directory for cached parquet files. Defaults to 'data/raw'.
    force_download : bool
        If True, re-downloads even if local cache exists.
    **kwargs : Any
        Additional keyword arguments passed to specific loaders.

    Returns
    -------
    TabularDataset
        Loaded and standardized dataset instance.
    """
    key = name.lower().strip()
    if key not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{name}' is not registered. Available datasets: {available}"
        )

    loader = DATASET_REGISTRY[key]
    return loader(data_dir=data_dir, force_download=force_download, **kwargs)


def load_all_datasets(
    data_dir: Optional[Path] = None, force_download: bool = False
) -> Dict[str, TabularDataset]:
    """Load and cache all registered benchmark datasets.

    Parameters
    ----------
    data_dir : Optional[Path]
        Target cache directory.
    force_download : bool
        Whether to bypass cache.

    Returns
    -------
    Dict[str, TabularDataset]
        Dictionary mapping dataset name to TabularDataset instance.
    """
    datasets = {}
    for name in DATASET_REGISTRY:
        print(f"Loading and validating dataset: {name}...")
        ds = load_dataset(name, data_dir=data_dir, force_download=force_download)
        datasets[name] = ds
        print(f"  [OK] {ds.name}: {ds.metadata.n_rows:,} rows, {ds.metadata.n_features} features")
    return datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preload and cache benchmark datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name ('adult', 'credit_fraud', 'heloc', 'hospital', or 'all')",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Cache directory path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cache exists",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        results = load_all_datasets(data_dir=args.data_dir, force_download=args.force)
        print("\nAll datasets successfully loaded and cached.")
    else:
        ds = load_dataset(args.dataset, data_dir=args.data_dir, force_download=args.force)
        print("\n" + ds.summary())
