"""Hardware resource profiling and cost-benefit metric tracker."""

import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import psutil


@dataclass
class ResourceProfile:
    """Hardware resource consumption snapshot."""

    duration_sec: float
    cpu_time_sec: float
    peak_memory_mb: float
    memory_delta_mb: float
    energy_kwh: Optional[float] = None


@contextmanager
def track_resources() -> Generator[ResourceProfile, None, None]:
    """Context manager to measure execution wall time, CPU time, and peak RAM."""
    gc.collect()
    process = psutil.Process(os.getpid())

    start_rss = process.memory_info().rss / (1024 * 1024)
    start_wall = time.perf_counter()
    cpu_times_start = process.cpu_times()
    start_cpu = cpu_times_start.user + cpu_times_start.system

    profile = ResourceProfile(
        duration_sec=0.0,
        cpu_time_sec=0.0,
        peak_memory_mb=start_rss,
        memory_delta_mb=0.0,
        energy_kwh=None,
    )

    try:
        yield profile
    finally:
        end_wall = time.perf_counter()
        cpu_times_end = process.cpu_times()
        end_cpu = cpu_times_end.user + cpu_times_end.system
        end_rss = process.memory_info().rss / (1024 * 1024)

        duration = max(end_wall - start_wall, 1e-6)
        cpu_duration = max(end_cpu - start_cpu, 1e-6)

        profile.duration_sec = duration
        profile.cpu_time_sec = cpu_duration
        profile.peak_memory_mb = max(start_rss, end_rss)
        profile.memory_delta_mb = max(0.0, end_rss - start_rss)


def compute_break_even_ratio(
    f1_cleaned: float, f1_baseline: float, cpu_seconds: float
) -> float:
    """Compute break-even efficiency index: delta F1 per CPU second spent.

    Parameters
    ----------
    f1_cleaned : float
        F1-score obtained after applying cleaning strategy.
    f1_baseline : float
        F1-score obtained on baseline uncleaned dataset.
    cpu_seconds : float
        Total CPU seconds expended on cleaning.

    Returns
    -------
    float
        Break-even index. Values <= 0 indicate negative or unremunerative return.
    """
    delta_f1 = f1_cleaned - f1_baseline
    effective_seconds = max(cpu_seconds, 1e-4)
    return float(delta_f1 / effective_seconds)


def identify_pareto_frontier(
    costs: np.ndarray, qualities: np.ndarray
) -> np.ndarray:
    """Identify Pareto-optimal points minimizing cost while maximizing quality.

    Parameters
    ----------
    costs : np.ndarray
        1D array of resource costs (e.g. CPU seconds or peak RAM).
    qualities : np.ndarray
        1D array of downstream qualities (e.g. F1-score or delta F1).

    Returns
    -------
    np.ndarray
        Boolean array indicating True for Pareto-optimal points.
    """
    n_points = len(costs)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j and is_pareto[i]:
                # j dominates i if j has <= cost and >= quality (with at least one strict inequality)
                if (costs[j] <= costs[i] and qualities[j] >= qualities[i]) and (
                    costs[j] < costs[i] or qualities[j] > qualities[i]
                ):
                    is_pareto[i] = False
                    break

    return is_pareto
