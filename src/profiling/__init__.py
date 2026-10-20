"""Hardware profiling and cost-benefit analysis module."""

from src.profiling.tracker import (
    ResourceProfile,
    compute_break_even_ratio,
    identify_pareto_frontier,
    track_resources,
)

__all__ = [
    "ResourceProfile",
    "track_resources",
    "compute_break_even_ratio",
    "identify_pareto_frontier",
]
