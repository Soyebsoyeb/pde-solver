"""Utility modules."""

from pde_solver.utils.visualization import (
    visualize_solution,
    plot_1d_time_evolution,
    create_animation,
    plot_loss_history,
)
from pde_solver.utils.result_analysis import (
    compute_l2_error,
    compute_linf_error,
    evaluate_solution,
)
from pde_solver.utils.logger import get_logger, StructuredLogger
from pde_solver.utils.exceptions import (
    PDESolverError,
    ConfigurationError,
    ModelError,
    TrainingError,
    NumericalError,
    ValidationError,
    ConvergenceError,
)
from pde_solver.utils.metrics import get_metrics_collector, MetricsCollector
from pde_solver.utils.config_validator import load_config, validate_config, SolverConfig

__all__ = [
    "visualize_solution",
    "plot_1d_time_evolution",
    "create_animation",
    "plot_loss_history",
    "compute_l2_error",
    "compute_linf_error",
    "evaluate_solution",
    "get_logger",
    "StructuredLogger",
    "PDESolverError",
    "ConfigurationError",
    "ModelError",
    "TrainingError",
    "NumericalError",
    "ValidationError",
    "ConvergenceError",
    "get_metrics_collector",
    "MetricsCollector",
    "load_config",
    "validate_config",
    "SolverConfig",
]

