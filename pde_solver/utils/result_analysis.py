"""Result analysis and evaluation metrics."""

from typing import Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:  # for type hints only, avoid runtime import
    import torch


def compute_l2_error(
    pred: "torch.Tensor", true: "torch.Tensor"
) -> float:
    """Compute L2 error.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        L2 error
    """
    import torch
    error = pred - true
    l2_error = torch.sqrt(torch.mean(error ** 2))
    return l2_error.item()


def compute_linf_error(
    pred: "torch.Tensor", true: "torch.Tensor"
) -> float:
    """Compute L∞ error.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        L∞ error
    """
    import torch
    error = pred - true
    linf_error = torch.max(torch.abs(error))
    return linf_error.item()


def compute_relative_error(
    pred: "torch.Tensor", true: "torch.Tensor"
) -> float:
    """Compute relative L2 error.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        Relative error
    """
    import torch
    l2_error = compute_l2_error(pred, true)
    l2_true = torch.sqrt(torch.mean(true ** 2))
    relative_error = l2_error / (l2_true.item() + 1e-10)
    return relative_error


def compute_mass_conservation(
    solution: "torch.Tensor", coords: "torch.Tensor"
) -> float:
    """Compute mass conservation violation.

    Parameters
    ----------
    solution : torch.Tensor
        Solution values
    coords : torch.Tensor
        Coordinates

    Returns
    -------
    float
        Mass conservation error
    """
    # Simplified: integral of solution
    # For proper integration, use trapezoidal rule or quadrature
    import torch
    mass = torch.sum(solution)
    return mass.item()


def compute_energy(
    solution: "torch.Tensor", coords: "torch.Tensor"
) -> float:
    """Compute energy (L2 norm squared).

    Parameters
    ----------
    solution : torch.Tensor
        Solution values
    coords : torch.Tensor
        Coordinates

    Returns
    -------
    float
        Energy
    """
    import torch
    energy = torch.sum(solution ** 2)
    return energy.item()


def evaluate_solution(
    pred: "torch.Tensor",
    true: "torch.Tensor",
    coords: Optional["torch.Tensor"] = None,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted solution
    true : torch.Tensor
        True solution
    coords : torch.Tensor, optional
        Coordinates for conservation checks

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics
    """
    metrics = {
        "l2_error": compute_l2_error(pred, true),
        "linf_error": compute_linf_error(pred, true),
        "relative_error": compute_relative_error(pred, true),
    }

    if coords is not None:
        metrics["mass_pred"] = compute_mass_conservation(pred, coords)
        metrics["mass_true"] = compute_mass_conservation(true, coords)
        metrics["mass_error"] = abs(metrics["mass_pred"] - metrics["mass_true"])

        metrics["energy_pred"] = compute_energy(pred, coords)
        metrics["energy_true"] = compute_energy(true, coords)
        metrics["energy_error"] = abs(metrics["energy_pred"] - metrics["energy_true"])

    return metrics

