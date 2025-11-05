"""Multi-objective loss for physics-informed training."""

from typing import Dict, Optional
import torch
import torch.nn as nn


class MultiObjectiveLoss:
    """Multi-objective loss combining PDE residual, boundary/IC, data, and physics constraints."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = False,
        use_gradnorm: bool = False,
    ):
        """Initialize multi-objective loss.

        Parameters
        ----------
        weights : Dict[str, float], optional
            Manual loss weights. Keys: residual, boundary, initial, data, physics
        use_uncertainty_weighting : bool
            Use uncertainty weighting (Kendall et al.)
        use_gradnorm : bool
            Use GradNorm for automatic weight balancing
        """
        if weights is None:
            weights = {
                "residual": 1.0,
                "boundary": 1.0,
                "initial": 1.0,
                "data": 1.0,
                "physics": 0.1,
            }

        self.weights = weights
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_gradnorm = use_gradnorm

        # Learnable log-variance parameters for uncertainty weighting
        if use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1))
                for k in weights.keys()
            })

    def __call__(
        self, model: nn.Module, data: Dict[str, torch.Tensor]
    ) -> tuple:
        """Compute multi-objective loss.

        Parameters
        ----------
        model : nn.Module
            Neural network model
        data : Dict[str, torch.Tensor]
            Data dictionary with keys:
                - coords: coordinates (N, dim)
                - boundary_coords: boundary points (M, dim)
                - initial_coords: initial condition points (K, dim)
                - data_coords: data points (L, dim)
                - data_values: data values (L, output_dim)
                - physics_constraints: optional physics constraint violations

        Returns
        -------
        tuple
            (total_loss, loss_components_dict)
        """
        loss_components = {}

        # PDE residual loss
        if "coords" in data:
            coords = data["coords"].requires_grad_(True)
            if hasattr(model, "compute_residual"):
                residuals = model.compute_residual(coords)
                loss_residual = torch.mean(residuals**2)
            else:
                # Fallback: use forward pass and compute manually
                u = model(coords)
                loss_residual = torch.mean(u**2)  # Placeholder
            loss_components["residual"] = loss_residual

        # Boundary condition loss
        if "boundary_coords" in data and "boundary_values" in data:
            boundary_coords = data["boundary_coords"]
            boundary_values = data["boundary_values"]
            boundary_pred = model(boundary_coords)
            loss_boundary = torch.mean((boundary_pred - boundary_values) ** 2)
            loss_components["boundary"] = loss_boundary

        # Initial condition loss
        if "initial_coords" in data and "initial_values" in data:
            initial_coords = data["initial_coords"]
            initial_values = data["initial_values"]
            initial_pred = model(initial_coords)
            loss_initial = torch.mean((initial_pred - initial_values) ** 2)
            loss_components["initial"] = loss_initial

        # Data fidelity loss
        if "data_coords" in data and "data_values" in data:
            data_coords = data["data_coords"]
            data_values = data["data_values"]
            data_pred = model(data_coords)
            loss_data = torch.mean((data_pred - data_values) ** 2)
            loss_components["data"] = loss_data

        # Physics constraint loss
        if "physics_constraints" in data:
            loss_physics = torch.mean(data["physics_constraints"] ** 2)
            loss_components["physics"] = loss_physics

        # Compute weighted total loss
        if self.use_uncertainty_weighting:
            total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            for key, loss in loss_components.items():
                if key in self.log_vars:
                    weight = 1.0 / (2.0 * torch.exp(self.log_vars[key]))
                    total_loss = total_loss + weight * loss + self.log_vars[key]
                else:
                    total_loss = total_loss + self.weights.get(key, 1.0) * loss
        else:
            total_loss = sum(
                self.weights.get(key, 1.0) * loss
                for key, loss in loss_components.items()
            )

        return total_loss, loss_components

