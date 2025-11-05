"""Physics constraints for hard/soft enforcement in PDE solving."""

from typing import Dict, Callable, Optional
import torch
import numpy as np


class PhysicsConstraints:
    """Manages physics constraints (conservation laws, boundary conditions, etc.)."""

    def __init__(self):
        """Initialize physics constraints."""
        self.constraints: Dict[str, Callable] = {}

    def register_constraint(
        self, name: str, constraint_fn: Callable, weight: float = 1.0
    ) -> None:
        """Register a physics constraint.

        Parameters
        ----------
        name : str
            Constraint name
        constraint_fn : Callable
            Function computing constraint violation (returns tensor)
        weight : float
            Weight for soft constraint enforcement
        """
        self.constraints[name] = {"fn": constraint_fn, "weight": weight}

    def compute_violations(
        self, predictions: torch.Tensor, inputs: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute all constraint violations.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions
        inputs : torch.Tensor
            Input coordinates
        **kwargs
            Additional arguments for constraint functions

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of constraint violations
        """
        violations = {}
        for name, constraint_info in self.constraints.items():
            violations[name] = constraint_info["fn"](predictions, inputs, **kwargs)
        return violations

    def compute_loss(
        self, predictions: torch.Tensor, inputs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute total constraint loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions
        inputs : torch.Tensor
            Input coordinates
        **kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Total weighted constraint loss
        """
        violations = self.compute_violations(predictions, inputs, **kwargs)
        total_loss = torch.tensor(0.0, device=predictions.device)

        for name, violation in violations.items():
            weight = self.constraints[name]["weight"]
            # L2 norm of violation
            loss = weight * torch.mean(violation**2)
            total_loss = total_loss + loss

        return total_loss

    # Common constraint implementations

    @staticmethod
    def mass_conservation(predictions: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute mass conservation violation (divergence).

        For incompressible flow: div(u) = 0

        Parameters
        ----------
        predictions : torch.Tensor
            Velocity field [u, v] or [u, v, w]
        inputs : torch.Tensor
            Spatial coordinates [x, y] or [x, y, z]

        Returns
        -------
        torch.Tensor
            Divergence field
        """
        if inputs.requires_grad:
            # Compute divergence via autograd
            u = predictions[..., 0]
            v = predictions[..., 1] if predictions.shape[-1] > 1 else torch.zeros_like(u)

            du_dx = torch.autograd.grad(
                u.sum(), inputs, create_graph=True, retain_graph=True
            )[0][..., 0]
            dv_dy = torch.autograd.grad(
                v.sum(), inputs, create_graph=True, retain_graph=True
            )[0][..., 1]

            return du_dx + dv_dy
        else:
            # Numerical approximation (finite differences)
            return torch.zeros_like(predictions[..., 0])

    @staticmethod
    def energy_conservation(
        predictions: torch.Tensor, inputs: torch.Tensor, dt: float = 1e-3
    ) -> torch.Tensor:
        """Compute energy conservation violation.

        Parameters
        ----------
        predictions : torch.Tensor
            Field predictions
        inputs : torch.Tensor
            Coordinates including time
        dt : float
            Time step

        Returns
        -------
        torch.Tensor
            Energy change rate
        """
        # Simplified: assumes energy = ||u||^2
        energy = torch.sum(predictions**2, dim=-1)
        if inputs.shape[-1] > 2:  # Has time dimension
            # Approximate dE/dt
            dE_dt = torch.autograd.grad(
                energy.sum(), inputs, create_graph=True, retain_graph=True
            )[0][..., -1]
            return dE_dt
        return torch.zeros_like(energy)

    @staticmethod
    def hamiltonian_constraint(
        predictions: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamiltonian constraint violation (for GR).

        H = 0 (simplified form)

        Parameters
        ----------
        predictions : torch.Tensor
            Metric components or field variables
        inputs : torch.Tensor
            Coordinates

        Returns
        -------
        torch.Tensor
            Hamiltonian constraint violation
        """
        # Placeholder: simplified Hamiltonian constraint
        return torch.zeros_like(predictions[..., 0])

    @staticmethod
    def momentum_constraint(
        predictions: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute momentum constraint violation (for GR).

        Parameters
        ----------
        predictions : torch.Tensor
            Field variables
        inputs : torch.Tensor
            Coordinates

        Returns
        -------
        torch.Tensor
            Momentum constraint violation
        """
        # Placeholder
        return torch.zeros_like(predictions[..., 0])

    @staticmethod
    def norm_preservation(
        predictions: torch.Tensor, inputs: torch.Tensor, target_norm: float = 1.0
    ) -> torch.Tensor:
        """Compute norm preservation violation (for quantum wavefunctions).

        Parameters
        ----------
        predictions : torch.Tensor
            Wavefunction (complex-valued)
        inputs : torch.Tensor
            Coordinates
        target_norm : float
            Target normalization (typically 1.0)

        Returns
        -------
        torch.Tensor
            Norm violation
        """
        if torch.is_complex(predictions):
            norm = torch.abs(predictions) ** 2
        else:
            # Assume real/imaginary concatenation
            if predictions.shape[-1] == 2:
                real = predictions[..., 0]
                imag = predictions[..., 1]
                norm = real**2 + imag**2
            else:
                norm = predictions**2

        total_norm = torch.sum(norm, dim=-1)
        return total_norm - target_norm


