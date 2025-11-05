"""Adaptive meshing and remeshing strategies."""

from typing import Callable, Tuple, Optional
import torch
import numpy as np


class AdaptiveMesh:
    """Adaptive mesh refinement based on error indicators."""

    def __init__(
        self,
        initial_coords: torch.Tensor,
        error_indicator: Optional[Callable] = None,
        refinement_ratio: float = 0.2,
        max_refinements: int = 5,
    ):
        """Initialize adaptive mesh.

        Parameters
        ----------
        initial_coords : torch.Tensor
            Initial coordinate points
        error_indicator : Callable, optional
            Function computing error indicator per point
        refinement_ratio : float
            Fraction of points to refine (0-1)
        max_refinements : int
            Maximum number of refinement iterations
        """
        self.coords = initial_coords.clone()
        self.error_indicator = error_indicator or self._default_error_indicator
        self.refinement_ratio = refinement_ratio
        self.max_refinements = max_refinements
        self.refinement_count = 0

    def _default_error_indicator(
        self, residuals: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Default error indicator: absolute residual value.

        Parameters
        ----------
        residuals : torch.Tensor
            Residual values
        coords : torch.Tensor
            Coordinates

        Returns
        -------
        torch.Tensor
            Error indicator per point
        """
        return torch.abs(residuals)

    def refine(
        self, residuals: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        """Refine mesh based on error indicator.

        Parameters
        ----------
        residuals : torch.Tensor
            PDE residuals at current points
        model : torch.nn.Module
            Current model for generating new points

        Returns
        -------
        torch.Tensor
            Refined coordinates
        """
        if self.refinement_count >= self.max_refinements:
            return self.coords

        # Compute error indicators
        error = self.error_indicator(residuals, self.coords)

        # Select points to refine (highest error)
        num_refine = int(self.refinement_ratio * len(self.coords))
        _, indices = torch.topk(error, num_refine, largest=True)

        # Generate new points near high-error regions
        new_points = self._generate_refined_points(indices)

        # Combine with existing points
        self.coords = torch.cat([self.coords, new_points], dim=0)
        self.refinement_count += 1

        return self.coords

    def _generate_refined_points(self, indices: torch.Tensor) -> torch.Tensor:
        """Generate new points near high-error regions.

        Parameters
        ----------
        indices : torch.Tensor
            Indices of points to refine around

        Returns
        -------
        torch.Tensor
            New coordinate points
        """
        selected_coords = self.coords[indices]
        dim = selected_coords.shape[-1]

        # Add small perturbations around selected points
        noise_scale = 0.1 * torch.std(self.coords, dim=0)
        noise = torch.randn_like(selected_coords) * noise_scale.unsqueeze(0)

        # Generate 2-3 points per selected point
        num_new_per_point = 2
        new_points_list = []

        for _ in range(num_new_per_point):
            perturbed = selected_coords + noise * torch.randn(
                len(selected_coords), 1, device=selected_coords.device
            )
            new_points_list.append(perturbed)

        new_points = torch.cat(new_points_list, dim=0)

        # Ensure points stay within domain bounds
        coord_min = self.coords.min(dim=0)[0]
        coord_max = self.coords.max(dim=0)[0]
        new_points = torch.clamp(new_points, coord_min, coord_max)

        return new_points

    def get_coords(self) -> torch.Tensor:
        """Get current coordinates.

        Returns
        -------
        torch.Tensor
            Current coordinate mesh
        """
        return self.coords

    def reset(self, new_coords: torch.Tensor) -> None:
        """Reset mesh with new coordinates.

        Parameters
        ----------
        new_coords : torch.Tensor
            New coordinate points
        """
        self.coords = new_coords.clone()
        self.refinement_count = 0


