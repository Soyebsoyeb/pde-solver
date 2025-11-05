"""Adaptive sampling strategies for physics-informed training."""

from typing import Optional, Callable
import torch
import numpy as np


class AdaptiveSampler:
    """Adaptive sampling based on residual-driven refinement."""

    def __init__(
        self,
        initial_coords: torch.Tensor,
        domain_bounds: tuple,
        num_samples: int = 1000,
        refinement_ratio: float = 0.2,
    ):
        """Initialize adaptive sampler.

        Parameters
        ----------
        initial_coords : torch.Tensor
            Initial coordinate points
        domain_bounds : tuple
            Domain bounds for each dimension [(min, max), ...]
        num_samples : int
            Number of samples to maintain
        refinement_ratio : float
            Fraction of points to replace per refinement
        """
        self.coords = initial_coords.clone()
        self.domain_bounds = domain_bounds
        self.num_samples = num_samples
        self.refinement_ratio = refinement_ratio

    def refine(
        self,
        model: torch.nn.Module,
        residual_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Refine sampling based on residual map.

        Parameters
        ----------
        model : torch.nn.Module
            Current model
        residual_fn : Callable, optional
            Function to compute residuals (if None, use model.compute_residual)

        Returns
        -------
        torch.Tensor
            Refined coordinates
        """
        # Compute residuals at current points
        with torch.no_grad():
            if residual_fn:
                residuals = residual_fn(self.coords, model)
            elif hasattr(model, "compute_residual"):
                residuals = model.compute_residual(self.coords)
            else:
                # Fallback: use prediction variance
                predictions = model(self.coords)
                residuals = torch.var(predictions, dim=-1, keepdim=True)

        # Select points with highest residuals
        residual_magnitude = torch.abs(residuals).squeeze()
        num_remove = int(self.refinement_ratio * len(self.coords))
        _, remove_indices = torch.topk(residual_magnitude, num_remove, largest=True)

        # Remove high-residual points
        keep_mask = torch.ones(len(self.coords), dtype=torch.bool, device=self.coords.device)
        keep_mask[remove_indices] = False
        self.coords = self.coords[keep_mask]

        # Generate new points near high-residual regions
        new_coords = self._generate_near_high_residual(
            remove_indices, num_remove
        )

        # Combine
        self.coords = torch.cat([self.coords, new_coords], dim=0)

        # Trim to target size if needed
        if len(self.coords) > self.num_samples:
            # Keep points with highest residuals
            with torch.no_grad():
                if hasattr(model, "compute_residual"):
                    residuals = model.compute_residual(self.coords)
                else:
                    residuals = torch.var(model(self.coords), dim=-1, keepdim=True)
            residual_magnitude = torch.abs(residuals).squeeze()
            _, keep_indices = torch.topk(
                residual_magnitude, self.num_samples, largest=True
            )
            self.coords = self.coords[keep_indices]

        return self.coords

    def _generate_near_high_residual(
        self, indices: torch.Tensor, num_new: int
    ) -> torch.Tensor:
        """Generate new points near high-residual regions.

        Parameters
        ----------
        indices : torch.Tensor
            Indices of high-residual points
        num_new : int
            Number of new points to generate

        Returns
        -------
        torch.Tensor
            New coordinate points
        """
        high_residual_coords = self.coords[indices]
        dim = high_residual_coords.shape[-1]

        # Generate points near high-residual regions
        noise_scale = 0.1
        noise = torch.randn(num_new, dim, device=self.coords.device) * noise_scale

        # Sample from neighborhood of high-residual points
        sample_indices = torch.randint(
            0, len(high_residual_coords), (num_new,), device=self.coords.device
        )
        new_coords = high_residual_coords[sample_indices] + noise

        # Clip to domain bounds
        for i, (min_val, max_val) in enumerate(self.domain_bounds):
            new_coords[:, i] = torch.clamp(new_coords[:, i], min_val, max_val)

        return new_coords

    def get_coords(self) -> torch.Tensor:
        """Get current coordinates.

        Returns
        -------
        torch.Tensor
            Current coordinate points
        """
        return self.coords

