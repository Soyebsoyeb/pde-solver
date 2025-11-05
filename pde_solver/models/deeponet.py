"""DeepONet implementation for operator learning."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from pde_solver.models.burgers import MLP


class DeepONet(nn.Module):
    """Deep Operator Network (DeepONet).

    Learns to map input functions to solution operators.
    Branch net: encodes input function (e.g., initial condition)
    Trunk net: encodes evaluation coordinates
    """

    def __init__(
        self,
        branch_input_dim: int,  # e.g., number of sensor points for IC
        trunk_input_dim: int,  # e.g., spatial coordinates (x, t)
        branch_hidden_dims: list = [256, 256, 256],
        trunk_hidden_dims: list = [256, 256, 256],
        output_dim: int = 1,
        num_basis: int = 128,  # Number of basis functions
    ):
        """Initialize DeepONet.

        Parameters
        ----------
        branch_input_dim : int
            Input dimension for branch net (e.g., IC sampled at sensor points)
        trunk_input_dim : int
            Input dimension for trunk net (e.g., (x, t))
        branch_hidden_dims : list
            Branch network hidden dimensions
        trunk_hidden_dims : list
            Trunk network hidden dimensions
        output_dim : int
            Output dimension
        num_basis : int
            Number of basis functions (output dimension of branch/trunk)
        """
        super().__init__()
        self.num_basis = num_basis
        self.output_dim = output_dim

        # Branch network: encodes input function
        self.branch = MLP(
            branch_input_dim,
            num_basis * output_dim,
            branch_hidden_dims,
        )

        # Trunk network: encodes evaluation coordinates
        self.trunk = MLP(
            trunk_input_dim,
            num_basis * output_dim,
            trunk_hidden_dims,
        )

    def forward(
        self, branch_input: torch.Tensor, trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        branch_input : torch.Tensor
            Input function encoding (N, branch_input_dim)
        trunk_input : torch.Tensor
            Evaluation coordinates (N, trunk_input_dim)

        Returns
        -------
        torch.Tensor
            Operator output (N, output_dim)
        """
        # Branch network output
        branch_out = self.branch(branch_input)  # (N, num_basis * output_dim)
        branch_out = branch_out.reshape(-1, self.num_basis, self.output_dim)

        # Trunk network output
        trunk_out = self.trunk(trunk_input)  # (N, num_basis * output_dim)
        trunk_out = trunk_out.reshape(-1, self.num_basis, self.output_dim)

        # Dot product: sum over basis functions
        output = torch.sum(branch_out * trunk_out, dim=1)  # (N, output_dim)

        return output

    def compute_gradient_penalty(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """Compute gradient penalty for operator outputs (WGAN-style regularization).

        Parameters
        ----------
        branch_input : torch.Tensor
            Branch input
        trunk_input : torch.Tensor
            Trunk input
        lambda_gp : float
            Gradient penalty weight

        Returns
        -------
        torch.Tensor
            Gradient penalty loss
        """
        trunk_input.requires_grad_(True)
        output = self.forward(branch_input, trunk_input)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=trunk_input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Gradient penalty: (||grad||_2 - 1)^2
        gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

