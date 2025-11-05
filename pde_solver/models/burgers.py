"""Burgers' equation solvers (classical + PINN)."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint


class FourierFeatures(nn.Module):
    """Fourier feature encoding for positional inputs."""

    def __init__(self, input_dim: int, num_features: int = 128):
        """Initialize Fourier features.

        Parameters
        ----------
        input_dim : int
            Input dimension (e.g., 2 for (x, t))
        num_features : int
            Number of Fourier features
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features

        # Random frequencies for Fourier encoding
        B = torch.randn(input_dim, num_features // 2) * 2 * np.pi
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, input_dim)

        Returns
        -------
        torch.Tensor
            Encoded features of shape (N, 2 * num_features)
        """
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLP(nn.Module):
    """Multi-layer perceptron with residual connections."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [256, 256, 256, 256],
        activation: str = "swish",
        use_layer_norm: bool = True,
    ):
        """Initialize MLP.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        hidden_dims : list
            Hidden layer dimensions
        activation : str
            Activation function name
        use_layer_norm : bool
            Whether to use layer normalization
        """
        super().__init__()
        self.layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims + [output_dim]

        # Activation function
        if activation == "swish":
            act_fn = nn.SiLU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.SiLU()

        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(linear)

            if i < len(dims) - 2:  # Not the last layer
                if use_layer_norm:
                    self.layers.append(nn.LayerNorm(dims[i + 1]))
                self.layers.append(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x


class BurgersPINN(nn.Module):
    """Physics-Informed Neural Network for Burgers' equation.

    Solves: u_t + u * u_x - nu * u_xx = 0
    """

    def __init__(
        self,
        input_dim: int = 2,  # (x, t)
        output_dim: int = 1,  # u(x, t)
        hidden_dims: list = [256, 256, 256, 256],
        num_fourier_features: int = 128,
        nu: float = 0.01,  # Viscosity
    ):
        """Initialize Burgers PINN.

        Parameters
        ----------
        input_dim : int
            Input dimension (2 for (x, t))
        output_dim : int
            Output dimension (1 for u)
        hidden_dims : list
            Hidden layer dimensions
        num_fourier_features : int
            Number of Fourier features
        nu : float
            Viscosity coefficient
        """
        super().__init__()
        self.nu = nu

        # Fourier feature encoding
        self.fourier_features = FourierFeatures(input_dim, num_fourier_features)

        # MLP network
        # Fourier features outputs num_fourier_features (not 2x due to // 2 in init)
        mlp_input_dim = num_fourier_features + input_dim  # Fourier + original
        self.mlp = MLP(mlp_input_dim, output_dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 2) where columns are [x, t]

        Returns
        -------
        torch.Tensor
            Solution u(x, t) of shape (N, 1)
        """
        # Fourier encoding
        x_fourier = self.fourier_features(x)

        # Concatenate original and Fourier features
        x_combined = torch.cat([x, x_fourier], dim=-1)

        # MLP
        u = self.mlp(x_combined)
        return u

    def compute_residual(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute PDE residual.

        Residual: u_t + u * u_x - nu * u_xx

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates (N, 2) [x, t]
        u : torch.Tensor, optional
            Pre-computed solution (if None, compute via forward)

        Returns
        -------
        torch.Tensor
            Residual values (N, 1)
        """
        if u is None:
            u = self.forward(x)

        x_coords = x[:, 0:1].requires_grad_(True)
        t_coords = x[:, 1:2].requires_grad_(True)
        x_full = torch.cat([x_coords, t_coords], dim=1)

        u = self.forward(x_full)

        # Compute gradients
        u_t = torch.autograd.grad(
            u.sum(), t_coords, create_graph=True, retain_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u.sum(), x_coords, create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x.sum(), x_coords, create_graph=True, retain_graph=True
        )[0]

        # Burgers equation: u_t + u * u_x - nu * u_xx
        residual = u_t + u * u_x - self.nu * u_xx
        return residual


class BurgersClassicalSolver:
    """Classical numerical solver for Burgers' equation using method of characteristics + diffusion."""

    def __init__(self, nu: float = 0.01, nx: int = 256, nt: int = 100):
        """Initialize classical solver.

        Parameters
        ----------
        nu : float
            Viscosity coefficient
        nx : int
            Number of spatial points
        nt : int
            Number of time steps
        """
        self.nu = nu
        self.nx = nx
        self.nt = nt

    def solve(
        self,
        x_domain: Tuple[float, float] = (-1.0, 1.0),
        t_domain: Tuple[float, float] = (0.0, 1.0),
        u0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve Burgers' equation.

        Parameters
        ----------
        x_domain : Tuple[float, float]
            Spatial domain [x_min, x_max]
        t_domain : Tuple[float, float]
            Time domain [t_min, t_max]
        u0 : np.ndarray, optional
            Initial condition (if None, use default)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (x_grid, t_grid, u_solution) where u_solution is (nt, nx)
        """
        # Spatial grid
        x = np.linspace(x_domain[0], x_domain[1], self.nx)
        dx = x[1] - x[0]

        # Time grid
        t = np.linspace(t_domain[0], t_domain[1], self.nt)
        dt = t[1] - t[0]

        # Initial condition
        if u0 is None:
            u0 = -np.sin(np.pi * x)

        u = u0.copy()

        # Solve using finite difference method (viscous Burgers)
        # u_t + u * u_x = nu * u_xx
        u_history = [u.copy()]

        for n in range(self.nt - 1):
            # Convection term: u * u_x (upwind)
            u_x = np.gradient(u, dx)
            convection = u * u_x

            # Diffusion term: nu * u_xx
            u_xx = np.gradient(u_x, dx)
            diffusion = self.nu * u_xx

            # Update: u_t = -convection + diffusion
            u_new = u - dt * convection + dt * diffusion

            # Boundary conditions (periodic or zero)
            u_new[0] = u_new[-1] = 0.0

            u = u_new
            u_history.append(u.copy())

        u_solution = np.array(u_history)

        # Create meshgrids
        X, T = np.meshgrid(x, t, indexing="ij")

        return X, T, u_solution.T  # Return as (nt, nx)

