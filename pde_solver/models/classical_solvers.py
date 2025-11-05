"""Classical numerical solvers for PDEs (no PyTorch dependencies)."""

from typing import Optional, Tuple
import numpy as np


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
