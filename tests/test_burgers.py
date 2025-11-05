"""Tests for Burgers equation solver."""

import torch
import numpy as np
import pytest

from pde_solver.models.burgers import BurgersPINN, BurgersClassicalSolver
from pde_solver.training.multi_objective_loss import MultiObjectiveLoss


def test_burgers_pinn_forward():
    """Test Burgers PINN forward pass."""
    model = BurgersPINN(input_dim=2, output_dim=1, hidden_dims=[64, 64])
    coords = torch.randn(100, 2)
    output = model(coords)
    assert output.shape == (100, 1)


def test_burgers_pinn_residual():
    """Test Burgers PINN residual computation."""
    model = BurgersPINN(input_dim=2, output_dim=1, nu=0.01)
    coords = torch.randn(100, 2, requires_grad=True)
    residuals = model.compute_residual(coords)
    assert residuals.shape == (100, 1)


def test_burgers_classical_solver():
    """Test classical Burgers solver."""
    solver = BurgersClassicalSolver(nu=0.01, nx=64, nt=50)
    X, T, u = solver.solve()
    assert X.shape == (64, 50)
    assert T.shape == (64, 50)
    assert u.shape == (50, 64)


def test_multi_objective_loss():
    """Test multi-objective loss computation."""
    model = BurgersPINN(input_dim=2, output_dim=1, hidden_dims=[32, 32])
    loss_fn = MultiObjectiveLoss()

    data = {
        "coords": torch.randn(100, 2, requires_grad=True),
        "initial_coords": torch.randn(50, 2),
        "initial_values": torch.randn(50, 1),
        "boundary_coords": torch.randn(25, 2),
        "boundary_values": torch.randn(25, 1),
    }

    loss, components = loss_fn(model, data)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert "residual" in components
    assert "initial" in components
    assert "boundary" in components


def test_burgers_regression():
    """Regression test: train tiny model on manufactured solution."""
    # Simple test: check that model can overfit to initial condition
    model = BurgersPINN(input_dim=2, output_dim=1, hidden_dims=[32, 32], nu=0.01)
    loss_fn = MultiObjectiveLoss(weights={"initial": 1.0})

    # Initial condition points
    x_ic = torch.linspace(-1, 1, 50)
    t_ic = torch.zeros(50)
    initial_coords = torch.stack([x_ic, t_ic], dim=1)
    initial_values = -torch.sin(np.pi * x_ic).unsqueeze(1)

    data = {
        "initial_coords": initial_coords,
        "initial_values": initial_values,
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Train for a few steps
    for _ in range(100):
        optimizer.zero_grad()
        loss, _ = loss_fn(model, data)
        loss.backward()
        optimizer.step()

    # Check that loss decreased
    final_loss, _ = loss_fn(model, data)
    assert final_loss.item() < 0.1  # Should fit initial condition well

