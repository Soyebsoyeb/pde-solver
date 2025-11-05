"""Tests for core modules."""

import torch
import pytest

from pde_solver.core.neural_symbolic_solver import NeuralSymbolicSolver, SolverConfig
from pde_solver.core.physics_constraints import PhysicsConstraints
from pde_solver.models.burgers import BurgersPINN


def test_solver_config():
    """Test solver configuration."""
    config = SolverConfig(device="cpu", seed=42, deterministic=True)
    assert config.device == "cpu"
    assert config.seed == 42


def test_neural_symbolic_solver():
    """Test neural-symbolic solver."""
    model = BurgersPINN(input_dim=2, output_dim=1, hidden_dims=[32, 32])
    config = SolverConfig(device="cpu")
    solver = NeuralSymbolicSolver(model, config)

    coords = torch.randn(10, 2)
    output = solver.infer(coords)
    assert output.shape == (10, 1)


def test_physics_constraints():
    """Test physics constraints."""
    constraints = PhysicsConstraints()

    # Register a simple constraint
    def test_constraint(predictions, inputs):
        return predictions - 1.0

    constraints.register_constraint("test", test_constraint, weight=1.0)

    predictions = torch.ones(10, 1)
    inputs = torch.randn(10, 2)
    violations = constraints.compute_violations(predictions, inputs)
    assert "test" in violations

    loss = constraints.compute_loss(predictions, inputs)
    assert loss.item() == 0.0  # predictions = 1.0, so violation = 0


def test_mass_conservation():
    """Test mass conservation constraint."""
    constraints = PhysicsConstraints()
    predictions = torch.randn(10, 2)  # (u, v) velocity field
    inputs = torch.randn(10, 2).requires_grad_(True)  # (x, y) coordinates

    # This is a simplified test - actual divergence requires proper gradients
    violation = PhysicsConstraints.mass_conservation(predictions, inputs)
    assert violation.shape == (10,)

