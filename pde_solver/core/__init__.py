"""Core modules for PDE solving infrastructure."""

from pde_solver.core.neural_symbolic_solver import NeuralSymbolicSolver
from pde_solver.core.physics_constraints import PhysicsConstraints
from pde_solver.core.adaptive_mesh import AdaptiveMesh

__all__ = [
    "NeuralSymbolicSolver",
    "PhysicsConstraints",
    "AdaptiveMesh",
]


