"""PDE Solver: Research-grade hybrid physics + AI PDE engine.

This package provides solvers for complex PDEs including:
- Navier-Stokes equations (2D incompressible)
- Einstein field equations (GR)
- Schrödinger/Dirac equations
- Burgers' equation

Combines classical numerical methods (FEM/FVM/spectral) with neural-symbolic
methods (PINNs, DeepONet, operator learning).
"""

__version__ = "0.1.0"

# Lazy imports to avoid dependency issues at startup
def _lazy_import_neural_symbolic_solver():
    """Lazy import for NeuralSymbolicSolver."""
    try:
        from pde_solver.core.neural_symbolic_solver import NeuralSymbolicSolver
        return NeuralSymbolicSolver
    except ImportError as e:
        raise ImportError(
            f"NeuralSymbolicSolver requires PyTorch. Error: {e}. "
            "Install with: pip install torch"
        )

def _lazy_import_physics_constraints():
    """Lazy import for PhysicsConstraints."""
    try:
        from pde_solver.core.physics_constraints import PhysicsConstraints
        return PhysicsConstraints
    except ImportError:
        # PhysicsConstraints doesn't strictly require torch, but it's better with it
        from pde_solver.core.physics_constraints import PhysicsConstraints
        return PhysicsConstraints

# Make imports available but lazy
def __getattr__(name):
    """Lazy attribute loading."""
    if name == "NeuralSymbolicSolver":
        return _lazy_import_neural_symbolic_solver()
    elif name == "PhysicsConstraints":
        return _lazy_import_physics_constraints()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "NeuralSymbolicSolver",
    "PhysicsConstraints",
    "__version__",
]


