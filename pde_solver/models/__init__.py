"""Neural and classical PDE solver models."""

# Lazy imports to avoid torch DLL issues on Windows
def __getattr__(name):
    """Lazy attribute loading."""
    if name == "BurgersPINN":
        from pde_solver.models.burgers import BurgersPINN
        return BurgersPINN
    elif name == "BurgersClassicalSolver":
        from pde_solver.models.classical_solvers import BurgersClassicalSolver
        return BurgersClassicalSolver
    elif name == "DeepONet":
        from pde_solver.models.deeponet import DeepONet
        return DeepONet
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "BurgersPINN",
    "BurgersClassicalSolver",
    "DeepONet",
]

