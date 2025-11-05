"""Training infrastructure for physics-informed models."""

from pde_solver.training.physics_informed_trainer import PhysicsInformedTrainer
from pde_solver.training.multi_objective_loss import MultiObjectiveLoss

__all__ = [
    "PhysicsInformedTrainer",
    "MultiObjectiveLoss",
]

