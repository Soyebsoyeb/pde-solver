"""Validate project structure and imports."""

import sys
from pathlib import Path

def check_imports():
    """Check if all modules can be imported."""
    errors = []
    
    try:
        import pde_solver
        print("✓ pde_solver package imported")
    except Exception as e:
        errors.append(f"Failed to import pde_solver: {e}")
    
    try:
        from pde_solver.core import NeuralSymbolicSolver, PhysicsConstraints
        print("✓ Core modules imported")
    except Exception as e:
        errors.append(f"Failed to import core modules: {e}")
    
    try:
        from pde_solver.models import BurgersPINN, DeepONet
        print("✓ Model modules imported")
    except Exception as e:
        errors.append(f"Failed to import model modules: {e}")
    
    try:
        from pde_solver.training import PhysicsInformedTrainer, MultiObjectiveLoss
        print("✓ Training modules imported")
    except Exception as e:
        errors.append(f"Failed to import training modules: {e}")
    
    try:
        from pde_solver.utils import visualize_solution, evaluate_solution
        print("✓ Utility modules imported")
    except Exception as e:
        errors.append(f"Failed to import utility modules: {e}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    
    print("\n✓ All imports successful!")
    return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)

