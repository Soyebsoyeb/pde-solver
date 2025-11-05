"""Test that the code structure is correct and can be imported (if dependencies installed)."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that modules can be imported (if dependencies available)."""
    errors = []
    
    # Check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
        print("[WARNING] PyTorch not installed - skipping torch-dependent imports")
    
    if torch_available:
        try:
            from pde_solver.core import NeuralSymbolicSolver, PhysicsConstraints
            print("[OK] Core modules imported")
        except Exception as e:
            errors.append(f"Core modules: {e}")
        
        try:
            from pde_solver.models import BurgersPINN, DeepONet
            print("[OK] Model modules imported")
        except Exception as e:
            errors.append(f"Model modules: {e}")
        
        try:
            from pde_solver.training import PhysicsInformedTrainer, MultiObjectiveLoss
            print("[OK] Training modules imported")
        except Exception as e:
            errors.append(f"Training modules: {e}")
        
        try:
            from pde_solver.utils import visualize_solution, evaluate_solution
            print("[OK] Utility modules imported")
        except Exception as e:
            errors.append(f"Utility modules: {e}")
        
        # Test basic functionality
        try:
            from pde_solver.models.burgers import BurgersPINN
            model = BurgersPINN(input_dim=2, output_dim=1, hidden_dims=[32, 32])
            coords = torch.randn(10, 2)
            output = model(coords)
            assert output.shape == (10, 1), f"Expected shape (10, 1), got {output.shape}"
            print("[OK] Burgers PINN forward pass works")
        except Exception as e:
            errors.append(f"Burgers PINN test: {e}")
        
        try:
            from pde_solver.models.burgers import BurgersClassicalSolver
            solver = BurgersClassicalSolver(nu=0.01, nx=64, nt=50)
            X, T, u = solver.solve()
            assert u.shape == (50, 64), f"Expected shape (50, 64), got {u.shape}"
            print("[OK] Classical Burgers solver works")
        except Exception as e:
            errors.append(f"Classical solver test: {e}")
    
    # Test non-torch modules (only if torch is available, since CLI imports torch)
    if torch_available:
        try:
            from pde_solver.symbolic.symbolic_engine import SymbolicEngine
            engine = SymbolicEngine()
            eq = engine.create_burgers_equation(nu=0.01)
            print("[OK] Symbolic engine works")
        except Exception as e:
            errors.append(f"Symbolic engine: {e}")
        
        # Check CLI structure (requires torch)
        try:
            import pde_solver.cli
            print("[OK] CLI module exists")
        except Exception as e:
            errors.append(f"CLI module: {e}")
    else:
        # Test symbolic engine without torch (should work)
        try:
            import sympy
            from pde_solver.symbolic.symbolic_engine import SymbolicEngine
            engine = SymbolicEngine()
            eq = engine.create_burgers_equation(nu=0.01)
            print("[OK] Symbolic engine works (torch not needed)")
        except ImportError:
            errors.append("SymPy not installed (needed for symbolic engine)")
        except Exception as e:
            errors.append(f"Symbolic engine: {e}")
    
    if errors:
        print("\n[ERROR] Errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\n[SUCCESS] All tests passed!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n[TIP] Install dependencies with: pip install -r requirements.txt")
    sys.exit(0 if success else 1)

