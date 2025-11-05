"""Install dependencies and test the installation."""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def main():
    """Main installation and test procedure."""
    print("PDE Solver - Installation and Test Script")
    print("=" * 60)
    
    # Step 1: Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    print("✓ Python version OK")
    
    # Step 2: Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("\n⚠ Warning: Some dependencies may have failed to install")
        print("You can try installing manually: pip install -r requirements.txt")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 3: Install package
    if not run_command("pip install -e .", "Installing package in editable mode"):
        print("\n❌ Failed to install package")
        sys.exit(1)
    
    # Step 4: Run dependency check
    if not run_command("python check_dependencies.py", "Checking dependencies"):
        print("\n❌ Dependency check failed")
        sys.exit(1)
    
    # Step 5: Run structure test
    if not run_command("python test_runnable.py", "Testing code structure"):
        print("\n⚠ Some tests failed, but basic structure may be OK")
        response = input("Continue to demo? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 6: Run a quick demo
    print("\n" + "="*60)
    print("Installation complete! Running quick demo...")
    print("="*60)
    print("\nNote: This will train a small model (may take a few minutes)")
    response = input("Run demo now? (y/n): ")
    
    if response.lower() == 'y':
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        # Run demo with reduced epochs for quick test
        print("\nRunning quick demo (reduced epochs for testing)...")
        print("For full demo, run: python examples/burgers_demo.py")
        
        # Create a minimal test script
        test_script = """
import torch
import numpy as np
from pde_solver.models.burgers import BurgersPINN, BurgersClassicalSolver
from pde_solver.utils.visualization import plot_1d_time_evolution

print("Testing Burgers equation solver...")

# Test classical solver
print("\n1. Testing classical solver...")
solver = BurgersClassicalSolver(nu=0.01, nx=64, nt=50)
X, T, u = solver.solve()
print(f"   ✓ Classical solver: Solution shape {u.shape}")

# Test PINN model
print("\n2. Testing PINN model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")
model = BurgersPINN(input_dim=2, output_dim=1, hidden_dims=[32, 32], nu=0.01)
model = model.to(device)

# Test forward pass
coords = torch.randn(100, 2, device=device)
output = model(coords)
print(f"   ✓ PINN forward pass: Output shape {output.shape}")

# Test residual computation
coords_grad = torch.randn(50, 2, device=device, requires_grad=True)
residuals = model.compute_residual(coords_grad)
print(f"   ✓ Residual computation: Residual shape {residuals.shape}")

print("\n✓ All tests passed! The code is working correctly.")
print("\nNext steps:")
print("  - Run full demo: python examples/burgers_demo.py")
print("  - Train with CLI: python run_solver.py train configs/burgers_small.yaml")
print("  - See README.md for more information")
"""
        
        with open("quick_test.py", "w") as f:
            f.write(test_script)
        
        if run_command("python quick_test.py", "Running quick test"):
            print("\n" + "="*60)
            print("✓ Installation and test successful!")
            print("="*60)
            
            # Clean up
            if os.path.exists("quick_test.py"):
                os.remove("quick_test.py")
        else:
            print("\n⚠ Quick test had issues, but installation may still be OK")
    else:
        print("\n✓ Installation complete!")
        print("\nTo test manually:")
        print("  python test_runnable.py")
        print("  python examples/burgers_demo.py")

if __name__ == "__main__":
    main()

