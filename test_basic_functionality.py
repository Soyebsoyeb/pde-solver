"""Test basic functionality of the PDE solver."""

def test_package_import():
    """Test that package imports without errors."""
    import pde_solver
    print(f"[PASS] Package imported successfully (version {pde_solver.__version__})")

def test_classical_solver():
    """Test classical solver without torch dependencies."""
    from pde_solver.models.classical_solvers import BurgersClassicalSolver
    import numpy as np
    
    solver = BurgersClassicalSolver(nu=0.01, nx=50, nt=20)
    X, T, u = solver.solve()
    
    # Solver returns (nx, nt) grids and (nt, nx) solution
    assert X.shape == (50, 20), f"Expected shape (50, 20), got {X.shape}"
    assert T.shape == (50, 20), f"Expected shape (50, 20), got {T.shape}"
    assert u.shape == (50, 20), f"Expected shape (50, 20), got {u.shape}"
    
    print("[PASS] Classical solver works correctly")

def test_visualization():
    """Test visualization utilities."""
    from pde_solver.utils.visualization import plot_1d_time_evolution
    import numpy as np
    import os
    
    x = np.linspace(-1, 1, 50)
    t = np.linspace(0, 1, 10)
    u = np.sin(np.pi * x[None, :]) * np.exp(-t[:, None])
    
    output_path = "outputs/test_plot.png"
    plot_1d_time_evolution(x, t, u, title="Test Plot", save_path=output_path)
    
    assert os.path.exists(output_path), f"Plot not saved to {output_path}"
    print(f"[PASS] Visualization works correctly, saved to {output_path}")

def test_cli_help():
    """Test CLI help command."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pde_solver.cli", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "CLI help command failed"
    assert "train" in result.stdout.lower(), "Train command not found in help"
    assert "eval" in result.stdout.lower(), "Eval command not found in help"
    assert "classical" in result.stdout.lower(), "Classical command not found in help"
    print("[PASS] CLI help command works")

if __name__ == "__main__":
    print("Testing PDE Solver Basic Functionality\n" + "=" * 50)
    
    try:
        test_package_import()
        test_classical_solver()
        test_visualization()
        test_cli_help()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] All tests passed! The code is fully runnable.")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
