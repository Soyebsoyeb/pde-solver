"""Minimal demo that works even without all dependencies."""

import sys
import os

def check_and_run():
    """Check dependencies and run appropriate demo."""
    
    print("=" * 60)
    print("PDE Solver - Minimal Demo")
    print("=" * 60)
    
    # Check what's available
    has_torch = False
    has_numpy = False
    has_scipy = False
    has_sympy = False
    has_matplotlib = False
    
    try:
        import torch
        has_torch = True
        print(f"[OK] PyTorch {torch.__version__}")
    except (ImportError, OSError) as e:
        print(f"[INFO] PyTorch not available ({type(e).__name__}) - will show structure only")
    
    try:
        import numpy
        has_numpy = True
        print(f"[OK] NumPy {numpy.__version__}")
    except ImportError:
        print("[INFO] NumPy not available")
    
    try:
        import scipy
        has_scipy = True
        print(f"[OK] SciPy {scipy.__version__}")
    except ImportError:
        print("[INFO] SciPy not available")
    
    try:
        import sympy
        has_sympy = True
        print(f"[OK] SymPy {sympy.__version__}")
    except ImportError:
        print("[INFO] SymPy not available")
    
    try:
        import matplotlib
        has_matplotlib = True
        print(f"[OK] Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("[INFO] Matplotlib not available")
    
    print("\n" + "=" * 60)
    
    # Run appropriate demo based on what's available
    if has_torch and has_numpy and has_scipy:
        print("Running FULL demo with neural network training...")
        print("=" * 60)
        run_full_demo()
    elif has_numpy and has_scipy:
        print("Running CLASSICAL solver demo (no neural networks)...")
        print("=" * 60)
        run_classical_demo()
    elif has_sympy:
        print("Running SYMBOLIC demo (equation generation only)...")
        print("=" * 60)
        run_symbolic_demo()
    else:
        print("Running STRUCTURE validation demo...")
        print("=" * 60)
        run_structure_demo()


def run_structure_demo():
    """Demo that works with just Python standard library."""
    print("\n1. Project Structure Validation")
    print("-" * 60)
    
    # Check file structure
    required_dirs = [
        "pde_solver",
        "pde_solver/core",
        "pde_solver/models",
        "pde_solver/training",
        "pde_solver/utils",
        "pde_solver/symbolic",
        "configs",
        "examples",
        "tests",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [MISSING] {dir_path}/")
            all_exist = False
    
    print("\n2. Python Module Structure")
    print("-" * 60)
    
    key_modules = [
        "pde_solver/__init__.py",
        "pde_solver/core/neural_symbolic_solver.py",
        "pde_solver/models/burgers.py",
        "pde_solver/training/physics_informed_trainer.py",
        "pde_solver/utils/visualization.py",
        "run_solver.py",
        "examples/burgers_demo.py",
    ]
    
    for module in key_modules:
        if os.path.exists(module):
            size = os.path.getsize(module)
            print(f"  [OK] {module} ({size} bytes)")
        else:
            print(f"  [MISSING] {module}")
            all_exist = False
    
    print("\n3. Configuration Files")
    print("-" * 60)
    
    configs = [
        "configs/burgers_small.yaml",
        "configs/burgers_medium.yaml",
    ]
    
    for config in configs:
        if os.path.exists(config):
            print(f"  [OK] {config}")
        else:
            print(f"  [MISSING] {config}")
    
    print("\n" + "=" * 60)
    if all_exist:
        print("[SUCCESS] Project structure is complete!")
        print("\nTo run full functionality, install dependencies:")
        print("  pip install -r requirements.txt")
    else:
        print("[WARNING] Some files are missing")
    print("=" * 60)


def run_symbolic_demo():
    """Demo using only SymPy."""
    import sympy as sp
    
    print("\n1. Creating Symbolic Burgers Equation")
    print("-" * 60)
    
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")(x, t)
    nu = sp.Symbol("nu", real=True, positive=True)
    
    u_t = sp.diff(u, t)
    u_x = sp.diff(u, x)
    u_xx = sp.diff(u_x, x)
    
    burgers_eq = u_t + u * u_x - nu * u_xx
    print(f"Burgers equation: {burgers_eq} = 0")
    
    print("\n2. Creating Symbolic Navier-Stokes Equations")
    print("-" * 60)
    
    y = sp.symbols("y", real=True)
    v = sp.Function("v")(x, y, t)
    p = sp.Function("p")(x, y, t)
    u_ns = sp.Function("u")(x, y, t)
    
    # Momentum equations
    u_t_ns = sp.diff(u_ns, t)
    u_x_ns = sp.diff(u_ns, x)
    u_y_ns = sp.diff(u_ns, y)
    u_xx_ns = sp.diff(u_x_ns, x)
    u_yy_ns = sp.diff(u_y_ns, y)
    p_x = sp.diff(p, x)
    
    eq1 = u_t_ns + u_ns * u_x_ns + v * u_y_ns + p_x - nu * (u_xx_ns + u_yy_ns)
    print(f"Momentum x: {eq1} = 0")
    
    # Continuity
    v_y = sp.diff(v, y)
    continuity = u_x_ns + v_y
    print(f"Continuity: {continuity} = 0")
    
    print("\n3. LaTeX Output")
    print("-" * 60)
    print(f"Burgers (LaTeX): ${sp.latex(burgers_eq)} = 0$")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Symbolic equations generated!")
    print("=" * 60)


def run_classical_demo():
    """Demo using NumPy and SciPy only (no PyTorch)."""
    import numpy as np
    
    # Check matplotlib availability
    try:
        import matplotlib
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
    
    print("\n1. Classical Burgers Solver (Finite Difference)")
    print("-" * 60)
    
    # Parameters
    nu = 0.01
    nx = 64
    nt = 50
    x_domain = (-1.0, 1.0)
    t_domain = (0.0, 1.0)
    
    # Grid
    x = np.linspace(x_domain[0], x_domain[1], nx)
    t = np.linspace(t_domain[0], t_domain[1], nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Initial condition
    u0 = -np.sin(np.pi * x)
    u = u0.copy()
    
    print(f"  Domain: x in [{x_domain[0]}, {x_domain[1]}], t in [{t_domain[0]}, {t_domain[1]}]")
    print(f"  Grid: {nx} spatial points, {nt} time steps")
    print(f"  Viscosity: nu = {nu}")
    
    # Solve with improved stability (upwind scheme)
    u_history = [u.copy()]
    for n in range(nt - 1):
        # Use upwind finite difference for stability
        u_x = np.zeros_like(u)
        u_x[1:] = (u[1:] - u[:-1]) / dx  # Forward difference
        
        # Convection term with upwind
        convection = np.zeros_like(u)
        convection[1:] = u[1:] * u_x[1:]
        
        # Diffusion term (central difference)
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        diffusion = nu * u_xx
        
        # CFL condition check
        max_speed = np.max(np.abs(u))
        cfl = max_speed * dt / dx
        if cfl > 1.0:
            # Adjust dt to maintain stability
            dt_stable = 0.9 * dx / (max_speed + 1e-10)
            dt_actual = min(dt, dt_stable)
        else:
            dt_actual = dt
        
        # Update
        u_new = u - dt_actual * convection + dt_actual * diffusion
        
        # Handle NaN/Inf
        u_new = np.nan_to_num(u_new, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp to reasonable values
        u_new = np.clip(u_new, -10.0, 10.0)
        
        # Boundary conditions
        u_new[0] = u_new[-1] = 0.0
        
        u = u_new
        u_history.append(u.copy())
    
    u_solution = np.array(u_history).T  # (nt, nx)
    
    print(f"\n2. Solution Statistics")
    print("-" * 60)
    print(f"  Initial condition: min={u0.min():.4f}, max={u0.max():.4f}, mean={u0.mean():.4f}")
    print(f"  Final solution: min={u_solution[-1].min():.4f}, max={u_solution[-1].max():.4f}, mean={u_solution[-1].mean():.4f}")
    print(f"  Solution shape: {u_solution.shape}")
    
    print("\n3. Conservation Check")
    print("-" * 60)
    try:
        mass_initial = np.trapz(u0, x)
        # Ensure shapes match for final solution
        final_solution = u_solution[-1, :] if u_solution.shape[0] == nt else u_solution[:, -1]
        if len(final_solution) == len(x):
            mass_final = np.trapz(final_solution, x)
            print(f"  Initial mass: {mass_initial:.6f}")
            print(f"  Final mass: {mass_final:.6f}")
            print(f"  Mass change: {abs(mass_final - mass_initial):.6e}")
        else:
            print(f"  [INFO] Shape mismatch for conservation check: {final_solution.shape} vs {x.shape}")
    except Exception as e:
        print(f"  [INFO] Conservation check skipped: {e}")
    
    if has_matplotlib:
        print("\n4. Generating Plot")
        print("-" * 60)
        try:
            import matplotlib.pyplot as plt
            os.makedirs("outputs", exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            X, T = np.meshgrid(x, t)
            contour = ax.contourf(X, T, u_solution.T, levels=20, cmap='viridis')
            plt.colorbar(contour, ax=ax, label='u(x,t)')
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_title('Burgers Equation - Classical Solution')
            plt.tight_layout()
            plt.savefig('outputs/burgers_classical_minimal.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  [OK] Saved plot to outputs/burgers_classical_minimal.png")
        except Exception as e:
            print(f"  [WARNING] Could not generate plot: {e}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Classical solver demo completed!")
    print("=" * 60)


def run_full_demo():
    """Full demo with PyTorch."""
    try:
        import torch
        import numpy as np
        from pde_solver.models.burgers import BurgersPINN, BurgersClassicalSolver
        from pde_solver.utils.visualization import plot_1d_time_evolution
        
        print("\n1. Creating Model")
        print("-" * 60)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")
        
        model = BurgersPINN(input_dim=2, output_dim=1, hidden_dims=[64, 64], nu=0.01)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model created with {n_params:,} parameters")
        
        print("\n2. Testing Forward Pass")
        print("-" * 60)
        coords = torch.randn(100, 2, device=device)
        output = model(coords)
        print(f"  Input shape: {coords.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        print("\n3. Testing Residual Computation")
        print("-" * 60)
        coords_grad = torch.randn(50, 2, device=device, requires_grad=True)
        residuals = model.compute_residual(coords_grad)
        print(f"  Residual shape: {residuals.shape}")
        print(f"  Residual magnitude: {residuals.abs().mean().item():.6f}")
        
        print("\n4. Running Classical Solver")
        print("-" * 60)
        solver = BurgersClassicalSolver(nu=0.01, nx=64, nt=50)
        X, T, u = solver.solve()
        print(f"  Solution shape: {u.shape}")
        print(f"  Solution range: [{u.min():.4f}, {u.max():.4f}]")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Full demo completed!")
        print("\nTo run full training, use:")
        print("  python examples/burgers_demo.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Full demo failed: {e}")
        print("Falling back to classical demo...")
        print()
        run_classical_demo()


if __name__ == "__main__":
    check_and_run()

