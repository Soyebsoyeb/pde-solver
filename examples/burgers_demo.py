"""Demo script for Burgers equation solver."""

import sys
import os

# Check dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available. Running classical solver only.")
    print("Install with: pip install torch")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[ERROR] NumPy is required. Install with: pip install numpy")
    sys.exit(1)

from pathlib import Path

if TORCH_AVAILABLE:
    from pde_solver.models.burgers import BurgersPINN, BurgersClassicalSolver
    from pde_solver.training.physics_informed_trainer import (
        PhysicsInformedTrainer,
        TrainingConfig,
    )
    from pde_solver.training.multi_objective_loss import MultiObjectiveLoss
    from pde_solver.utils.visualization import plot_1d_time_evolution, plot_loss_history
    from pde_solver.utils.result_analysis import evaluate_solution
else:
    # Fallback: use only classical solver
    try:
        from pde_solver.models.burgers import BurgersClassicalSolver
    except ImportError:
        print("[INFO] Running minimal classical solver demo instead...")
        BurgersClassicalSolver = None

# Create output directory
Path("outputs").mkdir(exist_ok=True)

if not TORCH_AVAILABLE:
    print("\n" + "=" * 60)
    print("Running Classical Solver Only (PyTorch not available)")
    print("=" * 60 + "\n")
    
    if BurgersClassicalSolver:
        # Run classical solver only
        solver = BurgersClassicalSolver(nu=0.01, nx=256, nt=100)
        X, T, u = solver.solve()
        
        # Try to plot if matplotlib available
        try:
            import matplotlib.pyplot as plt
            x = X[:, 0]
            t = T[0, :]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            time_indices = np.linspace(0, len(t) - 1, 10, dtype=int)
            colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
            
            for i, idx in enumerate(time_indices):
                ax.plot(x, u[idx, :], label=f"t={t[idx]:.3f}", color=colors[i])
            
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.set_title("Burgers Classical Solution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("outputs/burgers_classical_solution.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("Saved plot to outputs/burgers_classical_solution.png")
        except ImportError:
            print("Matplotlib not available - skipping plot")
        
        print("\nClassical solver completed!")
        print(f"Solution shape: {u.shape}")
        print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
        sys.exit(0)
    else:
        print("Cannot run without PyTorch. Install with: pip install torch")
        sys.exit(1)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Domain
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]
nu = 0.01

# Create model
model = BurgersPINN(
    input_dim=2,
    output_dim=1,
    hidden_dims=[128, 128, 128],
    nu=nu,
)
model = model.to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Generate training data
n_points = 5000
n_ic = 500
n_bc = 250

# Interior points
x_coords = torch.rand(n_points, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_coords = torch.rand(n_points, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
coords = torch.stack([x_coords, t_coords], dim=1).requires_grad_(True)

# Initial condition (t=0): u(x, 0) = -sin(pi*x)
x_ic = torch.rand(n_ic, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_ic = torch.zeros(n_ic, device=device)
initial_coords = torch.stack([x_ic, t_ic], dim=1)
initial_values = -torch.sin(np.pi * x_ic).unsqueeze(1)

# Boundary conditions: u(-1, t) = u(1, t) = 0
x_bc = torch.tensor([x_domain[0], x_domain[1]], device=device).repeat(n_bc // 2)
t_bc = torch.rand(n_bc, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
boundary_coords = torch.stack([x_bc, t_bc], dim=1)
boundary_values = torch.zeros(n_bc, 1, device=device)

train_data = {
    "coords": coords,
    "initial_coords": initial_coords,
    "initial_values": initial_values,
    "boundary_coords": boundary_coords,
    "boundary_values": boundary_values,
}

# Loss function
loss_fn = MultiObjectiveLoss(
    weights={
        "residual": 1.0,
        "boundary": 1.0,
        "initial": 1.0,
    }
)

# Training config
train_config = TrainingConfig(
    num_epochs=500,
    learning_rate=1e-3,
    batch_size=512,
    use_wandb=False,
    checkpoint_freq=50,
    log_freq=10,
)

# Trainer
trainer = PhysicsInformedTrainer(
    model=model,
    loss_fn=loss_fn,
    config=train_config,
    device=device,
)

# Train
print("Starting training...")
history = trainer.train(train_data)
print("Training completed!")

# Plot loss
plot_loss_history(history, save_path="outputs/burgers_training_loss.png")

# Evaluate on test grid
print("Evaluating model...")
x_test = torch.linspace(-1, 1, 256, device=device)
t_test = torch.linspace(0, 1, 100, device=device)
X, T = torch.meshgrid(x_test, t_test, indexing="ij")
coords_test = torch.stack([X.flatten(), T.flatten()], dim=1)

model.eval()
with torch.no_grad():
    u_pred = model(coords_test).cpu().numpy()

u_pred_grid = u_pred.reshape(256, 100)

# Compare with classical solver
print("Running classical solver for comparison...")
classical = BurgersClassicalSolver(nu=nu, nx=256, nt=100)
X_cl, T_cl, u_cl = classical.solve()

# Compute error
u_cl_torch = torch.from_numpy(u_cl.T).float()
u_pred_torch = torch.from_numpy(u_pred_grid).float()
metrics = evaluate_solution(u_pred_torch, u_cl_torch)

print("\nEvaluation metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.6e}")

# Plot solutions
plot_1d_time_evolution(
    x_test.cpu().numpy(),
    t_test.cpu().numpy(),
    u_pred_grid,
    title="Burgers PINN Solution",
    save_path="outputs/burgers_pinn_solution.png",
)

plot_1d_time_evolution(
    X_cl[:, 0],
    T_cl[0, :],
    u_cl.T,
    title="Burgers Classical Solution",
    save_path="outputs/burgers_classical_solution.png",
)

print("\nDemo completed! Check outputs/ directory for results.")

