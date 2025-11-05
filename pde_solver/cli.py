"""Command-line interface for PDE solver."""

import typer
from pathlib import Path
import yaml
from typing import Optional

# Heavy deps are imported lazily inside commands to avoid Windows DLL issues
# and to make `--help` fast and robust.

app = typer.Typer()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config file


    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


@app.command()
def train(
    config_path: str = typer.Argument(..., help="Path to config file"),
    device: str = typer.Option("cpu", help="Device (cpu/cuda)"),
):
    """Train a PDE solver model."""
    import torch
    import numpy as np
    from pde_solver.models.burgers import BurgersPINN
    from pde_solver.training.physics_informed_trainer import (
        PhysicsInformedTrainer,
        TrainingConfig,
    )
    from pde_solver.training.multi_objective_loss import MultiObjectiveLoss

    config = load_config(config_path)
    pde_type = config.get("pde_type", "burgers")

    print(f"Training {pde_type} solver...")

    if pde_type == "burgers":
        # Create model
        model = BurgersPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=config.get("hidden_dims", [256, 256, 256, 256]),
            nu=config.get("nu", 0.01),
        )
        model = model.to(device)

        # Generate training data
        x_domain = config.get("x_domain", [-1.0, 1.0])
        t_domain = config.get("t_domain", [0.0, 1.0])
        n_points = config.get("n_points", 10000)

        # Interior points
        x_coords = torch.rand(n_points, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
        t_coords = torch.rand(n_points, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
        coords = torch.stack([x_coords, t_coords], dim=1).requires_grad_(True)

        # Initial condition (t=0)
        n_ic = config.get("n_ic", 1000)
        x_ic = torch.rand(n_ic, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
        t_ic = torch.zeros(n_ic, device=device)
        initial_coords = torch.stack([x_ic, t_ic], dim=1)
        initial_values = -torch.sin(np.pi * x_ic).unsqueeze(1)

        # Boundary conditions
        n_bc = config.get("n_bc", 500)
        n_bc_per_side = n_bc // 2
        x_bc_left = torch.full((n_bc_per_side,), x_domain[0], device=device)
        x_bc_right = torch.full((n_bc - n_bc_per_side,), x_domain[1], device=device)
        x_bc = torch.cat([x_bc_left, x_bc_right])
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
                "residual": config.get("weight_residual", 1.0),
                "boundary": config.get("weight_boundary", 1.0),
                "initial": config.get("weight_initial", 1.0),
            }
        )

        # Training config
        train_config = TrainingConfig(
            num_epochs=config.get("num_epochs", 1000),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            batch_size=config.get("batch_size", 1024),
            use_wandb=config.get("use_wandb", False),
            checkpoint_freq=config.get("checkpoint_freq", 100),
        )

        # Trainer
        trainer = PhysicsInformedTrainer(
            model=model,
            loss_fn=loss_fn,
            config=train_config,
            device=device,
        )

        # Train
        history = trainer.train(train_data)

        # Save final checkpoint
        trainer._save_checkpoint(train_config.num_epochs - 1)

        # Plot loss
        from pde_solver.utils.visualization import plot_loss_history
        plot_loss_history(history, save_path="outputs/training_loss.png")

        print("Training completed!")
    else:
        print(f"PDE type {pde_type} not yet implemented")


@app.command()
def eval(
    config_path: str = typer.Argument(..., help="Path to config file"),
    checkpoint_path: str = typer.Argument(..., help="Path to checkpoint"),
    device: str = typer.Option("cpu", help="Device (cpu/cuda)"),
):
    """Evaluate a trained model."""
    import torch
    import numpy as np
    from pde_solver.models.burgers import BurgersPINN
    from pde_solver.models.classical_solvers import BurgersClassicalSolver

    config = load_config(config_path)
    pde_type = config.get("pde_type", "burgers")

    print(f"Evaluating {pde_type} solver...")

    if pde_type == "burgers":
        # Load model
        model = BurgersPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=config.get("hidden_dims", [256, 256, 256, 256]),
            nu=config.get("nu", 0.01),
        )
        model = model.to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Generate test points
        x_test = torch.linspace(-1, 1, 256, device=device)
        t_test = torch.linspace(0, 1, 100, device=device)
        X, T = torch.meshgrid(x_test, t_test, indexing="ij")
        coords_test = torch.stack([X.flatten(), T.flatten()], dim=1)

        # Evaluate
        model.eval()
        with torch.no_grad():
            u_pred = model(coords_test).cpu().numpy()

        # Compare with classical solver
        classical = BurgersClassicalSolver(nu=config.get("nu", 0.01))
        X_cl, T_cl, u_cl = classical.solve()

        # Reshape for comparison
        u_pred_grid = u_pred.reshape(256, 100)

        # Compute error
        from pde_solver.utils.result_analysis import evaluate_solution
        u_cl_torch = torch.from_numpy(u_cl.T).float()
        u_pred_torch = torch.from_numpy(u_pred_grid).float()
        metrics = evaluate_solution(u_pred_torch, u_cl_torch)

        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6e}")

        # Plot
        from pde_solver.utils.visualization import plot_1d_time_evolution
        plot_1d_time_evolution(
            x_test.cpu().numpy(),
            t_test.cpu().numpy(),
            u_pred_grid,
            title="Burgers PINN Solution",
            save_path="outputs/burgers_pinn_solution.png",
        )

        print("Evaluation completed!")
    else:
        print(f"PDE type {pde_type} not yet implemented")


@app.command()
def classical(
    config_path: str = typer.Argument(..., help="Path to config file"),
):
    """Run classical numerical solver."""
    from pde_solver.models.classical_solvers import BurgersClassicalSolver

    config = load_config(config_path)
    pde_type = config.get("pde_type", "burgers")

    print(f"Running classical {pde_type} solver...")

    if pde_type == "burgers":
        solver = BurgersClassicalSolver(
            nu=config.get("nu", 0.01),
            nx=config.get("nx", 256),
            nt=config.get("nt", 100),
        )
        X, T, u = solver.solve()

        # Plot
        from pde_solver.utils.visualization import plot_1d_time_evolution
        x = X[:, 0]
        t = T[0, :]
        plot_1d_time_evolution(
            x, t, u.T,
            title="Burgers Classical Solution",
            save_path="outputs/burgers_classical_solution.png",
        )

        print("Classical solver completed!")
    else:
        print(f"PDE type {pde_type} not yet implemented")


if __name__ == "__main__":
    app()

