"""Visualization utilities for PDE solutions."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


def visualize_solution(
    coords: np.ndarray,
    solution: np.ndarray,
    title: str = "Solution",
    save_path: Optional[str] = None,
    cmap: str = "viridis",
) -> None:
    """Visualize 2D solution field.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array of shape (N, 2) or (N, 3)
    solution : np.ndarray
        Solution values of shape (N,) or (N, 1)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    cmap : str
        Colormap name
    """
    if solution.ndim > 1:
        solution = solution.squeeze()

    fig, ax = plt.subplots(figsize=(10, 8))

    if coords.shape[1] == 2:  # 2D spatial
        x, y = coords[:, 0], coords[:, 1]
        scatter = ax.scatter(x, y, c=solution, cmap=cmap, s=1)
        plt.colorbar(scatter, ax=ax, label="Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif coords.shape[1] == 3:  # 2D spatial + time
        # Show spatial distribution at a fixed time
        x, y, t = coords[:, 0], coords[:, 1], coords[:, 2]
        t_idx = len(np.unique(t)) // 2  # Middle time slice
        unique_t = np.unique(t)[t_idx]
        mask = np.abs(t - unique_t) < 1e-6
        scatter = ax.scatter(
            x[mask], y[mask], c=solution[mask], cmap=cmap, s=1
        )
        plt.colorbar(scatter, ax=ax, label="Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title} at t={unique_t:.3f}")

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.close()


def plot_1d_time_evolution(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    title: str = "Time Evolution",
    save_path: Optional[str] = None,
    n_snapshots: int = 10,
) -> None:
    """Plot 1D solution evolution over time.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates (nx,)
    t : np.ndarray
        Time coordinates (nt,)
    u : np.ndarray
        Solution array (nt, nx)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    n_snapshots : int
        Number of time snapshots to show
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    time_indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for i, idx in enumerate(time_indices):
        ax.plot(x, u[idx, :], label=f"t={t[idx]:.3f}", color=colors[i])

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.close()


def create_animation(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    save_path: str,
    title: str = "Solution Animation",
    interval: int = 50,
) -> None:
    """Create animated GIF of solution evolution.

    Parameters
    ----------
    x : np.ndarray
        Spatial coordinates
    t : np.ndarray
        Time coordinates
    u : np.ndarray
        Solution array (nt, nx)
    save_path : str
        Path to save animation
    title : str
        Animation title
    interval : int
        Frame interval in milliseconds
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, u[0, :], "b-", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(title)
    ax.set_ylim(u.min(), u.max())
    ax.grid(True, alpha=0.3)

    def animate(frame):
        line.set_ydata(u[frame, :])
        ax.set_title(f"{title} - t={t[frame]:.3f}")
        return line,

    anim = animation.FuncAnimation(
        fig, animate, frames=len(t), interval=interval, blit=True
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer="pillow", fps=10)
    print(f"Saved animation to {save_path}")
    plt.close()


def plot_loss_history(
    history: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot training loss history.

    Parameters
    ----------
    history : dict
        Training history dictionary
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss components
    if "loss_components" in history:
        for key, values in history["loss_components"].items():
            axes[1].plot(values, label=key)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss Components")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.close()

