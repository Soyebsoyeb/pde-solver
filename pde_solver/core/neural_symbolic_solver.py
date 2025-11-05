"""High-level solver orchestration interface."""

from typing import Dict, Any, Optional, Callable
import torch
import numpy as np
from dataclasses import dataclass

from pde_solver.core.physics_constraints import PhysicsConstraints
from pde_solver.utils.visualization import visualize_solution


@dataclass
class SolverConfig:
    """Configuration for neural-symbolic solver."""

    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    use_mixed_precision: bool = False
    deterministic: bool = True


class NeuralSymbolicSolver:
    """High-level solver orchestrating discretizers, models, and training."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: SolverConfig,
        physics_constraints: Optional[PhysicsConstraints] = None,
    ):
        """Initialize solver.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network model (PINN, DeepONet, etc.)
        config : SolverConfig
            Solver configuration
        physics_constraints : PhysicsConstraints, optional
            Physics constraints for hard/soft enforcement
        """
        self.model = model
        self.config = config
        self.physics_constraints = physics_constraints or PhysicsConstraints()

        # Set device and dtype
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.model = self.model.to(self.device).to(self.dtype)

        # Set deterministic behavior
        if config.deterministic:
            self._set_deterministic()

        # Mixed precision scaler
        self.scaler = (
            torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        )

    def _set_deterministic(self) -> None:
        """Set deterministic random seeds and algorithms."""
        import random
        import os

        os.environ["PYTHONHASHSEED"] = str(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            print(f"Warning: Could not set deterministic algorithms: {e}")

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        callbacks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        loss_fn : Callable
            Loss function
        optimizer : Optimizer
            Optimizer
        num_epochs : int
            Number of training epochs
        val_loader : DataLoader, optional
            Validation data loader
        callbacks : list, optional
            List of callback functions

        Returns
        -------
        Dict[str, Any]
            Training history with losses and metrics
        """
        self.model.train()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                # Forward pass (with mixed precision if enabled)
                if self.config.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        loss = loss_fn(batch, self.model)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss = loss_fn(batch, self.model)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_loss)

            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader, loss_fn)
                history["val_loss"].append(val_loss)

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, history, self.model)

        return history

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
    ) -> float:
        """Evaluate model on data.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader
        loss_fn : Callable
            Loss function

        Returns
        -------
        float
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                loss = loss_fn(batch, self.model)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def infer(self, coords: torch.Tensor) -> torch.Tensor:
        """Run inference on coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinate tensor of shape (N, dim) where dim is spatial + temporal

        Returns
        -------
        torch.Tensor
            Model predictions
        """
        self.model.eval()
        coords = coords.to(self.device).to(self.dtype)

        with torch.no_grad():
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(coords)
            else:
                predictions = self.model(coords)

        return predictions

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path
        optimizer : Optimizer, optional
            Optimizer state
        epoch : int, optional
            Epoch number
        metadata : dict, optional
            Additional metadata
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "epoch": epoch,
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path

        Returns
        -------
        Dict[str, Any]
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


