"""Training loop for physics-informed neural networks."""

from typing import Dict, Any, Optional, Callable, List
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
import os
import json

from pde_solver.training.multi_objective_loss import MultiObjectiveLoss


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    gradient_clip: float = 1.0
    use_scheduler: bool = True
    scheduler_patience: int = 50
    early_stopping_patience: int = 100
    checkpoint_freq: int = 100
    log_freq: int = 10
    use_wandb: bool = True
    wandb_project: str = "pde-solver"
    wandb_name: Optional[str] = None


class PhysicsInformedTrainer:
    """Trainer for physics-informed neural networks."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: MultiObjectiveLoss,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        """Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            Neural network model
        loss_fn : MultiObjectiveLoss
            Multi-objective loss function
        config : TrainingConfig
            Training configuration
        device : str
            Device to use
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        if config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=config.scheduler_patience,
            )
        else:
            self.scheduler = None

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if device != "cpu" else None

        # Initialize wandb
        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_name,
                    config=config.__dict__,
                )
            except ImportError:
                print("Warning: wandb not available, logging disabled")
                config.use_wandb = False

        # Training history
        self.history = {
            "train_loss": [],
            "loss_components": {},
            "val_loss": [],
        }

    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Optional[Dict[str, torch.Tensor]] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Parameters
        ----------
        train_data : Dict[str, torch.Tensor]
            Training data dictionary
        val_data : Dict[str, torch.Tensor], optional
            Validation data
        callbacks : List[Callable], optional
            List of callback functions

        Returns
        -------
        Dict[str, Any]
            Training history
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training step
            self.model.train()
            train_loss, loss_components = self._train_step(train_data)

            self.history["train_loss"].append(train_loss)
            for key, value in loss_components.items():
                if key not in self.history["loss_components"]:
                    self.history["loss_components"][key] = []
                self.history["loss_components"][key].append(value)

            # Validation
            if val_data is not None:
                val_loss = self._validate(val_data)
                self.history["val_loss"].append(val_loss)

                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if self.scheduler:
                    self.scheduler.step(train_loss)

            # Logging
            if epoch % self.config.log_freq == 0:
                self._log_metrics(epoch, train_loss, loss_components)

            # Checkpointing
            if epoch % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch)

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, self.history, self.model)

        return self.history

    def _train_step(self, train_data: Dict[str, torch.Tensor]) -> tuple:
        """Single training step.

        Parameters
        ----------
        train_data : Dict[str, torch.Tensor]
            Training data

        Returns
        -------
        tuple
            (total_loss, loss_components_dict)
        """
        self.optimizer.zero_grad()

        # Move data to device
        train_data_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in train_data.items()
        }

        # Forward pass
        if self.scaler:
            with torch.cuda.amp.autocast():
                loss, loss_components = self.loss_fn(
                    self.model, train_data_device
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, loss_components = self.loss_fn(self.model, train_data_device)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )

            self.optimizer.step()

        return loss.item(), {k: v.item() for k, v in loss_components.items()}

    def _validate(self, val_data: Dict[str, torch.Tensor]) -> float:
        """Validation step.

        Parameters
        ----------
        val_data : Dict[str, torch.Tensor]
            Validation data

        Returns
        -------
        float
            Validation loss
        """
        self.model.eval()
        val_data_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in val_data.items()
        }

        with torch.no_grad():
            loss, _ = self.loss_fn(self.model, val_data_device)

        return loss.item()

    def _log_metrics(
        self, epoch: int, loss: float, loss_components: Dict[str, float]
    ) -> None:
        """Log training metrics.

        Parameters
        ----------
        epoch : int
            Current epoch
        loss : float
            Total loss
        loss_components : Dict[str, float]
            Loss components
        """
        log_dict = {"epoch": epoch, "loss": loss, **loss_components}

        if self.config.use_wandb:
            try:
                import wandb

                wandb.log(log_dict, step=epoch)
            except:
                pass

        print(f"Epoch {epoch}: Loss = {loss:.6e}", end="")
        for key, value in loss_components.items():
            print(f", {key} = {value:.6e}", end="")
        print()

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config.__dict__,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = f"checkpoints/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.history = checkpoint.get("history", self.history)

