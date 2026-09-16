from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.losses import PianoGenieLoss


@dataclass
class EpochMetrics:
    total: float
    reconstruction: float
    margin: float
    contour: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.total,
            "reconstruction": self.reconstruction,
            "margin": self.margin,
            "contour": self.contour,
        }


class Trainer:
    """Trainer for Piano Genie.

    The trainer expects each dataloader batch to contain:
        pitches: Tensor[B, T]
        delta_times: Tensor[B, T]

    `pitches` should contain the target note sequence. `delta_times`
    contains the elapsed time since the previous note.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: PianoGenieLoss,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        device: torch.device | str | None = None,
        grad_clip_norm: float | None = 1.0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = torch.device(device)

        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        self.model.to(self.device)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_reconstruction": [],
            "train_margin": [],
            "train_contour": [],
            "val_loss": [],
            "val_reconstruction": [],
            "val_margin": [],
            "val_contour": [],
        }

        self.best_val_loss = float("inf")
        self.best_epoch: int | None = None

    def _move_batch_to_device(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device)
            for key, value in batch.items()
        }

    def _forward_and_loss(
        self,
        batch: dict[str, torch.Tensor],
    ):
        pitches = batch["pitches"]
        delta_times = batch["delta_times"]

        outputs = self.model(
            pitches=pitches,
            delta_times=delta_times,
        )

        losses = self.loss_fn(
            logits=outputs["logits"],
            target_pitches=pitches,
            continuous_latents=outputs["continuous_latents"],
        )

        return outputs, losses

    @staticmethod
    def _accumulate_metrics(
        totals: dict[str, float],
        losses: Any,
    ) -> None:
        totals["total"] += losses.total.item()
        totals["reconstruction"] += losses.reconstruction.item()
        totals["margin"] += losses.margin.item()
        totals["contour"] += losses.contour.item()

    def _epoch_metrics(
        self,
        totals: dict[str, float],
        num_batches: int,
    ) -> EpochMetrics:
        return EpochMetrics(
            total=totals["total"] / num_batches,
            reconstruction=totals["reconstruction"] / num_batches,
            margin=totals["margin"] / num_batches,
            contour=totals["contour"] / num_batches,
        )

    def train(self) -> EpochMetrics:
        """Run one training epoch."""
        self.model.train()

        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "margin": 0.0,
            "contour": 0.0,
        }

        num_batches = 0

        for batch in self.train_loader:
            

            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            
            _, losses = self._forward_and_loss(batch)

            losses.total.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm,
                )
            self.optimizer.step()

            self._accumulate_metrics(totals, losses)
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Training DataLoader is empty.")

        return self._epoch_metrics(totals, num_batches)

    @torch.no_grad()
    def eval(self) -> EpochMetrics:
        """Run one validation epoch."""
        self.model.eval()

        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "margin": 0.0,
            "contour": 0.0,
        }

        num_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)

            _, losses = self._forward_and_loss(batch)

            self._accumulate_metrics(totals, losses)
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Validation DataLoader is empty.")

        return self._epoch_metrics(totals, num_batches)

    def _save_checkpoint(self, epoch: int, val_metrics: EpochMetrics) -> None:
        if self.checkpoint_path is None:
            return

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics.as_dict(),
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(
        self,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load a previously saved checkpoint."""
        path = Path(path) if path is not None else self.checkpoint_path

        if path is None:
            raise ValueError("No checkpoint path was provided.")

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )

        return checkpoint

    def fit(
        self,
        num_epochs: int,
        *,
        early_stopping_patience: int | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train for multiple epochs and restore the best model.

        Args:
            num_epochs:
                Maximum number of epochs.
            early_stopping_patience:
                Number of consecutive epochs without validation
                improvement before stopping. None disables early stopping.
            verbose:
                Print epoch-level metrics.
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")

        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train()
            val_metrics = self.eval()

            self.history["train_loss"].append(train_metrics.total)
            self.history["train_reconstruction"].append(
                train_metrics.reconstruction
            )
            self.history["train_margin"].append(train_metrics.margin)
            self.history["train_contour"].append(train_metrics.contour)

            self.history["val_loss"].append(val_metrics.total)
            self.history["val_reconstruction"].append(
                val_metrics.reconstruction
            )
            self.history["val_margin"].append(val_metrics.margin)
            self.history["val_contour"].append(val_metrics.contour)

            improved = val_metrics.total < self.best_val_loss

            if improved:
                self.best_val_loss = val_metrics.total
                self.best_epoch = epoch
                epochs_without_improvement = 0

                self._save_checkpoint(
                    epoch=epoch,
                    val_metrics=val_metrics,
                )
            else:
                epochs_without_improvement += 1

            if verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train loss {train_metrics.total:.4f} "
                    f"(recon {train_metrics.reconstruction:.4f}, "
                    f"margin {train_metrics.margin:.4f}, "
                    f"contour {train_metrics.contour:.4f}) | "
                    f"val loss {val_metrics.total:.4f} "
                    f"(recon {val_metrics.reconstruction:.4f}, "
                    f"margin {val_metrics.margin:.4f}, "
                    f"contour {val_metrics.contour:.4f})"
                )

            if (
                early_stopping_patience is not None
                and epochs_without_improvement
                >= early_stopping_patience
            ):
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"Best validation loss: "
                        f"{self.best_val_loss:.4f} "
                        f"(epoch {self.best_epoch})."
                    )
                break

        # Restore the best validation model.
        if self.checkpoint_path is not None and self.best_epoch is not None:
            self.load_checkpoint()

        return self.history