from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Allow running this file directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from src.models.losses import PianoGenieLoss
from src.models.piano_genie import PianoGenie
from src.training.trainer import Trainer


class SyntheticPianoDataset(Dataset):
    """Small deterministic dataset for testing the Trainer."""

    def __init__(
        self,
        num_samples: int = 8,
        seq_len: int = 16,
        num_pitches: int = 55,
    ) -> None:
        generator = torch.Generator().manual_seed(42)

        self.pitches = torch.randint(
            low=0,
            high=num_pitches,
            size=(num_samples, seq_len),
            generator=generator,
        )

        # Positive integer delta times.
        self.delta_times = torch.randint(
            low=1,
            high=8,
            size=(num_samples, seq_len),
            generator=generator,
        )

    def __len__(self) -> int:
        return self.pitches.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "pitches": self.pitches[index],
            "delta_times": self.delta_times[index],
        }


def make_trainer(
    checkpoint_path: Path | None = None,
) -> Trainer:
    torch.manual_seed(42)

    num_pitches = 55
    num_buttons = 8

    model = PianoGenie(
        num_pitches=num_pitches,
        num_buttons=num_buttons,
    )

    loss_fn = PianoGenieLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    train_dataset = SyntheticPianoDataset(
        num_samples=8,
        seq_len=16,
        num_pitches=num_pitches,
    )

    val_dataset = SyntheticPianoDataset(
        num_samples=4,
        seq_len=16,
        num_pitches=num_pitches,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
    )

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        grad_clip_norm=1.0,
        checkpoint_path=checkpoint_path,
    )


def assert_finite_metrics(metrics) -> None:
    assert torch.isfinite(
        torch.tensor(metrics.total)
    ), "Total loss is not finite."

    assert torch.isfinite(
        torch.tensor(metrics.reconstruction)
    ), "Reconstruction loss is not finite."

    assert torch.isfinite(
        torch.tensor(metrics.margin)
    ), "Margin loss is not finite."

    assert torch.isfinite(
        torch.tensor(metrics.contour)
    ), "Contour loss is not finite."


def test_train() -> None:
    """train() should update model parameters."""
    trainer = make_trainer()

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }

    metrics = trainer.train()

    assert_finite_metrics(metrics)

    changed = False

    for name, parameter in trainer.model.named_parameters():
        if not parameter.requires_grad:
            continue

        if not torch.equal(before[name], parameter.detach()):
            changed = True
            break

    assert changed, "train() did not update any model parameters."

    print("PASS: train() updates model parameters.")


def test_eval() -> None:
    """eval() should not update model parameters."""
    trainer = make_trainer()

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    metrics = trainer.eval()

    assert_finite_metrics(metrics)

    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(
            before[name],
            parameter.detach(),
        ), f"eval() modified parameter: {name}"

    assert not trainer.model.training, (
        "eval() should leave the model in evaluation mode."
    )

    print("PASS: eval() does not update model parameters.")


def test_fit_history() -> None:
    """fit() should populate one history entry per completed epoch."""
    trainer = make_trainer()

    num_epochs = 3

    history = trainer.fit(
        num_epochs=num_epochs,
        early_stopping_patience=None,
        verbose=False,
    )

    expected_keys = {
        "train_loss",
        "train_reconstruction",
        "train_margin",
        "train_contour",
        "val_loss",
        "val_reconstruction",
        "val_margin",
        "val_contour",
    }

    assert set(history.keys()) == expected_keys

    for key in expected_keys:
        assert len(history[key]) == num_epochs, (
            f"{key} has {len(history[key])} entries; "
            f"expected {num_epochs}."
        )

        assert all(
            torch.isfinite(torch.tensor(value))
            for value in history[key]
        ), f"{key} contains a non-finite value."

    assert trainer.best_epoch is not None
    assert trainer.best_val_loss < float("inf")

    print("PASS: fit() records complete training history.")


def test_checkpointing() -> None:
    """fit() should save the best checkpoint and restore it."""
    checkpoint_path = (
        ROOT / "tmp_test_checkpoints" / "piano_genie_best.pt"
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    trainer = make_trainer(
        checkpoint_path=checkpoint_path,
    )

    trainer.fit(
        num_epochs=3,
        early_stopping_patience=None,
        verbose=False,
    )

    assert checkpoint_path.exists(), (
        "Best checkpoint was not created."
    )

    assert trainer.best_epoch is not None

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert "epoch" in checkpoint
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "val_metrics" in checkpoint
    assert "best_val_loss" in checkpoint

    # Create a fresh Trainer and verify that the checkpoint can be loaded.
    restored_trainer = make_trainer()

    restored_checkpoint = restored_trainer.load_checkpoint(
        checkpoint_path
    )

    assert (
        restored_checkpoint["epoch"]
        == trainer.best_epoch
    )

    for name, parameter in trainer.model.state_dict().items():
        restored_parameter = (
            restored_trainer.model.state_dict()[name]
        )

        assert torch.equal(
            parameter,
            restored_parameter,
        ), f"Checkpoint mismatch for parameter: {name}"

    print("PASS: checkpointing saves and restores the best model.")


def test_early_stopping() -> None:
    """fit() should respect the early-stopping patience argument."""
    trainer = make_trainer()

    history = trainer.fit(
        num_epochs=20,
        early_stopping_patience=2,
        verbose=False,
    )

    assert 1 <= len(history["train_loss"]) <= 20

    print(
        "PASS: early stopping runs without error "
        f"({len(history['train_loss'])} epochs)."
    )


def main() -> None:
    print("Testing Trainer...\n")

    test_train()
    test_eval()
    test_fit_history()
    test_checkpointing()
    test_early_stopping()

    print("\nAll Trainer tests passed.")


if __name__ == "__main__":
    main()