from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset


# Project imports

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from src.data.piano_genie_dataset import PianoGenieDataset
from src.models.losses import PianoGenieLoss
from src.models.piano_genie import PianoGenie
from src.training.trainer import Trainer


# Configuration

SEED = 42

DATA_PATH = ROOT / "data" / "processed" / "pop909_events.pt"
SPLIT_PATH = ROOT / "data" / "processed" / "pop909_splits.json"

NUM_PITCHES = 57
NUM_BUTTONS = 8

SEQ_LEN = 128
BATCH_SIZE = 4

LEARNING_RATE = 3e-4

# Number of times the same batch is presented to Trainer.train().
NUM_EPOCHS = 400

PRINT_EVERY = 10

# We primarily require substantial loss reduction rather than a
# particular absolute loss, because the paper's objective is summed.
LOSS_REDUCTION_RATIO = 0.10


# Helpers

def move_batch_to_device(batch: dict[str, torch.Tensor], device:torch.device) -> dict[str,torch.Tensor]:
    """Move all tensor values in a dataset batch to the requested device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def reconstruction_accuracy(model:torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    """Compute note reconstruction accuracy on the fixed batch."""
    model.eval()

    with torch.no_grad():
        outputs = model(
            pitches=batch["pitches"],
            delta_times=batch["delta_times"],
        )

        predictions = outputs["logits"].argmax(dim=-1)
        targets = batch["pitches"]

        accuracy = (predictions == targets).float().mean()

    return accuracy.item()


def parameter_distance(initial_state: dict[str, torch.Tensor], model:torch.nn.Module) -> float:
    """L2 distance between initial and current model parameters."""
    distance_squared = 0.0

    for name, parameter in model.named_parameters():
        distance_squared += torch.sum(
            (parameter.detach() - initial_state[name]) ** 2
        ).item()

    return distance_squared ** 0.5


# Dataset

def make_real_training_loader() -> DataLoader:
    """
    Construct a DataLoader containing exactly one real POP909 batch.

    The dataset itself is still the normal training dataset. We simply
    take the first BATCH_SIZE examples from it so that every call to
    Trainer.train() sees the exact same batch.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_PATH}"
        )

    # Use the same training configuration as the actual project.
    dataset = PianoGenieDataset(
        dataset_path=DATA_PATH,
        split_path=SPLIT_PATH,
        split="train",
        seq_len=SEQ_LEN,
        seed=SEED,
    )

    if len(dataset) < BATCH_SIZE:
        raise RuntimeError(
            f"Training dataset contains only {len(dataset)} examples, "
            f"but BATCH_SIZE={BATCH_SIZE}."
        )

    # Exactly one fixed batch.
    single_batch_dataset = Subset(
        dataset,
        indices=list(range(BATCH_SIZE)),
    )

    loader = DataLoader(
        single_batch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return loader


# Test

def test_single_batch_overfit():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Dataset: {DATA_PATH}")

    # Real POP909 training batch

    train_loader = make_real_training_loader()
    val_loader = make_real_training_loader()

    batch = next(iter(train_loader))
    batch = move_batch_to_device(batch, device)

    print("\nFixed training batch:")
    print(f"  pitches:      {tuple(batch['pitches'].shape)}")
    print(f"  delta_times:  {tuple(batch['delta_times'].shape)}")

    assert batch["pitches"].shape == (
        BATCH_SIZE,
        SEQ_LEN,
    )

    assert batch["delta_times"].shape == (
        BATCH_SIZE,
        SEQ_LEN,
    )

    # Model / loss / optimizer

    model = PianoGenie(
        num_pitches=NUM_PITCHES,
        num_buttons=NUM_BUTTONS,
    )

    loss_fn = PianoGenieLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Trainer

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        grad_clip_norm=1.0,
        checkpoint_path=None,
    )

    # Initial evaluation
    # Save initial parameters so we can verify that Trainer.train()
    # actually updates the model.
    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    initial_metrics = trainer.eval()

    initial_loss = initial_metrics.total

    print("\nInitial metrics:")
    print(f"  total:          {initial_metrics.total:.6f}")
    print(f"  reconstruction: {initial_metrics.reconstruction:.6f}")
    print(f"  margin:         {initial_metrics.margin:.6f}")
    print(f"  contour:        {initial_metrics.contour:.6f}")

    assert torch.isfinite(
        torch.tensor(initial_loss)
    ), "Initial loss is not finite."

    initial_accuracy = reconstruction_accuracy(
        model,
        batch,
    )

    print(f"  accuracy:       {initial_accuracy:.4%}")

    # Repeatedly train on exactly the same real batch

    print("\nTraining on the same real POP909 batch...")

    loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):

        # This is the actual Trainer training loop.
        metrics = trainer.train()

        loss_history.append(metrics.total)

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            accuracy = reconstruction_accuracy(
                model,
                batch,
            )

            print(
                f"Epoch {epoch:4d} | "
                f"loss={metrics.total:.6f} | "
                f"recon={metrics.reconstruction:.6f} | "
                f"margin={metrics.margin:.6f} | "
                f"contour={metrics.contour:.6f} | "
                f"accuracy={accuracy:.4%}"
            )

    # Final evaluation through Trainer

    final_metrics = trainer.eval()

    final_loss = final_metrics.total

    print("\nFinal metrics:")
    print(f"  total:          {final_metrics.total:.6f}")
    print(f"  reconstruction: {final_metrics.reconstruction:.6f}")
    print(f"  margin:         {final_metrics.margin:.6f}")
    print(f"  contour:        {final_metrics.contour:.6f}")

    final_accuracy = reconstruction_accuracy(
        model,
        batch,
    )

    print(f"  accuracy:       {final_accuracy:.4%}")

    # Check parameter updates

    distance = parameter_distance(
        initial_state,
        model,
    )

    print(f"\nParameter L2 change: {distance:.6f}")

    assert distance > 0.0, (
        "Trainer.train() did not change any model parameters."
    )

    # Check loss reduction

    print(
        f"\nLoss ratio: "
        f"{final_loss / initial_loss:.4f}"
    )

    assert torch.isfinite(
        torch.tensor(final_loss)
    ), "Final loss is not finite."

    assert final_loss < initial_loss * LOSS_REDUCTION_RATIO, (
        "The model failed to substantially overfit the single "
        "real-data batch.\n"
        f"Initial loss: {initial_loss:.6f}\n"
        f"Final loss:   {final_loss:.6f}\n"
        f"Required:     < {initial_loss * LOSS_REDUCTION_RATIO:.6f}"
    )

    # Reconstruction check

    # We don't require 100% accuracy because the model contains an
    # 8-way information bottleneck and the objective includes margin
    # and contour terms. The loss-reduction criterion above is the
    # primary overfit criterion.
    assert final_accuracy > 0.50, (
        "Final reconstruction accuracy is unexpectedly low:\n"
        f"{final_accuracy:.4%}"
    )

    print("\nSingle-batch real-data overfit test PASSED.")


# Main

if __name__ == "__main__":
    test_single_batch_overfit()