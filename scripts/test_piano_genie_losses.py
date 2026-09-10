from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.losses import PianoGenieLoss


BATCH_SIZE = 4
SEQUENCE_LENGTH = 8
NUM_PITCHES = 55

SEED = 42


def test_reconstruction_loss():
    torch.manual_seed(SEED)

    loss_fn = PianoGenieLoss()

    logits = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        NUM_PITCHES,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        NUM_PITCHES,
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
    )

    loss = loss_fn.reconstruction_loss(
        logits,
        targets,
    )

    if not torch.isfinite(loss):
        raise AssertionError(
            "Reconstruction loss is not finite."
        )

    loss.backward()

    if logits.grad is None:
        raise AssertionError(
            "Reconstruction loss produced no "
            "gradient for logits."
        )

    print(
        "PASS: reconstruction loss"
    )


def test_margin_loss():
    loss_fn = PianoGenieLoss()

    # Values inside [-1, 1] should incur zero loss.
    inside = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0]
        ]
    )

    loss_inside = loss_fn.margin_loss(
        inside
    )

    if loss_inside.item() != 0.0:
        raise AssertionError(
            "Margin loss should be zero for "
            "values inside [-1, 1]."
        )

    # Values outside the interval should incur
    # positive loss.
    outside = torch.tensor(
        [
            [-2.0, 0.0, 1.5]
        ]
    )

    loss_outside = loss_fn.margin_loss(
        outside
    )

    if loss_outside.item() <= 0.0:
        raise AssertionError(
            "Margin loss should be positive for "
            "values outside [-1, 1]."
        )

    print(
        "PASS: margin loss"
    )


def test_contour_loss():
    loss_fn = PianoGenieLoss()

    pitches = torch.tensor(
        [
            [60, 62, 64, 65]
        ],
        dtype=torch.long,
    )

    # Encoder representation has the same increasing
    # contour.
    matching = torch.tensor(
        [
            [0.0, 0.5, 1.0, 1.5]
        ],
        requires_grad=True,
    )

    matching_loss = loss_fn.contour_loss(
        pitches,
        matching,
    )

    # Because the encoder intervals are positive and
    # the pitch intervals are positive, this should be
    # substantially smaller than an opposite-direction
    # contour.
    opposite = torch.tensor(
        [
            [1.5, 1.0, 0.5, 0.0]
        ],
        requires_grad=True,
    )

    opposite_loss = loss_fn.contour_loss(
        pitches,
        opposite,
    )

    if opposite_loss <= matching_loss:
        raise AssertionError(
            "Opposite contour should have a greater "
            "contour penalty than matching contour."
        )

    matching_loss.backward()

    if matching.grad is None:
        raise AssertionError(
            "Contour loss produced no gradient."
        )

    print(
        "PASS: contour loss"
    )


def test_full_loss():
    torch.manual_seed(SEED)

    loss_fn = PianoGenieLoss()

    logits = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        NUM_PITCHES,
        requires_grad=True,
    )

    pitches = torch.randint(
        0,
        NUM_PITCHES,
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
    )

    continuous_latents = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        requires_grad=True,
    )

    output = loss_fn(
        logits=logits,
        target_pitches=pitches,
        continuous_latents=continuous_latents,
    )

    if not torch.isfinite(output.total):
        raise AssertionError(
            "Total loss is not finite."
        )

    output.total.backward()

    if logits.grad is None:
        raise AssertionError(
            "Total loss produced no decoder gradient."
        )

    if continuous_latents.grad is None:
        raise AssertionError(
            "Total loss produced no encoder gradient."
        )

    print(
        "PASS: complete Piano Genie loss"
    )

    print(
        f"  reconstruction: "
        f"{output.reconstruction.item():.6f}"
    )

    print(
        f"  margin:         "
        f"{output.margin.item():.6f}"
    )

    print(
        f"  contour:        "
        f"{output.contour.item():.6f}"
    )

    print(
        f"  total:          "
        f"{output.total.item():.6f}"
    )


def main():
    print("=" * 60)
    print("Piano Genie Loss Tests")
    print("=" * 60)

    test_reconstruction_loss()
    test_margin_loss()
    test_contour_loss()
    test_full_loss()

    print("=" * 60)
    print("ALL LOSS TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()