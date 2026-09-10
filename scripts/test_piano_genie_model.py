from pathlib import Path
import sys

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.encoder import PianoGenieEncoder
from src.models.quantizer import IntegerQuantizer
from src.models.decoder import PianoGenieDecoder
from src.models.piano_genie import PianoGenie


BATCH_SIZE = 4
SEQUENCE_LENGTH = 16

NUM_PITCHES = 55
NUM_BUTTONS = 8

SEED = 42


def assert_shape(
    tensor: torch.Tensor,
    expected_shape: tuple,
    name: str,
):
    if tuple(tensor.shape) != expected_shape:
        raise AssertionError(
            f"{name} has incorrect shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {tuple(tensor.shape)}"
        )


def assert_has_gradient(
    parameter: torch.Tensor,
    name: str,
):
    if parameter.grad is None:
        raise AssertionError(
            f"{name} has no gradient."
        )

    if not torch.any(parameter.grad != 0):
        raise AssertionError(
            f"{name} has an all-zero gradient."
        )


def create_synthetic_batch():
    """
    Create synthetic pitch and timing sequences.

    Pitch tokens:
        integers in [0, NUM_PITCHES)

    Delta times:
        first timestep is 0 (START)
        remaining timesteps are positive integers.
    """

    pitches = torch.randint(
        low=0,
        high=NUM_PITCHES,
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
    )

    delta_times = torch.randint(
        low=1,
        high=16,
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
    )

    # The first timestep represents the beginning of
    # an independent sequence.
    delta_times[:, 0] = 0

    return pitches, delta_times


def test_encoder(
    pitches: torch.Tensor,
):
    """
    Check that the encoder produces one scalar latent
    per sequence timestep.
    """

    encoder = PianoGenieEncoder(
        num_pitches=NUM_PITCHES,
    )

    continuous_latents = encoder(
        pitches
    )

    expected_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    assert_shape(
        continuous_latents,
        expected_shape,
        "Encoder output",
    )

    print(
        "PASS: encoder output shape "
        f"{expected_shape}"
    )


def test_quantizer():
    """
    Check that every quantized output belongs exactly
    to one of the 8 fixed centroids.
    """

    quantizer = IntegerQuantizer(
        num_buttons=NUM_BUTTONS,
    )

    continuous_latents = torch.tensor(
        [
            [
                -2.0,
                -0.9,
                -0.5,
                -0.1,
                0.1,
                0.5,
                0.9,
                2.0,
            ]
        ],
        dtype=torch.float,
    )

    output = quantizer(
        continuous_latents
    )

    quantized = output["quantized"]

    assert_shape(
        quantized,
        (1, 8),
        "Quantizer output",
    )

    centroids = quantizer.centroids

    # Each quantized value must exactly equal one
    # of the registered centroid values.
    matches_centroid = (
        quantized.unsqueeze(-1)
        == centroids
    ).any(dim=-1)

    if not torch.all(matches_centroid):
        raise AssertionError(
            "Quantizer produced a value that is not "
            "one of the fixed centroids."
        )

    # Also verify that the returned quantized values
    # agree with button_indices.
    expected_quantized = centroids[
        output["button_indices"]
    ]

    if not torch.equal(
        quantized,
        expected_quantized,
    ):
        raise AssertionError(
            "Quantized values do not match "
            "button_indices."
        )

    print(
        "PASS: quantizer outputs only valid "
        f"{NUM_BUTTONS} centroids"
    )

    print(
        "Centroids:",
        centroids.tolist(),
    )

    print(
        "Quantized values:",
        quantized.tolist(),
    )


def test_straight_through_gradient(
    pitches: torch.Tensor,
):
    """
    Verify that a loss depending on the
    straight-through quantized latent produces
    gradients for encoder parameters.

    This isolates the critical gradient path:

        encoder
            ->
        continuous latent
            ->
        quantizer
            ->
        straight-through latent
            ->
        loss
    """

    encoder = PianoGenieEncoder(
        num_pitches=NUM_PITCHES,
    )

    quantizer = IntegerQuantizer(
        num_buttons=NUM_BUTTONS,
    )

    continuous_latents = encoder(
        pitches
    )

    quantizer_output = quantizer(
        continuous_latents
    )

    straight_through = (
        quantizer_output["straight_through"]
    )

    # A simple differentiable loss.
    loss = straight_through.pow(2).mean()

    loss.backward()

    assert_has_gradient(
        encoder.output_projection.weight,
        "Encoder output_projection.weight",
    )

    assert_has_gradient(
        encoder.pitch_embedding.weight,
        "Encoder pitch_embedding.weight",
    )

    print(
        "PASS: straight-through gradients reach "
        "the encoder"
    )


def test_decoder(
    pitches: torch.Tensor,
    delta_times: torch.Tensor,
):
    """
    Check that the decoder produces a pitch-logit
    vector for every timestep.
    """

    decoder = PianoGenieDecoder(
        num_pitches=NUM_PITCHES,
    )

    # Use arbitrary scalar button values here.
    #
    # In the full model these come from the
    # straight-through quantizer.
    quantized_latents = torch.empty(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    ).uniform_(-1.0, 1.0)

    logits = decoder(
        target_pitches=pitches,
        quantized_latents=quantized_latents,
        delta_times=delta_times,
    )

    expected_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        NUM_PITCHES,
    )

    assert_shape(
        logits,
        expected_shape,
        "Decoder output",
    )

    print(
        "PASS: decoder output shape "
        f"{expected_shape}"
    )


def test_full_model_and_reconstruction_gradients(
    pitches: torch.Tensor,
    delta_times: torch.Tensor,
):
    """
    Verify the complete forward and backward path.

    Specifically:

        reconstruction loss
                |
                v
             decoder
                |
                v
       straight-through latent
                |
                v
             encoder

    Both encoder and decoder must receive gradients.
    """

    model = PianoGenie(
        num_pitches=NUM_PITCHES,
        num_buttons=NUM_BUTTONS,
    )

    model.train()

    output = model(
        pitches=pitches,
        delta_times=delta_times,
    )

    logits = output["logits"]

    expected_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        NUM_PITCHES,
    )

    assert_shape(
        logits,
        expected_shape,
        "Full model logits",
    )

    # CrossEntropyLoss expects:
    #
    # input:
    #     (N, C)
    #
    # target:
    #     (N,)
    #
    # Flatten batch and time dimensions.
    reconstruction_loss = F.cross_entropy(
        logits.reshape(
            -1,
            NUM_PITCHES,
        ),
        pitches.reshape(-1),
    )

    if not torch.isfinite(
        reconstruction_loss
    ):
        raise AssertionError(
            "Reconstruction loss is not finite."
        )

    reconstruction_loss.backward()

    # Encoder gradients

    assert_has_gradient(
        model.encoder.output_projection.weight,
        "Encoder output_projection.weight",
    )

    assert_has_gradient(
        model.encoder.pitch_embedding.weight,
        "Encoder pitch_embedding.weight",
    )

    # Decoder gradients

    assert_has_gradient(
        model.decoder.output_projection.weight,
        "Decoder output_projection.weight",
    )

    assert_has_gradient(
        model.decoder.pitch_embedding.weight,
        "Decoder pitch_embedding.weight",
    )

    assert_has_gradient(
        model.decoder.button_projection.weight,
        "Decoder button_projection.weight",
    )

    print(
        "PASS: reconstruction loss is finite"
    )

    print(
        "PASS: reconstruction loss produces "
        "encoder gradients"
    )

    print(
        "PASS: reconstruction loss produces "
        "decoder gradients"
    )

    print(
        f"Reconstruction loss: "
        f"{reconstruction_loss.item():.6f}"
    )


def main():
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Piano Genie Model Tests")
    print("=" * 60)

    pitches, delta_times = (
        create_synthetic_batch()
    )

    print(
        "\nSynthetic input shapes:"
    )

    print(
        "pitches:",
        tuple(pitches.shape),
    )

    print(
        "delta_times:",
        tuple(delta_times.shape),
    )

    print("\n" + "-" * 60)
    print("Testing encoder")
    print("-" * 60)

    test_encoder(
        pitches
    )

    print("\n" + "-" * 60)
    print("Testing quantizer")
    print("-" * 60)

    test_quantizer()

    print("\n" + "-" * 60)
    print("Testing straight-through gradients")
    print("-" * 60)

    test_straight_through_gradient(
        pitches
    )

    print("\n" + "-" * 60)
    print("Testing decoder")
    print("-" * 60)

    test_decoder(
        pitches,
        delta_times,
    )

    print("\n" + "-" * 60)
    print("Testing complete model")
    print("-" * 60)

    test_full_model_and_reconstruction_gradients(
        pitches,
        delta_times,
    )

    print("\n" + "=" * 60)
    print("ALL MODEL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()