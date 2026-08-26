from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import PianoGenieEncoder
from src.models.quantizer import IntegerQuantizer
from src.models.decoder import PianoGenieDecoder


class PianoGenie(nn.Module):
    """
    Complete Piano Genie IQAE model.

    Architecture:

        pitch sequence
              |
              v
        Bidirectional LSTM
              |
              v
        continuous scalar z_t
              |
              v
        8-level quantization
              |
              v
        discrete button sequence
              |
              v
        Unidirectional autoregressive LSTM
              |
              v
        reconstructed pitch logits
    """

    def __init__(
        self,
        num_pitches: int,
        num_buttons: int = 8,
        encoder_embedding_dim: int = 128,
        encoder_hidden_size: int = 256,
        decoder_pitch_embedding_dim: int = 128,
        decoder_button_embedding_dim: int = 32,
        decoder_time_embedding_dim: int = 32,
        decoder_hidden_size: int = 512,
        decoder_num_layers: int = 2,
        max_delta_time: int = 32,
    ):
        super().__init__()

        self.num_pitches = num_pitches
        self.num_buttons = num_buttons

        self.encoder = PianoGenieEncoder(
            num_pitches=num_pitches,
            embedding_dim=encoder_embedding_dim,
            hidden_size=encoder_hidden_size,
        )

        self.quantizer = IntegerQuantizer(
            num_buttons=num_buttons,
        )

        self.decoder = PianoGenieDecoder(
            num_pitches=num_pitches,
            pitch_embedding_dim=decoder_pitch_embedding_dim,
            button_embedding_dim=decoder_button_embedding_dim,
            time_embedding_dim=decoder_time_embedding_dim,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            max_delta_time=max_delta_time,
        )

    def forward(
        self,
        pitches: torch.Tensor,
        delta_times: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        pitches:
            Pitch tokens.

            Shape:
                (batch_size, sequence_length)

        delta_times:
            Raw onset delta times.

            Shape:
                (batch_size, sequence_length)

        Returns
        -------
        Dictionary containing:

            continuous_latents:
                Encoder scalar outputs.

            button_indices:
                Integer button IDs.

            quantized_latents:
                Quantized scalar values.

            logits:
                Reconstructed pitch logits.
        """

        continuous_latents = self.encoder(
            pitches
        )

        quantizer_output = self.quantizer(
            continuous_latents
        )

        logits = self.decoder(
            target_pitches=pitches,
            quantized_latents=quantizer_output["straight_through"],
            delta_times=delta_times,
        )

        return {
            "continuous_latents": quantizer_output["continuous"],

            "button_indices": quantizer_output["button_indices"],

            "quantized_latents": quantizer_output["quantized"],

            "straight_through_latents": quantizer_output["straight_through"],

            "logits": logits,
        }