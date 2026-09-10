from __future__ import annotations

import torch
from torch import nn


class PianoGenieDecoder(nn.Module):
    """
    Unidirectional autoregressive LSTM decoder.

    At each timestep, the decoder receives:

        1. Previous pitch token.
        2. Current quantized button value.
        3. Current delta-time token.

    It predicts a distribution over the current pitch.

    During training, teacher forcing is used.
    """

    def __init__(
        self,
        num_pitches: int,
        pitch_embedding_dim: int = 128,
        button_embedding_dim: int = 32,
        time_embedding_dim: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        max_delta_time: int = 32,
    ):
        super().__init__()

        self.num_pitches = num_pitches
        self.max_delta_time = max_delta_time

        self.start_pitch_token = num_pitches

        self.pitch_embedding = nn.Embedding(
            num_embeddings=num_pitches + 1,
            embedding_dim=pitch_embedding_dim,
        )

        # We embed the button ID rather than the scalar centroid.
        #
        # The button IDs are discrete controller values:
        #
        # 0, 1, ..., 7
        self.button_projection = nn.Linear(
            1,
            button_embedding_dim,
        )

        # Token 0 is START.
        #
        # Actual delta time d is represented as:
        #
        # min(d, max_delta_time) + 1
        self.time_embedding = nn.Embedding(
            num_embeddings=max_delta_time + 2,
            embedding_dim=time_embedding_dim,
        )

        input_size = (
            pitch_embedding_dim
            + button_embedding_dim
            + time_embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_projection = nn.Linear(
            hidden_size,
            num_pitches,
        )

    def _prepare_delta_times(
        self,
        delta_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert raw delta times into embedding tokens.

        Dataset representation:

            first note:
                0

            other notes:
                actual delta time

        Model representation:

            0:
                START

            1 ... max_delta_time + 1:
                clipped delta times shifted by 1
        """

        delta_tokens = torch.clamp(
            delta_times,
            min=0,
            max=self.max_delta_time,
        )

        delta_tokens = delta_tokens + 1

        # Restore the first-note START token.
        delta_tokens = torch.where(
            delta_times == 0,
            torch.zeros_like(delta_tokens),
            delta_tokens,
        )

        return delta_tokens

    def forward(
        self,
        target_pitches: torch.Tensor,
        quantized_latents: torch.Tensor,
        delta_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode with teacher forcing.

        Parameters
        ----------
        target_pitches:
            Ground-truth pitch tokens.

            Shape:
                (batch_size, sequence_length)

        quantized_latents:
            Quantized button latents.

            Shape:
                (batch_size, sequence_length)

        delta_times:
            Raw delta times.

            Shape:
                (batch_size, sequence_length)

        Returns
        -------
        logits:
            Pitch logits.

            Shape:
                (batch_size, sequence_length, num_pitches)
        """

        batch_size, sequence_length = (
            target_pitches.shape
        )

        # Shift pitch sequence right.
        #
        # At timestep t, the decoder sees pitch t - 1.

        start_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.start_pitch_token,
            dtype=target_pitches.dtype,
            device=target_pitches.device,
        ) # (B, 1)

        previous_pitches = torch.cat(
            [
                start_tokens,
                target_pitches[:, :-1],
            ],
            dim=1,
        ) # (B, T)

        # Embeddings

        pitch_features = self.pitch_embedding(
            previous_pitches
        ) # (B, T, C_pitch)

        button_features = self.button_projection(
            quantized_latents.unsqueeze(-1)
        ) # (B, T, C_button)

        delta_tokens = self._prepare_delta_times(
            delta_times
        ) # (B, T)

        time_features = self.time_embedding(
            delta_tokens
        ) # (B, T, C_time)

        # Recurrent decoder

        decoder_input = torch.cat(
            [
                pitch_features,
                button_features,
                time_features,
            ],
            dim=-1,
        ) # (B, T, C_total)

        hidden, _ = self.lstm(
            decoder_input
        ) # (B, T, hidden)

        logits = self.output_projection(
            hidden
        ) # (B, T, V)

        return logits