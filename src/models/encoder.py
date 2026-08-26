from __future__ import annotations

import torch
from torch import nn


class PianoGenieEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Maps a sequence of pitch tokens to one continuous scalar
    per timestep.

    Input:
        pitches: LongTensor of shape
            (batch_size, sequence_length)

    Output:
        continuous_latents: FloatTensor of shape
            (batch_size, sequence_length)

    The scalar output is subsequently quantized into one of the
    Piano Genie button values.
    """

    def __init__(
        self,
        num_pitches: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_pitches = num_pitches
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.pitch_embedding = nn.Embedding(
            num_embeddings=num_pitches,
            embedding_dim=embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional LSTM output has dimension:
        # 2 * hidden_size.
        #
        # Piano Genie IQAE produces one scalar per timestep.
        self.output_projection = nn.Linear(
            2 * hidden_size,
            1,
        )

    def forward(
        self,
        pitches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pitches:
            Integer pitch tokens.

            Shape:
                (batch_size, sequence_length)

        Returns
        -------
        continuous_latents:
            One real-valued scalar per timestep.

            Shape:
                (batch_size, sequence_length)
        """

        x = self.pitch_embedding(pitches) # (B, T, C)

        x, _ = self.lstm(x) # (B, T, 2*hidden)

        x = self.output_projection(x) # (B, T, 1)

        return x.squeeze(-1) # (B, T)