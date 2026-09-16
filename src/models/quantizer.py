from __future__ import annotations

import torch
from torch import nn


class IntegerQuantizer(nn.Module):
    """
    Integer-quantized autoencoder bottleneck.

    The encoder produces a scalar z at each timestep.

    z is quantized to the nearest one of `num_buttons`
    evenly spaced values between -1 and 1.

    For 8 buttons:

        [-1.0,
         -0.7142857,
         -0.4285714,
         -0.1428571,
          0.1428571,
          0.4285714,
          0.7142857,
          1.0]

    Straight-through estimation is used so gradients flow from
    the decoder through the quantization operation.
    """

    centroids:torch.Tensor
    
    def __init__(
        self,
        num_buttons: int = 8,
    ):
        super().__init__()

        if num_buttons < 2:
            raise ValueError(
                "num_buttons must be at least 2."
            )

        self.num_buttons = num_buttons

        centroids = torch.linspace(
            -1.0,
            1.0,
            steps=num_buttons,
        )

        self.register_buffer(
            "centroids",
            centroids,
        )

    def forward(
        self,
        continuous_latents: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        continuous_latents:
            Shape:
                (batch_size, sequence_length)

        Returns
        -------
        Dictionary containing:

        button_indices:
            Integer button IDs in [0, num_buttons).

        quantized:
            Quantized values.

        straight_through:
            Quantized values in the forward pass, but with
            gradients equal to the identity with respect to
            continuous_latents.

        continuous:
            Original encoder outputs.
        """

        continuous = continuous_latents # (B, T)
        # print(self.centroids)
        # Shape becomes:
        #
        # (batch, time, num_buttons)
        distances = torch.abs(
            continuous.unsqueeze(-1)
            - self.centroids
        ) # (B, T, 1) - (N,) -> (B, T, N)

        button_indices = torch.argmin(
            distances,
            dim=-1,
        ) # (B, T)

        quantized = self.centroids[
            button_indices
        ] # (B, T)

        # Straight-through estimator:
        #
        # Forward:
        #     straight_through == quantized
        #
        # Backward:
        #     d(straight_through)/d(continuous) = 1
        straight_through = (
            continuous
            + (quantized - continuous).detach()
        ) # (B, T)

        return {
            "continuous": continuous,
            "button_indices": button_indices,
            "quantized": quantized,
            "straight_through": straight_through,
        }