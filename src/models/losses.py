from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PianoGenieLossOutput:
    """
    Individual components of the Piano Genie objective.
    """

    total: torch.Tensor
    reconstruction: torch.Tensor
    margin: torch.Tensor
    contour: torch.Tensor


class PianoGenieLoss(nn.Module):
    """
    Piano Genie IQAE training objective.

    L = L_recons + L_margin + L_contour

    Reconstruction:
        Negative log likelihood of the target pitch
        under the decoder distribution.

    Margin:
        Penalizes continuous encoder outputs outside [-1, 1].

    Contour:
        Encourages the finite differences of the real-valued
        encoder representation to have the same direction
        as the finite differences of the input melody.

    Notes
    -----
    The original paper defines these losses as sums over
    timesteps. By default, this implementation preserves that
    behavior.
    """

    def __init__(
        self,
        reduction: str = "sum",
    ):
        super().__init__()

        if reduction not in {"sum", "mean"}:
            raise ValueError(
                "reduction must be either 'sum' or 'mean'."
            )

        self.reduction = reduction

    def reconstruction_loss(
        self,
        logits: torch.Tensor,
        target_pitches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute decoder reconstruction loss.

        Parameters
        ----------
        logits:
            Decoder logits.

            Shape:
                (B, T, num_pitches)

        target_pitches:
            Target pitch indices.

            Shape:
                (B, T)

        Returns
        -------
        torch.Tensor
            Negative log likelihood.
        """

        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape "
                "(B, T, num_pitches)."
            )

        if target_pitches.ndim != 2:
            raise ValueError(
                "target_pitches must have shape "
                "(B, T)."
            )

        if logits.shape[:2] != target_pitches.shape:
            raise ValueError(
                "The batch and sequence dimensions of "
                "logits and target_pitches must match."
            )

        # Cross entropy is equivalent to:
        #
        # -log P(target_pitch | decoder)
        #
        # Flatten B and T because cross_entropy expects
        # class logits in dimension 1.
        logits_flat = logits.reshape(
            -1,
            logits.shape[-1],
        )

        targets_flat = target_pitches.reshape(-1)

        return F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction=self.reduction,
        )

    def margin_loss(
        self,
        continuous_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize encoder outputs outside [-1, 1].

        L_margin =
            sum max(|enc(x)| - 1, 0)^2
        """

        excess = (
            continuous_latents.abs() - 1.0
        ).clamp_min(0.0)

        loss = excess.pow(2)

        if self.reduction == "sum":
            return loss.sum()

        return loss.mean()

    def contour_loss(
        self,
        pitches: torch.Tensor,
        continuous_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage encoder contours to follow melodic contours.

        L_contour =
            sum max(
                1 - Delta(x) * Delta(enc_s(x)),
                0
            )^2

        where:

            Delta(x_t) =
                x_t - x_{t-1}

            Delta(enc_s(x_t)) =
                enc_s(x_t) - enc_s(x_{t-1})

        The encoder representation used here is the
        continuous, pre-quantization representation.
        """

        if pitches.ndim != 2:
            raise ValueError(
                "pitches must have shape (B, T)."
            )

        if continuous_latents.ndim != 2:
            raise ValueError(
                "continuous_latents must have shape (B, T)."
            )

        if pitches.shape != continuous_latents.shape:
            raise ValueError(
                "pitches and continuous_latents must "
                "have identical shapes."
            )

        if pitches.shape[1] < 2:
            # There are no finite differences for a
            # one-timestep sequence.
            return continuous_latents.new_zeros(())

        # Melodic intervals.
        delta_x = (
            pitches[:, 1:].float()
            - pitches[:, :-1].float()
        )

        # Continuous encoder intervals.
        delta_encoder = (
            continuous_latents[:, 1:]
            - continuous_latents[:, :-1]
        )

        # Hinge-style contour penalty:
        #
        # Same-direction intervals produce a positive
        # product and therefore tend toward zero loss.
        #
        # Opposite-direction intervals produce a
        # negative product and incur a large penalty.
        violation = (
            1.0
            - delta_x * delta_encoder
        )

        loss = violation.clamp_min(0.0).pow(2)

        if self.reduction == "sum":
            return loss.sum()

        return loss.mean()

    def forward(
        self,
        logits: torch.Tensor,
        target_pitches: torch.Tensor,
        continuous_latents: torch.Tensor,
    ) -> PianoGenieLossOutput:
        """
        Compute the complete Piano Genie objective.
        """

        reconstruction = self.reconstruction_loss(
            logits=logits,
            target_pitches=target_pitches,
        )

        margin = self.margin_loss(
            continuous_latents=continuous_latents,
        )

        contour = self.contour_loss(
            pitches=target_pitches,
            continuous_latents=continuous_latents,
        )

        total = (
            reconstruction
            + margin
            + contour
        )

        return PianoGenieLossOutput(
            total=total,
            reconstruction=reconstruction,
            margin=margin,
            contour=contour,
        )