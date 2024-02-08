from typing import List, Tuple

import torch
from torch.nn import Module

from models.config import AcousticModelConfigType
from models.helpers import tools

from .variance_predictor import VariancePredictor


class LengthAdaptor(Module):
    r"""The LengthAdaptor module is used to adjust the duration of phonemes. Used in Tacotron 2 model.
    It contains a dedicated duration predictor and methods to upsample the input features to match predicted durations.

    Args:
        model_config (AcousticModelConfigType): The model configuration object containing model parameters.
    """

    def __init__(
        self,
        model_config: AcousticModelConfigType,
    ):
        super().__init__()
        # Initialize the duration predictor
        self.duration_predictor = VariancePredictor(
            channels_in=model_config.encoder.n_hidden,
            channels=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            p_dropout=model_config.variance_adaptor.p_dropout,
        )

    def length_regulate(
        self, x: torch.Tensor, duration: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Regulates the length of the input tensor using the duration tensor.

        Args:
            x (torch.Tensor): The input tensor.
            duration (torch.Tensor): The tensor containing duration for each time step in x.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The regulated output tensor and the tensor containing the length of each sequence in the batch.
        """
        output = torch.jit.annotate(List[torch.Tensor], [])
        mel_len = torch.jit.annotate(List[int], [])
        max_len = 0
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            if expanded.shape[0] > max_len:
                max_len = expanded.shape[0]
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        output = tools.pad(output, max_len)
        return output, torch.tensor(mel_len, dtype=torch.int64)

    def expand(self, batch: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        r"""Expands the input tensor based on the predicted values.

        Args:
            batch (torch.Tensor): The input tensor.
            predicted (torch.Tensor): The tensor containing predicted expansion factors.

        Returns:
            torch.Tensor: The expanded tensor.
        """
        out = torch.jit.annotate(List[torch.Tensor], [])
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        return torch.cat(out, 0)

    def upsample_train(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor,
        duration_target: torch.Tensor,
        embeddings: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Upsamples the input tensor during training using ground truth durations.

        Args:
            x (torch.Tensor): The input tensor.
            x_res (torch.Tensor): Another input tensor for duration prediction.
            duration_target (torch.Tensor): The ground truth durations tensor.
            embeddings (torch.Tensor): The tensor containing phoneme embeddings.
            src_mask (torch.Tensor): The mask tensor indicating valid entries in x and x_res.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The upsampled x, log duration prediction, and upsampled embeddings.
        """
        x_res = x_res.detach()
        log_duration_prediction = self.duration_predictor(
            x_res, src_mask,
        )  # type: torch.Tensor
        x, _ = self.length_regulate(x, duration_target)
        embeddings, _ = self.length_regulate(embeddings, duration_target)
        return x, log_duration_prediction, embeddings

    def upsample(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor,
        src_mask: torch.Tensor,
        embeddings: torch.Tensor,
        control: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Upsamples the input tensor during inference.

        Args:
            x (torch.Tensor): The input tensor.
            x_res (torch.Tensor): Another input tensor for duration prediction.
            src_mask (torch.Tensor): The mask tensor indicating valid entries in x and x_res.
            embeddings (torch.Tensor): The tensor containing phoneme embeddings.
            control (float): A control parameter for pitch regulation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The upsampled x, approximated duration, and upsampled embeddings.
        """
        log_duration_prediction = self.duration_predictor(
            x_res,
            src_mask,
        )
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * control),
            min=0,
        )
        x, _ = self.length_regulate(x, duration_rounded)
        embeddings, _ = self.length_regulate(embeddings, duration_rounded)
        return x, duration_rounded, embeddings
