import json
from pathlib import Path
from typing import Tuple

import torch
from torch.nn import Module

from models.config import AcousticModelConfigType

from .embedding import Embedding
from .variance_predictor import VariancePredictor


class PitchAdaptor(Module):
    r"""DEPRECATED: The PitchAdaptor class is a pitch adaptor network in the model.

    It has methods to get the pitch embeddings for train and test,
    and to add pitch during training and in inference.

    Args:
        model_config (AcousticModelConfigType): The model configuration.
        data_path (str): The path to data.

    """

    def __init__(
        self,
        model_config: AcousticModelConfigType,
        data_path: str,
    ):
        super().__init__()
        self.pitch_predictor = VariancePredictor(
            channels_in=model_config.encoder.n_hidden,
            channels=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            p_dropout=model_config.variance_adaptor.p_dropout,
        )

        n_bins = model_config.variance_adaptor.n_bins

        # NOTE: I changed this logic to use the stats from the preprocessing data
        # We have pitch info for the batch insted of the whole dataset
        # Always use stats from preprocessing data, even in fine-tuning
        with open(Path(data_path) / "stats.json") as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]

        print(f"Min pitch: {pitch_min} - Max pitch: {pitch_max}")

        # TODO: change pitch_bins!
        self.register_buffer(
            "pitch_bins",
            torch.linspace(
                pitch_min,
                pitch_max,
                n_bins - 1,
            ),
        )
        self.pitch_embedding = Embedding(n_bins, model_config.encoder.n_hidden)

    def get_pitch_embedding_train(
        self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute pitch prediction and embeddings during training.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target or ground truth tensor for pitch values.
            mask (torch.Tensor): The mask tensor indicating valid values in the `target` tensor.

        Returns:
            Tuple of Tensors: The pitch prediction, true pitch embedding and predicted pitch embedding.
        """
        # Explicitly convert self.pitch_bins to a Tensor if it's not already
        pitch_bins = torch.Tensor(self.pitch_bins)

        prediction = self.pitch_predictor(x, mask)
        embedding_true = self.pitch_embedding(torch.bucketize(target, pitch_bins))
        embedding_pred = self.pitch_embedding(torch.bucketize(prediction, pitch_bins))
        return prediction, embedding_true, embedding_pred

    def get_pitch_embedding(
        self, x: torch.Tensor, mask: torch.Tensor, control: float,
    ) -> torch.Tensor:
        r"""Compute pitch embeddings during inference.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor indicating the valid entries in input.
            control (float): Scaling factor to control the effects of pitch.

        Returns:
            torch.Tensor: The tensor containing pitch embeddings.
        """
        # Explicitly convert self.pitch_bins to a Tensor if it's not already
        pitch_bins = torch.Tensor(self.pitch_bins)

        prediction = self.pitch_predictor(x, mask)
        prediction = prediction * control
        return self.pitch_embedding(torch.bucketize(prediction, pitch_bins))

    def add_pitch_train(
        self,
        x: torch.Tensor,
        pitch_target: torch.Tensor,
        src_mask: torch.Tensor,
        use_ground_truth: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Apply pitch embeddings to the input tensor during training.

        Args:
            x (torch.Tensor): The input tensor.
            pitch_target (torch.Tensor): The target or ground truth tensor for pitch values.
            src_mask (torch.Tensor): The mask tensor indicating valid values in the `pitch_target`.
            use_ground_truth (bool): A flag indicating whether or not to use ground truth values for pitch.

        Returns:
            Tuple of Tensors: The tensor resulting from addition of pitch embeddings and input tensor, pitch prediction, true pitch embedding and predicted pitch embedding.
        """
        (
            pitch_prediction,
            pitch_embedding_true,
            pitch_embedding_pred,
        ) = self.get_pitch_embedding_train(x, pitch_target, src_mask)
        x = x + pitch_embedding_true if use_ground_truth else x + pitch_embedding_pred
        return x, pitch_prediction, pitch_embedding_true, pitch_embedding_pred

    def add_pitch(
        self, x: torch.Tensor, src_mask: torch.Tensor, control: float,
    ) -> torch.Tensor:
        r"""Apply pitch embeddings to the input tensor during inference.

        Args:
            x (torch.Tensor): The input tensor.
            src_mask (torch.Tensor): The mask tensor indicating the valid entries in input.
            control (float): Scaling factor to control the effects of pitch.

        Returns:
            torch.Tensor: The tensor resulting from addition of pitch embeddings and input tensor.
        """
        pitch_embedding_pred = self.get_pitch_embedding(x, src_mask, control=control)
        return x + pitch_embedding_pred
