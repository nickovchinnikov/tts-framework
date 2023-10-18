from typing import Tuple

from lightning.pytorch import LightningModule
import torch

from model.config import AcousticModelConfigType

from .embedding import Embedding
from .variance_predictor import VariancePredictor


class PitchAdaptor(LightningModule):
    r"""
    The PitchAdaptor class is a pitch adaptor network in the model.

    It has methods to get the pitch embeddings for train and test,
    and to add pitch during training and in inference.

    Args:
        model_config (AcousticModelConfigType): The model configuration.
    """

    def __init__(
        self,
        model_config: AcousticModelConfigType,
    ):
        super().__init__()
        self.pitch_predictor = VariancePredictor(
            channels_in=model_config.encoder.n_hidden,
            channels=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            p_dropout=model_config.variance_adaptor.p_dropout,
        )

        self.n_bins = model_config.variance_adaptor.n_bins
        self.pitch_embedding = Embedding(self.n_bins, model_config.encoder.n_hidden)

    def get_pitch_bins(self, pitch_min: float, pitch_max: float) -> torch.Tensor:
        r"""
        Get the pitch bins.

        Args:
            pitch_min (float): The minimum pitch value.
            pitch_max (float): The maximum pitch value.

        Returns:
            torch.Tensor: The tensor containing pitch bins.
        """
        result = torch.linspace(
            pitch_min,
            pitch_max,
            self.n_bins - 1,
            device=self.device,
        )
        return result

    def get_pitch_embedding_train(
        self,
        x: torch.Tensor,
        pitch_min: float,
        pitch_max: float,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute pitch prediction and embeddings during training.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target or ground truth tensor for pitch values.
            mask (torch.Tensor): The mask tensor indicating valid values in the `target` tensor.

        Returns:
            Tuple of Tensors: The pitch prediction, true pitch embedding and predicted pitch embedding.
        """
        pitch_bins = self.get_pitch_bins(pitch_min, pitch_max)

        prediction = self.pitch_predictor(x, mask)
        embedding_true = self.pitch_embedding(torch.bucketize(target, pitch_bins))
        embedding_pred = self.pitch_embedding(torch.bucketize(prediction, pitch_bins))
        return prediction, embedding_true, embedding_pred

    def get_pitch_embedding(
        self,
        x: torch.Tensor,
        pitch_min: float,
        pitch_max: float,
        mask: torch.Tensor,
        control: float,
    ) -> torch.Tensor:
        r"""
        Compute pitch embeddings during inference.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor indicating the valid entries in input.
            control (float): Scaling factor to control the effects of pitch.

        Returns:
            torch.Tensor: The tensor containing pitch embeddings.
        """
        pitch_bins = self.get_pitch_bins(pitch_min, pitch_max)

        prediction = self.pitch_predictor(x, mask)
        prediction = prediction * control
        embedding = self.pitch_embedding(torch.bucketize(prediction, pitch_bins))
        return embedding

    def add_pitch_train(
        self,
        x: torch.Tensor,
        pitch_min: float,
        pitch_max: float,
        pitch_target: torch.Tensor,
        src_mask: torch.Tensor,
        use_ground_truth: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Apply pitch embeddings to the input tensor during training.

        Args:
            x (torch.Tensor): The input tensor.
            pitch_min (float): The minimum pitch value.
            pitch_max (float): The maximum pitch value.
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
        ) = self.get_pitch_embedding_train(
            x, pitch_min, pitch_max, pitch_target, src_mask
        )
        if use_ground_truth:
            x = x + pitch_embedding_true
        else:
            x = x + pitch_embedding_pred
        return x, pitch_prediction, pitch_embedding_true, pitch_embedding_pred

    def add_pitch(
        self,
        x: torch.Tensor,
        pitch_min: float,
        pitch_max: float,
        src_mask: torch.Tensor,
        control: float,
    ) -> torch.Tensor:
        r"""
        Apply pitch embeddings to the input tensor during inference.

        Args:
            x (torch.Tensor): The input tensor.
            pitch_min (float): The minimum pitch value.
            pitch_max (float): The maximum pitch value.
            src_mask (torch.Tensor): The mask tensor indicating the valid entries in input.
            control (float): Scaling factor to control the effects of pitch.

        Returns:
            torch.Tensor: The tensor resulting from addition of pitch embeddings and input tensor.
        """
        pitch_embedding_pred = self.get_pitch_embedding(
            x, pitch_min, pitch_max, src_mask, control=control
        )
        x = x + pitch_embedding_pred
        return x
