import torch
import torch.nn as nn

from typing import Tuple

from model.constants import LEAKY_RELU_SLOPE
from model.basenn import BaseNNModule

from helpers.tools import get_device

from .mas import b_mas


class Aligner(BaseNNModule):
    r"""
    Aligner class represents a PyTorch module responsible for alignment tasks
    in a sequence-to-sequence model. It uses convolutional layers combined with
    LeakyReLU activation functions to project inputs to a hidden representation.
    The class utilizes both softmax and log-softmax to calculate softmax
    along dimension 3.

    Args:
        d_enc_in (int): Number of channels in the input for the encoder.
        d_dec_in (int): Number of channels in the input for the decoder.
        d_hidden (int): Number of channels in the output (hidden layers).
        kernel_size_enc (int, optional): Size of the convolving kernel for encoder, default is 3.
        kernel_size_dec (int, optional): Size of the convolving kernel for decoder, default is 7.
        temperature (float, optional): The temperature value applied in Gaussian isotropic
            attention mechanism, default is 0.0005.
        leaky_relu_slope (float, optional): Controls the angle of the negative slope of
            LeakyReLU activation, default is LEAKY_RELU_SLOPE.
        device (torch.device): The device to which the model should be moved. Defaults `get_device()`

    """

    def __init__(
        self,
        d_enc_in: int,
        d_dec_in: int,
        d_hidden: int,
        kernel_size_enc: int = 3,
        kernel_size_dec: int = 7,
        temperature: float = 0.0005,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
        device: torch.device = get_device(),
    ):
        super().__init__(device=device)
        self.temperature = temperature

        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            nn.Conv1d(
                d_enc_in,
                d_hidden,
                kernel_size=kernel_size_enc,
                padding=kernel_size_enc // 2,
                device=self.device,
            ),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size_enc,
                padding=kernel_size_enc // 2,
                device=self.device,
            ),
            nn.LeakyReLU(leaky_relu_slope),
        )

        self.query_proj = nn.Sequential(
            nn.Conv1d(
                d_dec_in,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
                device=self.device,
            ),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
                device=self.device,
            ),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv1d(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
                device=self.device,
            ),
            nn.LeakyReLU(leaky_relu_slope),
        )

    def binarize_attention_parallel(
        self, attn: torch.Tensor, in_lens: torch.Tensor, out_lens: torch.Tensor
    ) -> torch.Tensor:
        r"""
        For training purposes only! Binarizes attention with MAS.
        Binarizes the attention tensor using Maximum Attention Strategy (MAS).

        This process is applied for training purposes only and the resulting
        binarized attention tensor will no longer receive a gradient in the
        backpropagation process.

        Args:
            attn (Tensor): The attention tensor. Must be of shape (B, 1, max_mel_len, max_text_len),
                where B represents the batch size, max_mel_len represents the maximum length
                of the mel spectrogram, and max_text_len represents the maximum length of the text.
            in_lens (Tensor): A 1D tensor of shape (B,) that contains the input sequence lengths,
                which likely corresponds to text sequence lengths.
            out_lens (Tensor): A 1D tensor of shape (B,) that contains the output sequence lengths,
                which likely corresponds to mel spectrogram lengths.

        Returns:
            Tensor: The binarized attention tensor. The output tensor has the same shape as the input `attn` tensor.
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(
                attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1
            )
        return torch.from_numpy(attn_out).to(attn.device)

    def forward(
        self,
        enc_in: torch.Tensor,
        dec_in: torch.Tensor,
        enc_len: torch.Tensor,
        dec_len: torch.Tensor,
        enc_mask: torch.Tensor,
        attn_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Performs the forward pass through the Aligner module.

        Args:
            enc_in (Tensor): The text encoder outputs.
                Must be of shape (B, C_1, T_1), where B is the batch size, C_1 the number of
                channels in encoder inputs,
                and T_1 the sequence length of encoder inputs.
            dec_in (Tensor): The data to align with encoder outputs.
                Must be of shape (B, C_2, T_2), where C_2 is the number of channels in decoder inputs,
                and T_2 the sequence length of decoder inputs.
            enc_len (Tensor): 1D tensor representing the lengths of each sequence in the batch in `enc_in`.
            dec_len (Tensor): 1D tensor representing the lengths of each sequence in the batch in `dec_in`.
            enc_mask (Tensor): Binary mask tensor used to avoid attention to certain timesteps.
            attn_prior (Tensor): Previous attention values for attention calculation.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Returns a tuple of Tensors representing the log-probability, soft attention, hard attention, and hard attention duration.
        """
        queries = dec_in
        keys = enc_in
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (
            queries_enc[:, :, :, None] - keys_enc[:, :, None]
        ) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape}
            # mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(
                attn_prior.permute((0, 2, 1))[:, None] + 1e-8
            )
            # print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")"""

        attn_logprob = attn.clone()

        if enc_mask is not None:
            attn.masked_fill(enc_mask.unsqueeze(1).unsqueeze(1), -float("inf"))

        attn_soft = self.softmax(attn)  # softmax along T2
        attn_hard = self.binarize_attention_parallel(attn_soft, enc_len, dec_len)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        return attn_logprob, attn_soft, attn_hard, attn_hard_dur
