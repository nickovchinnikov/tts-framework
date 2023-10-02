import torch
from torch import nn
from torch.nn import functional as F


class ForwardSumLoss(nn.Module):
    r"""
    Computes the forward sum loss for sequence-to-sequence models with attention.

    Args:
        blank_logprob (float): The log probability of the blank symbol. Default: -1.

    Attributes:
        log_softmax (nn.LogSoftmax): The log softmax function.
        ctc_loss (nn.CTCLoss): The CTC loss function.
        blank_logprob (float): The log probability of the blank symbol.

    Methods:
        forward: Computes the forward sum loss for sequence-to-sequence models with attention.

    """

    def __init__(self, blank_logprob: float = -1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(
        self, attn_logprob: torch.Tensor, in_lens: torch.Tensor, out_lens: torch.Tensor
    ) -> float:
        r"""
        Computes the forward sum loss for sequence-to-sequence models with attention.

        Args:
            attn_logprob (torch.Tensor): The attention log probabilities of shape (batch_size, max_out_len, max_in_len).
            in_lens (torch.Tensor): The input lengths of shape (batch_size,).
            out_lens (torch.Tensor): The output lengths of shape (batch_size,).

        Returns:
            float: The forward sum loss.

        """
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0), value=self.blank_logprob
        )

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, int(key_lens[bid]) + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : int(query_lens[bid]), :, : int(key_lens[bid]) + 1
            ]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss
