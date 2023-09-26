import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basenn import BaseNNModule

from helpers import tools


class Embedding(BaseNNModule):
    r"""
    This class represents a simple embedding layer but without any learning of the embeddings.
    The embeddings are initialized with random values and kept static throughout training (They are parameters, not model's state).

    Args:
        num_embeddings (int): Size of the dictionary of embeddings, typically size of the vocabulary.
        embedding_dim (int): The size of each embedding vector.
        device (torch.device): The device to which the model should be moved. Defaults `get_device()`

    Returns:
        torch.Tensor: An output tensor resulting from the lookup operation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = tools.get_device(),
    ):
        super().__init__(device=device)
        self.embeddings = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim, device=self.device)
        ).to(self.device)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        r"""
        Forward propagation for the Embedding implementation.

        Args:
            idx (torch.Tensor): A tensor containing the indices of the embeddings to be accessed.

        Returns:
            torch.Tensor: An output tensor resulting from the lookup operation.
        """
        x = F.embedding(idx, self.embeddings)
        return x
