import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class Embedding(Module):
    r"""Class represents a simple embedding layer but without any learning of the embeddings.
    The embeddings are initialized with random values and kept static throughout training (They are parameters, not model's state).

    Args:
        num_embeddings (int): Size of the dictionary of embeddings, typically size of the vocabulary.
        embedding_dim (int): The size of each embedding vector.

    Returns:
        torch.Tensor: An output tensor resulting from the lookup operation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        r"""Forward propagation for the Embedding implementation.

        Args:
            idx (torch.Tensor): A tensor containing the indices of the embeddings to be accessed.

        Returns:
            torch.Tensor: An output tensor resulting from the lookup operation.
        """
        return F.embedding(idx, self.embeddings)
