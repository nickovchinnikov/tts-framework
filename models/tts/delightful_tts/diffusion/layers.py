import math

from einops import rearrange
import torch
from torch import Tensor, nn
from torch.nn import Module


class Mish(Module):
    r"""Applies the Mish activation function.

    Mish is a smooth, non-monotonic function that attempts to mitigate the
    problems of dying ReLU units in deep neural networks.
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the Mish activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Mish activation.
        """
        return x * torch.tanh(nn.functional.softplus(x))


class Upsample(Module):
    r"""Upsamples the input tensor using a transposed convolution operation.

    The transposed convolution operation effectively performs the opposite
    operation of a regular convolution, increasing the spatial dimensions
    of the input tensor.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Downsample(Module):
    r"""Downsamples the input tensor using a convolution operation.

    The convolution operation reduces the spatial dimensions of the input tensor.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Rezero(Module):
    r"""Applies a function to the input tensor and scales the result by a learnable parameter.

    The learnable parameter is initialized to zero, hence the name "Rezero".
    """

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) * self.g


class Block(Module):
    r"""Applies a sequence of operations to the input tensor.

    The operations are a convolution, a group normalization, and a Mish activation function.
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish(),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(Module):
    r"""Applies a sequence of operations to the input tensor, including a residual connection.

    The operations are two Blocks and a linear transformation of a time embedding.
    The output of these operations is added to a transformed version of the input tensor.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out),
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x: Tensor, mask: Tensor, time_emb: Tensor) -> Tensor:
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(Module):
    r"""Applies the linear attention mechanism to the input tensor.

    The input tensor is first transformed into query, key, and value tensors.
    These are used to compute an attention matrix, which is used to weight the value tensor.
    The result is transformed and returned.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)",
                            heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)

        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)

        out = rearrange(out, "b heads c (h w) -> b (heads c) h w",
                        heads=self.heads, h=h, w=w)

        return self.to_out(out)


class Residual(Module):
    r"""Applies a function to the input tensor and adds the result to the input tensor.

    This implements a residual connection, which can help to mitigate the vanishing gradient problem in deep networks.
    """

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(Module):
    r"""Applies a sinusoidal positional embedding to the input tensor.

    The positional embedding is a function of the position in the input tensor and the dimension of the embedding.
    It is designed to provide a unique, learnable representation for each position in the input tensor.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor, scale: int = 1000):
        device = x.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)

        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb
