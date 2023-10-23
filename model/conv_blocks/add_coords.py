import torch
from torch.nn import Module


class AddCoords(Module):
    r"""AddCoords is a PyTorch module that adds additional channels to the input tensor containing the relative
    (normalized to `[-1, 1]`) coordinates of each input element along the specified number of dimensions (`rank`).
    Essentially, it adds spatial context information to the tensor.

    Typically, these inputs are feature maps coming from some CNN, where the spatial organization of the input
    matters (such as an image or speech signal).

    This additional spatial context allows subsequent layers (such as convolutions) to learn position-dependent
    features. For example, in tasks where the absolute position of features matters (such as denoising and
    segmentation tasks), it helps the model to know where (in terms of relative position) the features are.

    Args:
        rank (int): The dimensionality of the input tensor. That is to say, this tells us how many dimensions the
                    input tensor's spatial context has. It's assumed to be 1, 2, or 3 corresponding to some 1D, 2D,
                    or 3D data (like an image).

        with_r (bool): Boolean indicating whether to add an extra radial distance channel or not. If True, an extra
                       channel is appended, which measures the Euclidean (L2) distance from the center of the image.
                       This might be useful when the proximity to the center of the image is important to the task.
    """

    def __init__(self, rank: int, with_r: bool = False):
        super().__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the AddCoords module. Depending on the rank of the tensor, it adds one or more new channels
        with relative coordinate values. If `with_r` is True, an extra radial channel is included.

        For example, for an image (`rank=2`), two channels would be added which contain the normalized x and y
        coordinates respectively of each pixel.

        Calling the forward method updates the original tensor `x` with the added channels.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            out (torch.Tensor): The input tensor with added coordinate and possibly radial channels.
        """
        if self.rank == 1:
            batch_size_shape, _, dim_x = x.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32, device=x.device)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            out = torch.cat([x, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, _, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32, device=x.device)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32, device=x.device)

            xx_range = torch.arange(dim_y, dtype=torch.int32, device=x.device)
            yy_range = torch.arange(dim_x, dtype=torch.int32, device=x.device)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            out = torch.cat([x, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2),
                )
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, _, dim_z, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32, device=x.device)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32, device=x.device)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32, device=x.device)

            xy_range = torch.arange(dim_y, dtype=torch.int32, device=x.device)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32, device=x.device)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32, device=x.device)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            out = torch.cat([x, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2)
                    + torch.pow(yy_channel - 0.5, 2)
                    + torch.pow(zz_channel - 0.5, 2),
                )
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out
