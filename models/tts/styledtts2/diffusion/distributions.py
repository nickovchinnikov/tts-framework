from math import atan, pi

import torch
from torch import Tensor

""" Distributions """


class Distribution:
    r"""Base class for all distributions."""

    def __call__(self, num_samples: int) -> Tensor:
        r"""Generate a number of samples from the distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError


class LogNormalDistribution(Distribution):
    r"""Log-normal distribution."""

    def __init__(self, mean: float, std: float):
        r"""Initialize the distribution with a mean and standard deviation.

        Args:
            mean (float): The mean of the log-normal distribution.
            std (float): The standard deviation of the log-normal distribution.
        """
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples: int,
    ) -> Tensor:
        r"""Generate a number of samples from the log-normal distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            Tensor: A tensor of samples from the log-normal distribution.
        """
        normal = self.mean + self.std * torch.randn((num_samples,))
        return normal.exp()


class UniformDistribution(Distribution):
    r"""Uniform distribution."""

    def __call__(self, num_samples: int):
        r"""Generate a number of samples from the uniform distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            Tensor: A tensor of samples from the uniform distribution.
        """
        return torch.rand(num_samples)


class VKDistribution(Distribution):
    r"""VK distribution."""

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = float("inf"),
        sigma_data: float = 1.0,
    ):
        r"""Initialize the distribution with a minimum value, maximum value, and sigma data.

        Args:
            min_value (float): The minimum value for the inverse CDF. Defaults to 0.0.
            max_value (float): The maximum value for the inverse CDF. Defaults to infinity.
            sigma_data (float): The sigma data of the VK distribution. Defaults to 1.0.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.sigma_data = sigma_data

    def __call__(
        self, num_samples: int,
    ) -> Tensor:
        r"""Generate a number of samples from the VK distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            Tensor: A tensor of samples from the VK distribution.
        """
        sigma_data = self.sigma_data
        min_cdf = atan(self.min_value / sigma_data) * 2 / pi
        max_cdf = atan(self.max_value / sigma_data) * 2 / pi
        u = (max_cdf - min_cdf) * torch.randn((num_samples,)) + min_cdf
        return torch.tan(u * pi / 2) * sigma_data
