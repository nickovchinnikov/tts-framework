from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.nn import functional as F

####################################################################################################
# TODO: this code from coqui-ai, don't see this in the original Mozilla implementation of Tacotron2
# DEPRECATED ?
####################################################################################################

class CapacitronVAE(nn.Module):
    r"""Effective Use of Variational Embedding Capacity for prosody transfer.

    Args:
        num_mel (int): Number of mel filters.
        capacitron_VAE_embedding_dim (int): Dimension of the VAE embedding.
        encoder_output_dim (int, optional): Dimension of the encoder output. Defaults to 256.
        reference_encoder_out_dim (int, optional): Dimension of the reference encoder output. Defaults to 128.
        speaker_embedding_dim (int, optional): Dimension of the speaker embedding. Defaults to None.
        text_summary_embedding_dim (int, optional): Dimension of the text summary embedding. Defaults to None.

    See https://arxiv.org/abs/1906.03402
    """

    def __init__(
        self,
        num_mel: int,
        capacitron_VAE_embedding_dim: int,
        encoder_output_dim: int=256,
        reference_encoder_out_dim: int=128,
        speaker_embedding_dim: Optional[int]=None,
        text_summary_embedding_dim: Optional[int]=None,
    ):
        super().__init__()
        # Init distributions
        self.prior_distribution = MVN(
            torch.zeros(capacitron_VAE_embedding_dim), torch.eye(capacitron_VAE_embedding_dim),
        )
        self.approximate_posterior_distribution = None
        # define output ReferenceEncoder dim to the capacitron_VAE_embedding_dim
        self.encoder = ReferenceEncoder(num_mel, out_dim=reference_encoder_out_dim)

        # Init beta, the lagrange-like term for the KL distribution
        self.beta = torch.nn.Parameter(torch.log(torch.exp(torch.Tensor([1.0])) - 1), requires_grad=True)
        mlp_input_dimension = reference_encoder_out_dim

        if text_summary_embedding_dim is not None:
            self.text_summary_net = TextSummary(text_summary_embedding_dim, encoder_output_dim=encoder_output_dim)
            mlp_input_dimension += text_summary_embedding_dim
        if speaker_embedding_dim is not None:
            # TODO: Test a multispeaker model!
            mlp_input_dimension += speaker_embedding_dim
        self.post_encoder_mlp = PostEncoderMLP(mlp_input_dimension, capacitron_VAE_embedding_dim)

    def forward(
        self,
        reference_mel_info: Optional[Tuple[Tensor, Tensor]]=None,
        text_info: Optional[Tuple[Tensor, Tensor]]=None,
        speaker_embedding: Optional[Tensor]=None,
    ) -> Tuple[Tensor, MVN | None, MVN, Tensor]:
        r"""Forward pass of the CapacitronVAE.

        Args:
        reference_mel_info (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Tuple containing reference mels and their lengths. Defaults to None.
        text_info (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Tuple containing text inputs and their lengths. Defaults to None.
        speaker_embedding (Optional[torch.Tensor], optional): Speaker embedding tensor. Defaults to None.

        Returns:
        Tuple[torch.Tensor, MVN, MVN, torch.Tensor]: Returns the VAE embedding, the approximate posterior distribution, the prior distribution, and the beta parameter.
        """
        # Use reference
        if reference_mel_info is not None:
            reference_mels = reference_mel_info[0]  # [batch_size, num_frames, num_mels]
            mel_lengths = reference_mel_info[1]  # [batch_size]
            enc_out = self.encoder(reference_mels, mel_lengths)

            # concat speaker_embedding and/or text summary embedding
            if text_info is not None:
                text_inputs = text_info[0]  # [batch_size, num_characters, num_embedding]
                input_lengths = text_info[1]
                text_summary_out = self.text_summary_net(text_inputs, input_lengths).to(reference_mels.device)
                enc_out = torch.cat([enc_out, text_summary_out], dim=-1)
            if speaker_embedding is not None:
                speaker_embedding = torch.squeeze(speaker_embedding)
                enc_out = torch.cat([enc_out, speaker_embedding], dim=-1)

            # Feed the output of the ref encoder and information about text/speaker into
            # an MLP to produce the parameteres for the approximate poterior distributions
            mu, sigma = self.post_encoder_mlp(enc_out)
            # convert to cpu because prior_distribution was created on cpu
            mu = mu.cpu()
            sigma = sigma.cpu()

            # Sample from the posterior: z ~ q(z|x)
            self.approximate_posterior_distribution = MVN(mu, torch.diag_embed(sigma))
            VAE_embedding = self.approximate_posterior_distribution.rsample()
        # Infer from the model, bypasses encoding
        else:
            # Sample from the prior: z ~ p(z)
            VAE_embedding = self.prior_distribution.sample().unsqueeze(0)

        # reshape to [batch_size, 1, capacitron_VAE_embedding_dim]
        return VAE_embedding.unsqueeze(1), self.approximate_posterior_distribution, self.prior_distribution, self.beta


class ReferenceEncoder(nn.Module):
    r"""NN module creating a fixed size prosody embedding from a spectrogram.

    Args:
        num_mel (int): Number of mel filters. Shape: [batch_size, num_spec_frames, num_mel]
        out_dim (int): Output dimension. Shape: [batch_size, embedding_dim]

    Attributes:
        num_mel (int): Number of mel filters.
        convs (nn.ModuleList): List of convolutional layers.
        bns (nn.ModuleList): List of batch normalization layers.
        recurrence (nn.LSTM): LSTM layer.
    """

    def __init__(self, num_mel: int, out_dim: int):
        super().__init__()
        self.num_mel = num_mel
        filters = [1, 32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.training = False
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 2, num_layers)
        self.recurrence = nn.LSTM(
            input_size=filters[-1] * post_conv_height, hidden_size=out_dim, batch_first=True, bidirectional=False,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        r"""Forward pass of the ReferenceEncoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, num_spec_frames, num_mel).
            input_lengths (torch.Tensor): Tensor containing the lengths of the inputs.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embedding_dim).
        """
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)  # [batch_size, num_channels==1, num_frames, num_mel]
        valid_lengths = input_lengths.float()  # [batch_size]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

            # Create the post conv width mask based on the valid lengths of the output of the convolution.
            # The valid lengths for the output of a convolution on varying length inputs is
            # ceil(input_length/stride) + 1 for stride=3 and padding=2
            # For example (kernel_size=3, stride=2, padding=2):
            # 0 0 x x x x x 0 0 -> Input = 5, 0 is zero padding, x is valid values coming from padding=2 in conv2d
            # _____
            #   x _____
            #       x _____
            #           x  ____
            #               x
            # x x x x -> Output valid length = 4
            # Since every example in te batch is zero padded and therefore have separate valid_lengths,
            # we need to mask off all the values AFTER the valid length for each example in the batch.
            # Otherwise, the convolutions create noise and a lot of not real information
            valid_lengths = (valid_lengths / 2).float()
            valid_lengths = torch.ceil(valid_lengths).to(dtype=torch.int64) + 1  # 2 is stride -- size: [batch_size]
            post_conv_max_width = x.size(2)

            mask = torch.arange(post_conv_max_width).to(inputs.device).expand(
                len(valid_lengths), post_conv_max_width,
            ) < valid_lengths.unsqueeze(1)
            mask = mask.expand(1, 1, -1, -1).transpose(2, 0).transpose(-1, 2)  # [batch_size, 1, post_conv_max_width, 1]
            x = x * mask

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]

        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]

        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        post_conv_input_lengths = valid_lengths
        packed_seqs = nn.utils.rnn.pack_padded_sequence(
            x, post_conv_input_lengths, batch_first=True, enforce_sorted=False,
        )  # dynamic rnn sequence padding
        self.recurrence.flatten_parameters()
        _, (ht, _) = self.recurrence(packed_seqs)
        last_output = ht[-1]

        return last_output.to(inputs.device)  # [B, 128]

    @staticmethod
    def calculate_post_conv_height(
        height: int,
        kernel_size: int,
        stride: int,
        pad: int,
        n_convs: int,
    ) -> int:
        r"""Calculate the height of the output after n convolutions.

        Args:
            height (int): Initial height.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            pad (int): Padding of the convolution.
            n_convs (int): Number of convolutions.

        Returns:
            int: The height of the output after n convolutions.
        """
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height


class TextSummary(nn.Module):
    r"""NN module for creating a fixed size text summary embedding from a text encoder output.

    Args:
        embedding_dim (int): The dimension of the output embedding.
        encoder_output_dim (int): The dimension of the text encoder output.

    Attributes:
        lstm (nn.LSTM): LSTM layer.
    """

    def __init__(self, embedding_dim: int, encoder_output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            encoder_output_dim,  # text embedding dimension from the text encoder
            embedding_dim,  # fixed length output summary the lstm creates from the input
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        r"""Forward pass of the TextSummary.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, encoder_output_dim).
            input_lengths (torch.Tensor): Tensor containing the lengths of the inputs.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embedding_dim).
        """
        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        packed_seqs = nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths, batch_first=True, enforce_sorted=False,
        )  # dynamic rnn sequence padding
        self.lstm.flatten_parameters()
        _, (ht, _) = self.lstm(packed_seqs)
        last_output = ht[-1]
        return last_output


class PostEncoderMLP(nn.Module):
    r"""NN module for creating a mean and variance from a given input.

    Args:
        input_size (int): The size of the input tensor.
        hidden_size (int): The size of the hidden layer.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        net (nn.Sequential): Sequential model containing the layers of the network.
        softplus (nn.Softplus): Softplus activation function.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        modules = [
            nn.Linear(input_size, hidden_size),  # Hidden Layer
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * 2),
        ]  # Output layer twice the size for mean and variance
        self.net = nn.Sequential(*modules)
        self.softplus = nn.Softplus()

    def forward(self, _input: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward pass of the PostEncoderMLP.

        Args:
            _input (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors containing the mean and variance of shape (batch_size, hidden_size).
        """
        mlp_output = self.net(_input)
        # The mean parameter is unconstrained
        mu = mlp_output[:, : self.hidden_size]
        # The standard deviation must be positive. Parameterise with a softplus
        sigma = self.softplus(mlp_output[:, self.hidden_size :])
        return mu, sigma
