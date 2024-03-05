from dataclasses import dataclass


# TODO: DEPRECATED!
@dataclass
class PostNetConfig:
    p_dropout: float
    postnet_embedding_dim: int
    postnet_kernel_size: int
    postnet_n_convolutions: int

postnet_expetimental = PostNetConfig(
    p_dropout=0.1,
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=3,
)

# TODO: DEPRECATED!
@dataclass
class DiffusionConfig:
    # model parameters
    model: str
    n_mel_channels: int
    multi_speaker: bool
    # denoiser parameters
    residual_channels: int
    residual_layers: int
    denoiser_dropout: float
    noise_schedule_naive: str
    timesteps: int
    shallow_timesteps: int
    min_beta: float
    max_beta: float
    s: float
    pe_scale: int
    keep_bins: int
    # trainsformer params
    encoder_hidden: int
    decoder_hidden: int
    speaker_embed_dim: int
    # loss params
    noise_loss: str


diff_en = DiffusionConfig(
    # model parameters
    model="shallow",
    n_mel_channels=100,
    multi_speaker=True,
    # denoiser parameters
    # residual_channels=256,
    # residual_channels=384,
    residual_channels=100,
    residual_layers=20,
    denoiser_dropout=0.2,
    noise_schedule_naive="vpsde",
    timesteps=10,
    shallow_timesteps=1,
    min_beta=0.1,
    max_beta=40,
    s=0.008,
    keep_bins=80,
    pe_scale=1000,
    # trainsformer params
    # encoder_hidden=100,
    encoder_hidden=512,
    decoder_hidden=512,
    # Speaker_emb + lang_emb
    speaker_embed_dim=1025,
    # loss params
    noise_loss="l1",
)

diff_multi = DiffusionConfig(
    # model parameters
    model="shallow",
    n_mel_channels=100,
    multi_speaker=True,
    # denoiser parameters
    # residual_channels=256,
    residual_channels=100,
    residual_layers=20,
    denoiser_dropout=0.2,
    noise_schedule_naive="vpsde",
    timesteps=10,
    shallow_timesteps=1,
    min_beta=0.1,
    max_beta=40,
    s=0.008,
    pe_scale=1000,
    keep_bins=80,
    # trainsformer params
    encoder_hidden=512,
    decoder_hidden=512,
    # Speaker_emb + lang_emb
    speaker_embed_dim=1280,
    # loss params
    noise_loss="l1",
)
