## References

[UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889v1)

> Most neural vocoders employ band-limited mel-spectrograms to generate waveforms. If full-band spectral features are used as the input, the vocoder can be provided with as much acoustic information as possible. However, in some models employing full-band mel-spectrograms, an over-smoothing problem occurs as part of which non-sharp spectrograms are generated. To address this problem, we propose UnivNet, a neural vocoder that synthesizes high-fidelity waveforms in real time. Inspired by works in the field of voice activity detection, we added a multi-resolution spectrogram discriminator that employs multiple linear spectrogram magnitudes computed using various parameter sets. Using full-band mel-spectrograms as input, we expect to generate high-resolution signals by adding a discriminator that employs spectrograms of multiple resolutions as the input. In an evaluation on a dataset containing information on hundreds of speakers, UnivNet obtained the best objective and subjective results among competing models for both seen and unseen speakers. These results, including the best subjective score for text-to-speech, demonstrate the potential for fast adaptation to new speakers without a need for training from scratch. 

### [Univnet](./univnet.md)

The core module for the training.

### [Generator](generator.md)

UnivNet Generator.
Initializes the Generator module.

### [Traced Generator](traced_generator.md)

A traced version of the UnivNet Generator class that can be used for faster inference.

### [Kernel Predictor](kernel_predictor.md)

KernelPredictor is a class that predicts the kernel size for the convolutional layers in the UnivNet model.
The kernels of the LVC layers are predicted using a kernel predictor that takes the log-mel-spectrogram as the input.

### [LVC Block](lvc_block.md)

The location-variable convolutions block.
To efficiently capture the local information of the condition, location-variable convolution (LVC) obtained better sound quality and speed while maintaining the model size.

### [Discriminator](discriminator.md)

Discriminator for the UnuvNet vocoder.

This class implements a discriminator that consists of a `MultiResolutionDiscriminator` and a `MultiPeriodDiscriminator`.

### [MultiPeriodDiscriminator](multi_period_discriminator.md)

`MultiPeriodDiscriminator` is a class that implements a multi-period discriminator network for the UnivNet vocoder.

### [DiscriminatorP](discriminator_p.md)

`DiscriminatorP` is a class that implements a discriminator network for the UnivNet vocoder.

### [DiscriminatorR](discriminator_r.md)

A class representing the Residual Discriminator network for a UnivNet vocoder.

### [MultiResolutionDiscriminator](multi_resolution_discriminator.md)

Multi-resolution discriminator for the UnivNet vocoder.
