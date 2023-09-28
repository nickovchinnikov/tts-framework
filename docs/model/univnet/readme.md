## References

[UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889v1)

> Most neural vocoders employ band-limited mel-spectrograms to generate waveforms. If full-band spectral features are used as the input, the vocoder can be provided with as much acoustic information as possible. However, in some models employing full-band mel-spectrograms, an over-smoothing problem occurs as part of which non-sharp spectrograms are generated. To address this problem, we propose UnivNet, a neural vocoder that synthesizes high-fidelity waveforms in real time. Inspired by works in the field of voice activity detection, we added a multi-resolution spectrogram discriminator that employs multiple linear spectrogram magnitudes computed using various parameter sets. Using full-band mel-spectrograms as input, we expect to generate high-resolution signals by adding a discriminator that employs spectrograms of multiple resolutions as the input. In an evaluation on a dataset containing information on hundreds of speakers, UnivNet obtained the best objective and subjective results among competing models for both seen and unseen speakers. These results, including the best subjective score for text-to-speech, demonstrate the potential for fast adaptation to new speakers without a need for training from scratch. 

### [UnivNet Generator](generator.md)

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

### [STFT](stft.md)

Perform STFT and convert to magnitude spectrogram.
STFT stands for Short-Time Fourier Transform. It is a signal processing technique that is used to analyze the frequency content of a signal over time. The STFT is computed by dividing a long signal into shorter segments, and then computing the Fourier transform of each segment. This results in a time-frequency representation of the signal, where the frequency content of the signal is shown as a function of time.

### [Spectral Convergence Loss](spectral_convergence_loss.md)

Spectral convergence loss is a measure of the similarity between two magnitude spectrograms.

The spectral convergence loss is calculated as the Frobenius norm of the difference between the predicted and groundtruth magnitude spectrograms, divided by the Frobenius norm of the groundtruth magnitude spectrogram. The Frobenius norm is a matrix norm that is equivalent to the square root of the sum of the squared elements of a matrix.

The spectral convergence loss is a useful metric for evaluating the quality of a predicted signal, as it measures the degree to which the predicted signal matches the groundtruth signal in terms of its spectral content. A lower spectral convergence loss indicates a better match between the predicted and groundtruth signals.

### [Log STFT Magnitude Loss](log_stft_magnitude_loss.md)

Log STFT magnitude loss is a loss function that is commonly used in speech and audio signal processing tasks, such as speech enhancement and source separation. It is a modification of the spectral convergence loss, which measures the similarity between two magnitude spectrograms.

The log STFT magnitude loss is calculated as the mean squared error between the logarithm of the predicted and groundtruth magnitude spectrograms. The logarithm is applied to the magnitude spectrograms to convert them to a decibel scale, which is more perceptually meaningful than the linear scale. The mean squared error is used to penalize large errors between the predicted and groundtruth spectrograms.


### [STFT Loss](stft_loss.md)

STFT loss is a combination of two loss functions: the spectral convergence loss and the log STFT magnitude loss.

The spectral convergence loss measures the similarity between two magnitude spectrograms, while the log STFT magnitude loss measures the similarity between two logarithmically-scaled magnitude spectrograms. The logarithm is applied to the magnitude spectrograms to convert them to a decibel scale, which is more perceptually meaningful than the linear scale.

The STFT loss is a useful metric for evaluating the quality of a predicted signal, as it measures the degree to which the predicted signal matches the groundtruth signal in terms of its spectral content on both a linear and decibel scale. A lower STFT loss indicates a better match between the predicted and groundtruth signals.

### [Multi Resolution STFT Loss](multi_resolution_stft_loss.md)

The Multi resolution STFT loss module is a PyTorch module that computes the spectral convergence and log STFT magnitude losses for a predicted signal and a groundtruth signal at multiple resolutions. The module is designed for speech and audio signal processing tasks, such as speech enhancement and source separation.
