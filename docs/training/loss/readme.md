## References

Here you can find docs for the loss functions.

## Acoustic model loss

### [Bin loss](./bin_loss.md)

Binary cross-entropy loss for hard and soft attention.

### [Forward sum loss](./forward_sum_loss.md)

Computes the forward sum loss for sequence-to-sequence models with attention.

### [Fast Speech 2 loss](./fast_speech_2_loss_gen.md)

FastSpeech 2 Loss module

## Voicoder Univnet loss

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
