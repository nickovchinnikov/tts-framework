## Fixes

During the development process, a huge amount of job has been done. For example, the `stft` code changed to the latest way of conversion from `complex` to `real`. For example:

```python
# Compute the short-time Fourier transform of the input waveform
x = torch.stft(
    x,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    center=False,
    return_complex=True,
    # Add window parameter to prevent the signal leak
    window=torch.ones(win_length, device=x.device),
)  # [B, F, TT, 2]

# Convert to real as the additional step
x = torch.view_as_real(x)
```

It's a tested fix, and works stable.

Minor fixes, and `TODO` that I can't fix now, but wanted to fix for the future.

The ideas behind the model and architecture are mostly the same. The training code is completely different, there is a base of dunky11 and the architecture is completely new.

## Problems

### FIXME: Step param!

I have a `step` parameter, it requires the future investigation. Maybe I need to add this param to the model step with `self.register_buffer`. It's required for the [FastSpeech2LossGen](./loss/fast_speech_2_loss_gen.md)
Possible training issue!
