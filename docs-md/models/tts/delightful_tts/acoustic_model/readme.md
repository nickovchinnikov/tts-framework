## References

### [Accoustic Model](acoustic_model.md)

The DelightfulTTS AcousticModel class represents a PyTorch module for an acoustic model in text-to-speech (TTS).
The acoustic model is responsible for predicting speech signals from phoneme sequences.

The model comprises multiple sub-modules including encoder, decoder and various prosody encoders and predictors.
Additionally, a pitch and length adaptor are instantiated.

### [Embedding](embedding.md)

This class represents a simple embedding layer but without any learning of the embeddings.

### [Helpers](helpers.md)

Acoustic model helpers methods

### [Variance Predictor](variance_predictor.md)

This is a Duration and Pitch predictor neural network module in PyTorch.

### [Pitch Adaptor Conv](pitch_adaptor_conv.md)

Variance Adaptor with an added 1D conv layer. Used to get pitch embeddings.

## [Energy Adaptor](energy_adaptor.md)

Variance Adaptor with an added 1D conv layer. Used to get energy embeddings.

### [Length Adaptor](length_adaptor.md)

The LengthAdaptor module is used to adjust the duration of phonemes. Used in Tacotron 2 model.

### [Phoneme Prosody Predictor](phoneme_prosody_predictor.md)

A class to define the Phoneme Prosody Predictor. 
This prosody predictor is non-parallel and is inspired by the **work of Du et al., 2021 ?**.

In linguistics, prosody (/ˈprɒsədi, ˈprɒzədi/)is the study of elements of speech that are not individual phonetic segments (vowels and consonants) but which are properties of syllables and larger units of speech, including linguistic functions such as intonation, stress, and rhythm. Such elements are known as suprasegmentals.

[Wikipedia Prosody (linguistics)](https://en.wikipedia.org/wiki/Prosody_(linguistics))

### [Aligner](aligner.md)

Aligner class represents a PyTorch module responsible for alignment tasks in a sequence-to-sequence model. It uses convolutional layers combined with LeakyReLU activation functions to project inputs to a hidden representation.

Also, for training purposes, binarizes attention with [MAS](mas.md)

#### [Monotonic Alignments Shrink](mas.md)

`mas_width1` Applies a Monotonic Alignments Shrink (MAS) operation with a hard-coded width of 1 to an attention map.
Mas with hardcoded `width=1`

`b_mas` Applies Monotonic Alignments Shrink (MAS) operation in parallel to the batches of an attention map.
It uses the `mas_width1` function internally to perform MAS operation.
