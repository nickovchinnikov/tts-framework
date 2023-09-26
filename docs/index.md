# TTS-Framework docs

This is the documentation for the TTS-Framework. Here you can find the technical description of the project and the API reference.
Also, here you can find the code specs, the project structure and the development guidelines.

**Basically here you can find everything you need to know about the project.**

You can read about this mkdocs-material here: [mkdocs-material](readme.md)

## [WIP] Modified version of DelightfulTTS and UnivNet codec.

## References

## Model

### [Acoustic Model](model/acoustic_model/readme.md)

The [DelightfulTTS: The Microsoft Speech Synthesis System for Blizzard Challenge 2021](https://arxiv.org/abs/2110.12612) AcousticModel class represents a PyTorch module for an acoustic model in text-to-speech (TTS).
The acoustic model is responsible for predicting speech signals from phoneme sequences.

The model comprises multiple sub-modules including encoder, decoder and various prosody encoders and predictors.
Additionally, a pitch and length adaptor are instantiated.

### [Reference Encoder](model/reference_encoder/readme.md)

Similar to Tacotron model, the reference encoder is used to extract the high-level features from the reference

### [Convolution Blocks](model/conv_blocks/readme.md)

This part of the code responsible for the convolution blocks used in the model. Based on the FastSpeech models from [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) by Yi Ren et al and [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) by Yi Ren et al.

### [Attention](model/attention/readme.md)

Attention mechanizm used in the model. The concept of "global style tokens" (GST) was introduced in 
[Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) by Yuxuan Wang et al.

### [Helpers](helpers/readme.md)

Useful helper functions used in the model and test code.
