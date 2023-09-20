## References

### [Style Token Layer (STL)](STL.md)

This layer helps to encapsulate different speaking styles in token embeddings.

### [Reference Encoder](reference_encoder.md)

Similar to Tacotron model, the reference encoder is used to extract the high-level features from the reference
    
It consists of a number of convolutional blocks (`CoordConv1d` for the first one and `nn.Conv1d` for the rest), 
then followed by instance normalization and GRU layers.
The `CoordConv1d` at the first layer to better preserve positional information, paper:
[Robust and fine-grained prosody control of end-to-end speech synthesis](https://arxiv.org/pdf/1811.02122.pdf)

