## References

### [Style Embed Attention](style_embed_attention.md)
This mechanism is being used to extract style features from audio data in the form of spectrograms.

This technique is often used in text-to-speech synthesis (TTS) such as Tacotron-2, where the goal is to modulate the prosody, stress, and intonation of the synthesized speech based on the reference audio or some control parameters. The concept of "global style tokens" (GST) was introduced in 

[Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) by Yuxuan Wang et al.

### [Multi-Head Attention](multi_head_attention.md)

I found great explanations with code implementation of [Multi-Headed Attention (MHA)](https://nn.labml.ai/transformers/mha.html) by labml.ai Deep Learning Paper Implementations.

> This is a tutorial/implementation of multi-headed attention from paper Attention Is All You Need in PyTorch. The implementation is inspired from Annotated Transformer.

This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

$$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

In simple terms, it finds keys that matches the query, and gets the values of
    those keys.

It uses dot-product of query and key as the indicator of how matching they are.
Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
This is done to avoid large dot-product values causing softmax to
give very small gradients when $d_k$ is large.

Softmax is calculated along the axis of of the sequence (or time).

### [Relative Multi-Head Attention](relative_multi_head_attention.md)

Explanations with code implementation of [Relative Multi-Headed Attention](https://nn.labml.ai/transformers/xl/relative_mha.html) by labml.ai Deep Learning Paper Implementations.

Paper: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://papers.labml.ai/paper/1901.02860)
in [PyTorch](https://pytorch.org)


### [Conformer Multi-Headed Self Attention](conformer_multi_headed_self_attention.md)

Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
the relative sinusoidal positional encoding scheme.

### [Feed Forward](feed_forward.md)

Creates a feed-forward neural network.
The network includes a layer normalization, an activation function (`LeakyReLU`), and dropout layers.
