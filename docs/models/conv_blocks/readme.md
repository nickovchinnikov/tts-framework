## References

### [Activation Function GLU](activation.md)

Implements the Gated Linear Unit (GLU) activation function

Paper: [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083v3)

### [BSConv](bsconv.md)

`BSConv1d` implements the `BSConv` concept

Paper: [Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets](https://arxiv.org/abs/2003.13549)

### [Conv1d](conv1d.md)

Implements Depthwise 1D convolution. This module will apply a spatial convolution over inputs 
independently over each input channel in the style of depthwise convolutions.

### [Conv1dGLU](conv1d_glu.md)

It's based on the Deep Voice 3 project

Paper: [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)

* [ConvTransposed](conv_transposed.md) - `ConvTransposed` applies a 1D convolution operation, with the main difference that it transposes the 
last two dimensions of the input tensor before and after applying the `BSConv1d` convolution operation.

### [CoordConv1d](coord_conv1d.md)

Paper: [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)

* [AddCoords](add_coords.md)


