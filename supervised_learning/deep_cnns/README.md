# Deep Convolutional Architectures

This project implements several landmark deep convolutional neural network architectures using TensorFlow/Keras.

## Architectures Implemented

### Inception Block & Network (GoogLeNet)
- `0-inception_block.py`: Builds an inception block with parallel 1x1, 3x3, 5x5 convolutions and max pooling
- `1-inception_network.py`: Full GoogLeNet/Inception v1 network (224x224x3 input, 1000-class output)

### ResNet Blocks & ResNet-50
- `2-identity_block.py`: Identity block with bottleneck (1x1 -> 3x3 -> 1x1) and skip connection
- `3-projection_block.py`: Projection block with shortcut 1x1 conv to match dimensions
- `4-resnet50.py`: Full ResNet-50 architecture (224x224x3 input, 1000-class output)

### DenseNet-121
- `5-dense_block.py`: Dense block using DenseNet-B bottleneck layers with feature concatenation
- `6-transition_layer.py`: Transition layer with DenseNet-C compression
- `7-densenet121.py`: Full DenseNet-121 architecture (224x224x3 input, 1000-class output)

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
