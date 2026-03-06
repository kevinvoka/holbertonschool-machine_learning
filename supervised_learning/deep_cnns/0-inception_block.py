#!/usr/bin/env python3
"""Inception block module for deep convolutional architectures."""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Build an inception block as described in Going Deeper with Convolutions.

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3, F5R, F5, FPP

    Returns:
        concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu'
    )(A_prev)

    conv_3x3_reduce = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu'
    )(A_prev)
    conv_3x3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu'
    )(conv_3x3_reduce)

    conv_5x5_reduce = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu'
    )(A_prev)
    conv_5x5 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu'
    )(conv_5x5_reduce)

    max_pool = K.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same'
    )(A_prev)
    pool_proj = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu'
    )(max_pool)

    output = K.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    return output
