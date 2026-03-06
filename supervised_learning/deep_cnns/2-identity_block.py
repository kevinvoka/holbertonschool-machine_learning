#!/usr/bin/env python3
"""Identity block module for ResNet architecture."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block as described in Deep Residual Learning (2015).

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12

    Returns:
        activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    x = K.layers.Conv2D(
        F11, (1, 1), padding='same', kernel_initializer=init
    )(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(
        F3, (3, 3), padding='same', kernel_initializer=init
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(
        F12, (1, 1), padding='same', kernel_initializer=init
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
