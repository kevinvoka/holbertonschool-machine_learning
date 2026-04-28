#!/usr/bin/env python3
"""Convert a gensim word2vec model to a Keras Embedding layer."""
import tensorflow as tf


def gensim_to_keras(model):
    """Convert a gensim word2vec model to a trainable Keras Embedding layer.

    Args:
        model: a trained gensim word2vec model.

    Returns:
        a trainable keras Embedding layer.
    """
    weights = model.wv.vectors
    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
