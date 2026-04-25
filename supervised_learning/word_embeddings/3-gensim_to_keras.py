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
    vocab_size, vector_size = weights.shape
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        trainable=True
    )
    embedding_layer.build((None,))
    embedding_layer.set_weights([weights])
    return embedding_layer
