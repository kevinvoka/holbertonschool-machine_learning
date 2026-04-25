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
    features = sorted(
        model.wv.index_to_key,
        key=lambda word: (
            model.wv.get_vecattr(word, "count"),
            model.wv.key_to_index[word]
        ),
        reverse=True
    )
    order = [model.wv.key_to_index[word] for word in features]

    model.wv.vectors = model.wv.vectors[order]
    model.wv.index_to_key = features
    model.wv.key_to_index = {word: i for i, word in enumerate(features)}
    for attr, values in model.wv.expandos.items():
        model.wv.expandos[attr] = values[order]

    weights = model.wv.vectors
    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
