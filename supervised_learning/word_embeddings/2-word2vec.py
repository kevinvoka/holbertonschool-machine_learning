#!/usr/bin/env python3
"""Train a gensim Word2Vec model."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a gensim word2vec model.

    Args:
        sentences: list of sentences to be trained on.
        vector_size: dimensionality of the embedding layer.
        min_count: minimum number of occurrences of a word for use in training.
        window: maximum distance between the current and predicted word.
        negative: size of negative sampling.
        cbow: boolean; True for CBOW, False for Skip-gram.
        epochs: number of iterations to train over.
        seed: seed for the random number generator.
        workers: number of worker threads to train the model.

    Returns:
        the trained model.
    """
    sg = 0 if cbow else 1
    model = gensim.models.Word2Vec(
        sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model
