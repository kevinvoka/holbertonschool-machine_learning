#!/usr/bin/env python3
"""Train a gensim FastText model."""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a gensim fastText model.

    Args:
        sentences: list of sentences to be trained on.
        vector_size: dimensionality of the embedding layer.
        min_count: minimum number of occurrences of a word for use in training.
        negative: size of negative sampling.
        window: maximum distance between the current and predicted word.
        cbow: boolean; True for CBOW, False for Skip-gram.
        epochs: number of iterations to train over.
        seed: seed for the random number generator.
        workers: number of worker threads to train the model.

    Returns:
        the trained model.
    """
    sg = 0 if cbow else 1
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=sg,
        epochs=epochs,
        seed=seed,
        sorted_vocab=0,
        workers=workers
    )
    return model
