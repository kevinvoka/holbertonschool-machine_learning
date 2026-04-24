#!/usr/bin/env python3
"""Bag of Words embedding matrix."""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of vocabulary words to use for the analysis.
            If None, all words within sentences are used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings.
        features: list of the features used for embeddings.
    """
    def tokenize(sentence):
        """Lowercase and extract words, stripping punctuation."""
        words = re.findall(r"[a-zA-Z]+(?:'s)?", sentence.lower())
        cleaned = []
        for w in words:
            w = w.replace("'s", "")
            cleaned.append(w)
        return cleaned

    tokenized = [tokenize(s) for s in sentences]

    if vocab is None:
        all_words = set()
        for tokens in tokenized:
            all_words.update(tokens)
        features = sorted(list(all_words))
    else:
        features = list(vocab)

    word_to_idx = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, tokens in enumerate(tokenized):
        for token in tokens:
            if token in word_to_idx:
                embeddings[i][word_to_idx[token]] += 1

    return embeddings, np.array(features)
