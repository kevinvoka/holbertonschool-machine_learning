#!/usr/bin/env python3
"""TF-IDF embedding."""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of vocabulary words to use for the analysis.
            If None, all words within sentences are used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings.
        features: list of the features used for embeddings.
    """
    def tokenize(sentence):
        """Lowercase and extract words, stripping possessives."""
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

    tf = np.zeros((s, f))
    for i, tokens in enumerate(tokenized):
        if len(tokens) == 0:
            continue
        for token in tokens:
            if token in word_to_idx:
                tf[i][word_to_idx[token]] += 1
        tf[i] = tf[i] / len(tokens)

    df = np.zeros(f)
    for j, word in enumerate(features):
        for tokens in tokenized:
            if word in tokens:
                df[j] += 1

    idf = np.log((1 + s) / (1 + df)) + 1

    embeddings = tf * idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, np.array(features)
