#!/usr/bin/env python3
"""Calculates the cumulative n-gram BLEU score for a sentence."""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations, each a list of words
        sentence: list containing the model proposed sentence words
        n: size of the largest n-gram to use for evaluation

    Returns:
        The cumulative n-gram BLEU score
    """
    c = len(sentence)

    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c >= r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    def get_ngrams(words, size):
        """Extract n-grams from a list of words as a dict of counts."""
        ngrams = {}
        for i in range(len(words) - size + 1):
            gram = tuple(words[i:i + size])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    log_sum = 0.0
    weight = 1.0 / n

    for i in range(1, n + 1):
        sentence_ngrams = get_ngrams(sentence, i)
        total_ngrams = sum(sentence_ngrams.values())

        if total_ngrams == 0:
            return 0.0

        clipped_count = 0
        for gram, count in sentence_ngrams.items():
            max_ref_count = 0
            for ref in references:
                ref_ngrams = get_ngrams(ref, i)
                ref_count = ref_ngrams.get(gram, 0)
                if ref_count > max_ref_count:
                    max_ref_count = ref_count
            clipped_count += min(count, max_ref_count)

        if clipped_count == 0:
            return 0.0

        precision = clipped_count / total_ngrams
        log_sum += weight * np.log(precision)

    return bp * np.exp(log_sum)
