#!/usr/bin/env python3
"""Calculates the unigram BLEU score for a sentence."""
import numpy as np


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations, each a list of words
        sentence: list containing the model proposed sentence words

    Returns:
        The unigram BLEU score
    """
    c = len(sentence)

    ref_lengths = [len(ref) for ref in references]
    r = min(ref_lengths, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c >= r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    sentence_counts = {}
    for word in sentence:
        sentence_counts[word] = sentence_counts.get(word, 0) + 1

    clipped_count = 0
    for word, count in sentence_counts.items():
        max_ref_count = 0
        for ref in references:
            ref_count = ref.count(word)
            if ref_count > max_ref_count:
                max_ref_count = ref_count
        clipped_count += min(count, max_ref_count)

    precision = clipped_count / c

    return bp * precision
