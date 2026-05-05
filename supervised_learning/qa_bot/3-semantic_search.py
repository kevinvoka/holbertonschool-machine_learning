#!/usr/bin/env python3
"""Semantic search on a corpus of documents."""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents.

    Args:
        corpus_path: path to the corpus of reference documents.
        sentence: sentence from which to perform semantic search.

    Returns:
        string containing the reference text most similar to sentence.
    """
    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )

    docs = []
    texts = []
    for filename in sorted(os.listdir(corpus_path)):
        if not filename.endswith('.md'):
            continue
        filepath = os.path.join(corpus_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        docs.append(text)
        texts.append(text)

    embeddings = model([sentence] + docs).numpy()
    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    similarities = np.inner(query_embedding, doc_embeddings)
    best_idx = int(np.argmax(similarities))

    return docs[best_idx]
