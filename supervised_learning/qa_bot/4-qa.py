#!/usr/bin/env python3
"""Multi-reference Question Answering loop."""
semantic_search = __import__('3-semantic_search').semantic_search
qa = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """Answer questions from multiple reference texts.

    Args:
        corpus_path: path to the corpus of reference documents.
    """
    while True:
        question = input('Q: ')
        if question.lower() in ('exit', 'quit', 'goodbye', 'bye'):
            print('A: Goodbye')
            break
        reference = semantic_search(corpus_path, question)
        answer = qa(question, reference)
        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: ' + answer)
