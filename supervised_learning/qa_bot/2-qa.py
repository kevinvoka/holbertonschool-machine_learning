#!/usr/bin/env python3
"""Answer questions from a reference text in a loop."""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answer questions from a reference text.

    Args:
        reference: string containing the reference text.
    """
    while True:
        question = input('Q: ')
        if question.lower() in ('exit', 'quit', 'goodbye', 'bye'):
            print('A: Goodbye')
            break
        answer = question_answer(question, reference)
        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: ' + answer)
