#!/usr/bin/env python3
"""Question Answering using BERT."""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Find a snippet of text within a reference document to answer a question.

    Args:
        question: string containing the question to answer.
        reference: string containing the reference document.

    Returns:
        string containing the answer, or None if no answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    tokens = tokenizer(
        question,
        reference,
        return_tensors='tf',
        truncation=True,
        max_length=512
    )

    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    token_type_ids = tokens['token_type_ids']

    outputs = model([input_ids, attention_mask, token_type_ids])

    start_logits = outputs[0][0]
    end_logits = outputs[1][0]

    input_ids_list = input_ids.numpy()[0]
    sep_index = list(input_ids_list).index(tokenizer.sep_token_id)

    start = int(tf.argmax(start_logits[sep_index + 1:]) + sep_index + 1)
    end = int(tf.argmax(end_logits[sep_index + 1:]) + sep_index + 2)

    if start >= end:
        return None

    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids_list[start:end])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer.strip():
        return None

    return answer
