#!/usr/bin/env python3
"""Module for simple RNN forward propagation."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: RNNCell instance used for forward propagation
        X: numpy.ndarray of shape (t, m, i) with input data
        h_0: numpy.ndarray of shape (m, h) with initial hidden state

    Returns:
        H: numpy.ndarray containing all hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y_list = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y_list.append(y)

    Y = np.array(Y_list)
    return H, Y
