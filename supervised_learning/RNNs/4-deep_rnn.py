#!/usr/bin/env python3
"""Module for deep RNN forward propagation."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l
        X: numpy.ndarray of shape (t, m, i) with input data
        h_0: numpy.ndarray of shape (l, m, h) with initial hidden states

    Returns:
        H: numpy.ndarray containing all hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, i = X.shape
    layers = len(rnn_cells)
    h = h_0.shape[2]

    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0

    Y_list = []

    for step in range(t):
        layer_input = X[step]
        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(H[step, layer], layer_input)
            H[step + 1, layer] = h_next
            layer_input = h_next
        Y_list.append(y)

    Y = np.array(Y_list)
    return H, Y
