#!/usr/bin/env python3
"""Module for a simple RNN cell."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple Recurrent Neural Network."""

    def __init__(self, i, h, o):
        """Initialize the RNNCell.

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Compute softmax activation.

        Args:
            x: input array

        Returns:
            softmax of x
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            x_t: numpy.ndarray of shape (m, i) with data input for the cell

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y
