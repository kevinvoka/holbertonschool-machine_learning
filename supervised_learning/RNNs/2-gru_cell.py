#!/usr/bin/env python3
"""Module for a Gated Recurrent Unit (GRU) cell."""
import numpy as np


class GRUCell:
    """Represents a Gated Recurrent Unit cell."""

    def __init__(self, i, h, o):
        """Initialize the GRUCell.

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Compute sigmoid activation.

        Args:
            x: input array

        Returns:
            sigmoid of x
        """
        return 1 / (1 + np.exp(-x))

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

        z = self.sigmoid(concat @ self.Wz + self.bz)
        r = self.sigmoid(concat @ self.Wr + self.br)

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_candidate = np.tanh(concat_r @ self.Wh + self.bh)

        h_next = (1 - z) * h_prev + z * h_candidate

        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y
