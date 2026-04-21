#!/usr/bin/env python3
"""Module for a Long Short-Term Memory (LSTM) cell."""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Initialize the LSTMCell.

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            c_prev: numpy.ndarray of shape (m, h) with previous cell state
            x_t: numpy.ndarray of shape (m, i) with data input for the cell

        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(concat @ self.Wf + self.bf)
        u = self.sigmoid(concat @ self.Wu + self.bu)
        c_candidate = np.tanh(concat @ self.Wc + self.bc)
        o = self.sigmoid(concat @ self.Wo + self.bo)

        c_next = f * c_prev + u * c_candidate
        h_next = o * np.tanh(c_next)

        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, c_next, y
