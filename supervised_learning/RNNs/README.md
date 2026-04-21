# RNNs - Recurrent Neural Networks

This project implements various RNN architectures from scratch using NumPy.

## Files

| File | Description |
|------|-------------|
| `0-rnn_cell.py` | Simple RNN cell with forward propagation |
| `1-rnn.py` | Forward propagation over all time steps for a simple RNN |
| `2-gru_cell.py` | Gated Recurrent Unit (GRU) cell |
| `3-lstm_cell.py` | Long Short-Term Memory (LSTM) cell |
| `4-deep_rnn.py` | Forward propagation for a deep (multi-layer) RNN |

## Concepts

- **RNN**: Simple recurrent cell using tanh activation for hidden state and softmax for output
- **GRU**: Uses update and reset gates to control information flow, addressing the vanishing gradient problem
- **LSTM**: Uses forget, update, and output gates along with a separate cell state for long-term memory
- **Deep RNN**: Stacks multiple RNN layers where each layer's hidden state feeds into the next
