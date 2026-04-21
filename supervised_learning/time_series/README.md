# Time Series Forecasting — BTC Price Prediction

Predicts the Bitcoin closing price **1 hour into the future** using the previous **24 hours** of 1-minute OHLCV data (coinbase + bitstamp).

---

## Project Structure

```
time_series/
├── preprocess_data.py      # data cleaning, merging, normalization
├── forecast_btc.py         # LSTM model, tf.data pipeline, training
└── README.md
```

---

## Setup

Place the raw dataset CSV files in this directory (or update paths in `preprocess_data.py`):

```
coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv
bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv
```

---

## Usage

### Step 1 — Preprocess the data

```bash
python3 preprocess_data.py
```

Outputs:
- `preprocessed_data.npy` — normalized Close price array (float32)
- `scaler_params.npy` — `[min_price, max_price]` for inverse transform

### Step 2 — Train and validate the model

```bash
python3 forecast_btc.py
```

Outputs:
- `btc_forecast_model.keras` — saved trained model
- Console: validation MSE / MAE in normalized and USD units

---

## Preprocessing Decisions

| Decision | Rationale |
|---|---|
| Coinbase as primary source | Generally more liquid, fewer anomalies after 2015 |
| Bitstamp to fill gaps | Longer history; covers periods coinbase was offline |
| Forward-fill → backward-fill gaps | Short exchange downtime should not create artificial jumps |
| Close price only | Reduces model complexity; other features are highly correlated with Close |
| Min-max normalization | Keeps gradients stable; allows easy USD inverse transform |
| Chronological 80/20 split | Prevents data leakage; validation always uses future data |

---

## Model Architecture

```
Input → (1440, 1)          24 hours of 1-minute close prices
LSTM(64, return_sequences)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(16, relu)
Dense(1)                   predicted close price 1 hour ahead
```

- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr = 1e-3, ReduceLROnPlateau)
- **Early stopping**: patience = 5 epochs on `val_loss`

---

## Window Design

```
|← 1440 steps (24 h) →|← 60 steps →| target
 t₀                   t₁            t₁ + 60
```

Each training sample uses 1440 consecutive 1-minute closes as input and the close price 60 minutes after the window end as the target — matching the approximate Bitcoin transaction confirmation time.
