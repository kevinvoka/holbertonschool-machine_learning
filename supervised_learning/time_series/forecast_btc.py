#!/usr/bin/env python3
"""
Creates, trains, and validates a Keras LSTM model for BTC price forecasting.

The model uses a sliding window of the past 24 hours (1440 one-minute bars)
to predict the BTC closing price at the end of the following hour (60 bars
ahead), matching the typical Bitcoin transaction confirmation time.

Prerequisites:
  Run preprocess_data.py first to generate:
    - preprocessed_data.npy  : normalized Close price array
    - scaler_params.npy      : [min_price, max_price]

Outputs:
  - btc_forecast_model.keras : trained model saved to disk
"""
import numpy as np
import tensorflow as tf


WINDOW_SIZE = 24 * 60    # 1 440 one-minute bars  =  24 hours
HORIZON = 60             # predict close price 60 bars (1 hour) ahead
BATCH_SIZE = 64
EPOCHS = 50
TRAIN_SPLIT = 0.8
SHUFFLE_BUFFER = 2000

DATA_PATH = 'preprocessed_data.npy'
SCALER_PATH = 'scaler_params.npy'
MODEL_PATH = 'btc_forecast_model.keras'


def make_dataset(data, window_size=WINDOW_SIZE, horizon=HORIZON,
                 batch_size=BATCH_SIZE, shuffle=True):
    """Build a tf.data.Dataset of (window, target) pairs.

    Uses a sliding window across the time series.  Each sample is a
    sequence of `window_size` consecutive close prices (the past 24 h)
    and the corresponding target is the single close price `horizon`
    steps after the end of that window (1 h later).

    Args:
        data: 1-D numpy float32 array of normalized close prices
        window_size: number of input time steps (default 1440)
        horizon: steps ahead to forecast (default 60)
        batch_size: samples per batch (default 64)
        shuffle: whether to shuffle windows (True for training)

    Returns:
        tf.data.Dataset yielding (input_tensor, target_scalar) batches
            input_tensor shape : (batch, window_size, 1)
            target_scalar shape: (batch, 1)
    """
    total = window_size + horizon

    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(total, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(total))

    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=42)

    ds = ds.map(
        lambda w: (
            tf.expand_dims(w[:window_size], axis=-1),   # (window_size, 1)
            tf.expand_dims(w[-1], axis=-1)              # (1,)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(window_size=WINDOW_SIZE):
    """Build and compile a stacked LSTM model for BTC forecasting.

    Architecture:
      - LSTM(64, return_sequences=True) + Dropout(0.2)
      - LSTM(32) + Dropout(0.2)
      - Dense(16, relu)
      - Dense(1)  → single price prediction

    Loss: mean squared error (MSE)
    Optimizer: Adam with learning rate 1e-3

    Args:
        window_size: number of input time steps

    Returns:
        compiled tf.keras.Sequential model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='btc_lstm_forecaster')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


def main():
    """Load preprocessed data, train the model, and report validation MSE."""
    print("Loading preprocessed data...")
    data = np.load(DATA_PATH)
    scaler = np.load(SCALER_PATH)
    min_price, max_price = scaler[0], scaler[1]
    print(f"  Data points  : {len(data):,}")
    print(f"  Price range  : ${min_price:.2f}  –  ${max_price:.2f}")

    split_idx = int(len(data) * TRAIN_SPLIT)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"  Train samples: {len(train_data):,}")
    print(f"  Val   samples: {len(val_data):,}")

    print("\nBuilding tf.data pipelines...")
    train_ds = make_dataset(train_data, shuffle=True)
    val_ds = make_dataset(val_data, shuffle=False)

    print("\nBuilding model...")
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    print(f"\nTraining for up to {EPOCHS} epochs (early stopping enabled)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    val_mse, val_mae = model.evaluate(val_ds, verbose=0)

    price_range = max_price - min_price
    rmse_usd = (val_mse ** 0.5) * price_range
    mae_usd = val_mae * price_range

    print("\n--- Validation Results ---")
    print(f"  MSE  (normalized) : {val_mse:.6f}")
    print(f"  MAE  (normalized) : {val_mae:.6f}")
    print(f"  RMSE (USD)        : ${rmse_usd:.2f}")
    print(f"  MAE  (USD)        : ${mae_usd:.2f}")

    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    best_val = min(history.history['val_loss'])
    print(f"Best epoch: {best_epoch}  |  best val_loss: {best_val:.6f}")


if __name__ == '__main__':
    main()
