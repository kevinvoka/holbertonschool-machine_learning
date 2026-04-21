#!/usr/bin/env python3
"""
Preprocesses BTC coinbase and bitstamp datasets for time series forecasting.

Expected input files (place in the same directory or provide full paths):
  - coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv
  - bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv

Each CSV has columns:
  Timestamp, Open, High, Low, Close, Volume_(BTC),
  Volume_(Currency), Weighted_Price

Output:
  - preprocessed_data.npy  : normalized Close prices as float32 array
  - scaler_params.npy      : [min_price, max_price] for inverse transform
"""
import numpy as np
import pandas as pd


COINBASE_PATH = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
BITSTAMP_PATH = 'bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'


def load_dataset(filepath):
    """Load a raw BTC dataset from a CSV file.

    Parses the Unix Timestamp column into a DatetimeIndex and sorts
    chronologically.

    Args:
        filepath: path to the CSV file

    Returns:
        pandas DataFrame with a DatetimeIndex sorted ascending
    """
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    df = df.sort_index()
    return df


def merge_datasets(coinbase, bitstamp):
    """Merge coinbase and bitstamp datasets.

    Uses coinbase as the primary source and fills any missing timestamps
    or NaN values with bitstamp data. This ensures maximum coverage while
    preferring the generally more liquid coinbase exchange.

    Args:
        coinbase: DataFrame loaded from the coinbase CSV
        bitstamp: DataFrame loaded from the bitstamp CSV

    Returns:
        pandas Series of merged Close prices with a DatetimeIndex
    """
    cb_close = coinbase[['Close']]
    bs_close = bitstamp[['Close']]

    # combine_first: use coinbase where available, fall back to bitstamp
    merged = cb_close.combine_first(bs_close)
    return merged['Close']


def clean_series(series):
    """Remove NaN values and non-positive prices from a price series.

    Forward-fills then backward-fills short gaps (e.g. exchange downtime)
    before dropping any remaining NaNs. Rows with zero or negative prices
    (data errors) are also removed.

    Args:
        series: pandas Series of raw Close prices

    Returns:
        cleaned pandas Series
    """
    series = series.ffill().bfill()
    series = series[series > 0]
    series = series.dropna()
    return series


def normalize(series):
    """Apply min-max normalization to scale prices into [0, 1].

    Args:
        series: pandas Series of Close prices

    Returns:
        tuple of (normalized_array, min_val, max_val)
            normalized_array : numpy float32 array in [0, 1]
            min_val          : original minimum price (for inverse transform)
            max_val          : original maximum price (for inverse transform)
    """
    min_val = series.min()
    max_val = series.max()
    normalized = (series - min_val) / (max_val - min_val)
    return normalized.values.astype(np.float32), min_val, max_val


def preprocess(coinbase_path=COINBASE_PATH,
               bitstamp_path=BITSTAMP_PATH,
               output_path='preprocessed_data.npy',
               scaler_path='scaler_params.npy'):
    """Full preprocessing pipeline for BTC time series forecasting.

    Steps:
      1. Load both exchange datasets
      2. Merge, preferring coinbase over bitstamp
      3. Forward/backward fill NaN gaps; drop bad rows
      4. Min-max normalize Close prices to [0, 1]
      5. Save normalized array and scaler parameters

    Args:
        coinbase_path: path to coinbase CSV file
        bitstamp_path: path to bitstamp CSV file
        output_path: path to save preprocessed numpy array
        scaler_path: path to save [min, max] scaler parameters

    Returns:
        numpy float32 array of normalized Close prices
    """
    print("Loading datasets...")
    coinbase = load_dataset(coinbase_path)
    bitstamp = load_dataset(bitstamp_path)
    print(f"  Coinbase rows : {len(coinbase):,}")
    print(f"  Bitstamp rows : {len(bitstamp):,}")

    print("Merging datasets (coinbase primary, bitstamp fallback)...")
    close = merge_datasets(coinbase, bitstamp)
    print(f"  Merged rows   : {len(close):,}")

    print("Cleaning series (fill gaps, remove bad prices)...")
    close = clean_series(close)
    print(f"  Clean rows    : {len(close):,}")
    print(f"  Date range    : {close.index[0]}  →  {close.index[-1]}")

    print("Normalizing (min-max to [0, 1])...")
    data, min_val, max_val = normalize(close)
    print(f"  Price range   : ${min_val:.2f}  –  ${max_val:.2f}")

    np.save(output_path, data)
    np.save(scaler_path, np.array([min_val, max_val]))
    print(f"Saved → {output_path}  (shape: {data.shape})")
    print(f"Saved → {scaler_path}  ([min={min_val:.2f}, max={max_val:.2f}])")

    return data


if __name__ == '__main__':
    preprocess()
