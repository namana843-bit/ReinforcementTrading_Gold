import os
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except Exception:  # numpy>=2 removed np.NaN which older pandas_ta imports
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    import pandas_ta as ta


DATETIME_CANDIDATES = [
    "Date", "Time", "Timestamp", "Datetime", "DateTime", "Date/Time",
    "Time (EET)", "Date (EET)", "datetime", "timestamp",
]


def _detect_datetime_column(df, datetime_col=None):
    if datetime_col is not None and datetime_col in df.columns:
        return datetime_col
    for c in df.columns:
        if c.strip().lower() in [x.lower() for x in DATETIME_CANDIDATES]:
            return c
    # fallback: first column parseable as datetime
    for c in df.columns:
        try:
            pd.to_datetime(df[c].head(10), errors="raise")
            return c
        except Exception:
            continue
    return df.columns[0]


def _clean_ohlcv(df):
    """Clean missing / non-trading bars."""
    df = df.copy()
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ohlc = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df = df.dropna(subset=ohlc)
    if ohlc:
        df = df[(df["High"] >= df["Low"]) &
                (df["High"] >= df["Close"]) &
                (df["Low"] <= df["Close"]) &
                (df["Close"] > 0)]
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0.0).clip(lower=0)
    return df


def add_indicators(df):
    """Add relative technical features to an OHLCV DataFrame (in place copy)."""
    df = df.copy()

    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)

    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "rsi_14",
    "atr_14",
    "ma_20_slope",
    "ma_50_slope",
    "close_ma20_diff",
    "close_ma50_diff",
    "ma_spread",
    "ma_spread_slope",
]


def load_and_preprocess_data(csv_path: str, datetime_col=None, timezone="UTC"):
    """
    Loads OHLCV data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns include: a datetime column, Open, High, Low, Close, Volume.
    The returned DataFrame still contains OHLCV for env internals, plus indicators.
    `feature_cols` lists only the RELATIVE columns to feed the agent.
    """
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()

    dt_col = _detect_datetime_column(raw, datetime_col)
    raw[dt_col] = pd.to_datetime(raw[dt_col], errors="coerce", dayfirst=False)
    if raw[dt_col].isna().mean() > 0.5:
        raw[dt_col] = pd.to_datetime(raw[dt_col], errors="coerce", dayfirst=True)

    df = raw.set_index(dt_col)
    df.sort_index(inplace=True)

    # timezone handling
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize(timezone)
        except Exception:
            pass
    else:
        try:
            df.index = df.index.tz_convert(timezone)
        except Exception:
            pass

    df = _clean_ohlcv(df)
    df = add_indicators(df)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df, feature_cols


def dataframe_from_records(records, timezone="UTC"):
    """Build a preprocessed DataFrame from a list of OHLCV dicts (live/webhook use)."""
    if not records:
        return None, []
    df = pd.DataFrame(records)
    # normalize column names
    rename = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("date", "time", "timestamp", "datetime"):
            rename[c] = "Date"
        elif low == "open":
            rename[c] = "Open"
        elif low == "high":
            rename[c] = "High"
        elif low == "low":
            rename[c] = "Low"
        elif low == "close":
            rename[c] = "Close"
        elif low == "volume":
            rename[c] = "Volume"
    df = df.rename(columns=rename)
    if "Date" not in df.columns:
        df["Date"] = pd.RangeIndex(len(df))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date")
    df.sort_index(inplace=True)
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize(timezone)
        except Exception:
            pass
    df = _clean_ohlcv(df)
    if len(df) == 0:
        return df, []
    df = add_indicators(df)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df, feature_cols
