"""Public-only OHLCV access; timestamps are UTC epoch milliseconds."""
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from .config import STEPS, SYMBOLS

LOG = logging.getLogger(__name__)
API = "https://data-api.binance.vision/api/v3/klines"
COLS = ["open_time_ms", "open", "high", "low", "close", "base_volume", "close_time_ms"]


def validate(frame, timeframe, *, require_closed=False, now_ms=None):
    df = frame.copy().sort_values("open_time_ms").reset_index(drop=True)
    if df.empty or not set(COLS).issubset(df):
        raise ValueError("Missing OHLCV columns/data")
    if df.open_time_ms.duplicated().any():
        raise ValueError("Duplicate open times")
    for col in COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    values = df[["open", "high", "low", "close", "base_volume"]].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any()
            or (df.high < df[["open", "close", "low"]].max(axis=1)).any()
            or (df.low > df[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLCV price/volume")
    step = STEPS[timeframe]
    if (df.open_time_ms % step != 0).any():
        raise ValueError("Candle starts must align to UTC interval")
    if (df.open_time_ms.diff().dropna() != step).any():
        raise ValueError("Missing candles: refuse to backtest")
    if (df.close_time_ms != df.open_time_ms + step - 1).any():
        raise ValueError("Incorrect close times")
    if require_closed and (df.close_time_ms >= (now_ms or int(time.time() * 1000))).any():
        raise ValueError("Unclosed candle present")
    df["timestamp"] = pd.to_datetime(df.open_time_ms, unit="ms", utc=True)
    return df


def normalize(raw, symbol, timeframe, now_ms=None, *, ccxt=False):
    """Reject future bars and invalid market data; CCXT rows have 6 elements."""
    if symbol not in SYMBOLS:
        raise ValueError("Unsupported symbol")
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    step = STEPS[timeframe]
    rows = []
    for r in raw:
        opened = int(r[0])
        if opened > 10**14:  # archives may encode microseconds
            opened //= 1000
        if opened + step > now_ms:
            continue
        if ccxt:
            if len(r) < 6:
                raise ValueError("Malformed CCXT row")
            o, h, l, c, v = map(float, r[1:6])
        else:
            if len(r) < 7:
                raise ValueError("Malformed Binance row")
            o, h, l, c, v = map(float, r[1:6])
        qvol = None if ccxt or len(r) < 8 else float(r[7])
        rows.append((opened, o, h, l, c, v, opened + step - 1, qvol))
    return validate(pd.DataFrame(rows, columns=[*COLS, "quote_volume"]), timeframe, require_closed=True, now_ms=now_ms)


def fetch(symbol, timeframe, days=365, source="binance", now_ms=None):
    if symbol not in SYMBOLS or timeframe not in STEPS or not 1 <= days <= 3000:
        raise ValueError("Invalid symbol/timeframe/days")
    now = int(time.time() * 1000) if now_ms is None else now_ms
    step = STEPS[timeframe]
    cursor = now - days * 86_400_000 - (max(200, 2 * 21) * step)
    raw = []
    if source == "ccxt":
        try:
            import ccxt
        except ImportError as exc:
            raise RuntimeError("pip install ccxt to enable CCXT source") from exc
        exchange = ccxt.binance({"enableRateLimit": True})
        while cursor < now:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            raw.extend(batch)
            next_cursor = int(batch[-1][0]) + step
            if next_cursor <= cursor:
                raise RuntimeError("CCXT pagination stalled")
            cursor = next_cursor
        ccxt_mode = True
    elif source == "binance":
        session = requests.Session()
        while cursor < now:
            params = {"symbol": symbol.replace("/", ""), "interval": timeframe,
                      "startTime": cursor, "endTime": now, "limit": 1000}
            for attempt in range(4):
                try:
                    res = session.get(API, params=params, timeout=20)
                    if res.status_code in (418, 429) or res.status_code >= 500:
                        res.raise_for_status()
                    res.raise_for_status()
                    batch = res.json()
                    if not isinstance(batch, list):
                        raise ValueError(f"Unexpected Binance response: {batch}")
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            if not batch:
                break
            raw.extend(batch)
            next_cursor = int(batch[-1][0]) + step
            if next_cursor <= cursor:
                raise RuntimeError("Binance pagination stalled")
            cursor = next_cursor
            time.sleep(0.1)
        ccxt_mode = False
    else:
        raise ValueError("source must be binance or ccxt")
    # Duplicate boundary rows are harmless; missing intervals are not.
    raw = list({int(r[0]): r for r in raw}.values())
    df = normalize(raw, symbol, timeframe, now_ms=now, ccxt=ccxt_mode)
    df["symbol"] = symbol
    df["exchange"] = "binance"
    df["market_type"] = "spot"
    df["timeframe"] = timeframe
    df["source"] = source
    return df


def load_csv(path, symbol, timeframe):
    df = pd.read_csv(path)
    # Offline CSV accepts standard columns and treats all records as historical.
    if "open_time" in df and "open_time_ms" not in df:
        df = df.rename(columns={"open_time": "open_time_ms", "volume": "base_volume"})
    if "volume" in df.columns and "base_volume" not in df.columns:
        df = df.rename(columns={"volume": "base_volume"})
    if "symbol" in df.columns and (df["symbol"] != symbol).any():
        raise ValueError("CSV symbol does not match requested symbol")
    if "timeframe" in df.columns and (df["timeframe"] != timeframe).any():
        raise ValueError("CSV timeframe does not match requested timeframe")
    df["close_time_ms"] = df.open_time_ms + STEPS[timeframe] - 1
    df = validate(df, timeframe)
    df["symbol"] = symbol
    return df


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_parquet(df, path):
    """Optional pyarrow-backed columnar storage for repeat research runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)  # pip install pyarrow


def load_parquet(path, symbol, timeframe):
    df = pd.read_parquet(path)
    if "symbol" in df and (df.symbol != symbol).any():
        raise ValueError("Parquet symbol mismatch")
    if "timeframe" in df and (df.timeframe != timeframe).any():
        raise ValueError("Parquet timeframe mismatch")
    return validate(df, timeframe)
