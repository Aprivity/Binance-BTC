"""Public BTC/USDT spot OHLCV downloader and strict UTC bar validation."""
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from .config import STEPS, SYMBOLS

LOG=logging.getLogger(__name__)
API="https://data-api.binance.vision/api/v3/klines"
COLS=["open_time_ms","open","high","low","close","base_volume","close_time_ms"]


def validate(frame,timeframe,*,require_closed=False,now_ms=None):
    if timeframe not in STEPS:
        raise ValueError("Invalid timeframe")
    df=frame.copy().sort_values("open_time_ms").reset_index(drop=True)
    if df.empty or not set(COLS).issubset(df.columns):
        raise ValueError("Missing OHLCV columns/data")
    if df.open_time_ms.duplicated().any():
        raise ValueError("Duplicate candles")
    for col in COLS:
        df[col]=pd.to_numeric(df[col],errors="raise")
    a=df[["open","high","low","close","base_volume"]].to_numpy(float)
    if (not np.isfinite(a).all() or (a[:,:4]<=0).any() or (a[:,4]<0).any()
        or (df.high<df[["open","close","low"]].max(axis=1)).any()
        or (df.low>df[["open","close","high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLCV range or volume")
    step=STEPS[timeframe]
    if (df.open_time_ms%step!=0).any() or (df.open_time_ms.diff().dropna()!=step).any():
        raise ValueError("Missing/misaligned candles")
    if (df.close_time_ms!=df.open_time_ms+step-1).any():
        raise ValueError("Incorrect bar close time")
    if require_closed and (df.close_time_ms >= (now_ms if now_ms is not None else int(time.time()*1000))).any():
        raise ValueError("Unclosed candle")
    df["timestamp"]=pd.to_datetime(df.open_time_ms,unit="ms",utc=True)
    return df


def normalize(raw,symbol,timeframe,now_ms=None,*,ccxt=False):
    if symbol not in SYMBOLS or timeframe not in STEPS:
        raise ValueError("BTC spot timeframe only")
    now=int(time.time()*1000) if now_ms is None else now_ms
    step=STEPS[timeframe]
    rows=[]
    for r in raw:
        opened=int(r[0]);opened=opened//1000 if opened>10**14 else opened
        if opened+step>now:
            continue
        if len(r)<(6 if ccxt else 7):
            raise ValueError("Malformed OHLCV")
        o,h,l,c,v=map(float,r[1:6])
        quote=None if ccxt or len(r)<8 else float(r[7])
        rows.append((opened,o,h,l,c,v,opened+step-1,quote))
    return validate(pd.DataFrame(rows,columns=[*COLS,"quote_volume"]),timeframe,
                    require_closed=True,now_ms=now)


def fetch(symbol,timeframe,days=365,source="binance",now_ms=None):
    if symbol not in SYMBOLS or timeframe not in STEPS or not 1<=days<=3000:
        raise ValueError("Invalid BTC symbol/timeframe/days")
    now=int(time.time()*1000) if now_ms is None else now_ms
    step=STEPS[timeframe]
    cursor=now-days*86_400_000-200*step
    raw=[]
    if source=="ccxt":
        try:
            import ccxt
        except ImportError as exc:
            raise RuntimeError("pip install ccxt for optional PUBLIC market data") from exc
        exchange=ccxt.binance({"enableRateLimit":True})
        while cursor<now:
            batch=exchange.fetch_ohlcv(symbol,timeframe,since=cursor,limit=1000)
            if not batch:
                break
            raw.extend(batch)
            nxt=int(batch[-1][0])+step
            if nxt<=cursor:
                raise RuntimeError("CCXT pagination stalled")
            cursor=nxt
        ccxt_mode=True
    elif source=="binance":
        session=requests.Session()
        while cursor<now:
            params={"symbol":"BTCUSDT","interval":timeframe,"startTime":cursor,
                    "endTime":now,"limit":1000}
            for attempt in range(4):
                try:
                    response=session.get(API,params=params,timeout=20)
                    response.raise_for_status()
                    batch=response.json()
                    if not isinstance(batch,list):
                        raise ValueError("Unexpected public Binance response")
                    break
                except (requests.RequestException,ValueError):
                    if attempt==3:
                        raise
                    time.sleep(2**attempt)
            if not batch:
                break
            raw.extend(batch)
            nxt=int(batch[-1][0])+step
            if nxt<=cursor:
                raise RuntimeError("Binance pagination stalled")
            cursor=nxt
            time.sleep(0.1)
        ccxt_mode=False
    else:
        raise ValueError("Unknown data source")
    # Each bar is public, completed BTC spot data only.
    raw=list({int(row[0]):row for row in raw}.values())
    df=normalize(raw,symbol,timeframe,now_ms=now,ccxt=ccxt_mode)
    for col,value in (("symbol",symbol),("exchange","binance"),("market_type","spot"),
                      ("timeframe",timeframe),("source",source)):
        df[col]=value
    return df


def load_csv(path,symbol,timeframe):
    if symbol not in SYMBOLS:
        raise ValueError("BTC only")
    df=pd.read_csv(path)
    if "open_time" in df and "open_time_ms" not in df:
        df=df.rename(columns={"open_time":"open_time_ms"})
    if "volume" in df and "base_volume" not in df:
        df=df.rename(columns={"volume":"base_volume"})
    if "symbol" in df and (df["symbol"]!=symbol).any():
        raise ValueError("CSV symbol mismatch")
    if "timeframe" in df and (df["timeframe"]!=timeframe).any():
        raise ValueError("CSV timeframe mismatch")
    df["close_time_ms"]=df.open_time_ms+STEPS[timeframe]-1
    df=validate(df,timeframe)
    df["symbol"]=symbol
    return df


def save_csv(df,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False)


def save_parquet(df,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(p,index=False)


def load_parquet(path,symbol,timeframe):
    if symbol not in SYMBOLS:
        raise ValueError("BTC only")
    df=pd.read_parquet(path)
    if "symbol" in df and (df["symbol"]!=symbol).any():
        raise ValueError("Parquet symbol mismatch")
    if "timeframe" in df and (df["timeframe"]!=timeframe).any():
        raise ValueError("Parquet timeframe mismatch")
    return validate(df,timeframe)