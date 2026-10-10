"""Public daily cross-asset observations, cached with conservative availability.

Daily dates are session labels, never intraday timestamps. No authenticated APIs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from .data import fetch, save_csv

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
ASSETS = {"ndx": "^NDX", "dxy": "DX-Y.NYB", "gold": "GC=F"}


def available_times(dates):
    """Next calendar day 00:01 New York, handling DST before conversion to UTC."""
    local = pd.DatetimeIndex(pd.to_datetime(dates)).normalize() + pd.Timedelta(days=1)
    return local.tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(minutes=1)


def validate_daily(frame):
    df = frame.copy()
    if not {"date", "close"}.issubset(df) or df.empty:
        raise ValueError("Daily input requires date and close")
    df["date"] = pd.to_datetime(df.date, errors="raise").dt.strftime("%Y-%m-%d")
    df["close"] = pd.to_numeric(df.close, errors="raise")
    if df.date.duplicated().any() or df.close.isna().any() or not df.close.between(0, float("inf"), inclusive="neither").all():
        raise ValueError("Duplicate daily dates or invalid prices")
    return df.sort_values("date").reset_index(drop=True)


def download(directory, start="2023-01-01", end=None):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    end = now if end is None else pd.Timestamp(end, tz="UTC")
    if end > now or pd.Timestamp(start, tz="UTC") >= end:
        raise ValueError("Require start < end <= now")
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 BTCQuantResearch/0.1"
    provenance = {"retrieved_at_utc": now.isoformat(), "start": start,
                  "end_exclusive_utc": end.isoformat(), "inputs": {}}
    for asset, symbol in ASSETS.items():
        # Keep a year of preceding observations for 20-session feature warmup.
        params = {"period1": int((pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=365)).timestamp()),
                  "period2": int(end.timestamp()), "interval": "1d"}
        response = session.get(YAHOO + quote(symbol, safe=""), params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()["chart"]
        if payload.get("error") or not payload.get("result"):
            raise ValueError(f"No Yahoo daily data for {symbol}")
        item = payload["result"][0]
        if item["meta"]["exchangeTimezoneName"] != "America/New_York":
            raise ValueError("Unexpected source timezone; verify date mapping before use")
        dates = pd.to_datetime(item["timestamp"], unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d")
        daily = pd.DataFrame({"date": dates, "close": item["indicators"]["quote"][0]["close"]}).dropna()
        daily = validate_daily(daily)
        daily = daily.loc[available_times(daily.date) < end].reset_index(drop=True)
        path = out / f"{asset}_daily.csv"
        daily.to_csv(path, index=False)
        provenance["inputs"][asset] = {"symbol": symbol, "source_url": response.url,
            "rows": len(daily), "first_date": daily.date.iloc[0], "last_date": daily.date.iloc[-1],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_sha256": hashlib.sha256(response.content).hexdigest()}
    response = session.get(VIX_URL, timeout=30)
    response.raise_for_status()
    vix = pd.read_csv(io.StringIO(response.text)).rename(columns={"DATE": "date", "CLOSE": "close"})[["date", "close"]]
    vix = validate_daily(vix)
    vix = vix.loc[(vix.date >= str(pd.Timestamp(start) - pd.Timedelta(days=365))[:10]) & (available_times(vix.date) < end)]
    path = out / "vix_daily.csv"
    vix.to_csv(path, index=False)
    provenance["inputs"]["vix"] = {"symbol": "CBOE VIX", "source_url": VIX_URL,
        "rows": len(vix), "first_date": vix.date.iloc[0], "last_date": vix.date.iloc[-1],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(response.content).hexdigest()}
    days = (now - pd.Timestamp(start, tz="UTC")).days + 2
    btc = fetch("BTC/USDT", "4h", days=days, now_ms=int(end.timestamp() * 1000))
    btc = btc.loc[btc.open_time_ms >= pd.Timestamp(start, tz="UTC").timestamp() * 1000]
    path = out / "BTCUSDT_4h.csv"
    save_csv(btc, path)
    provenance["inputs"]["btc"] = {"symbol": "BTC/USDT", "source_url": "https://data-api.binance.vision/api/v3/klines",
        "rows": len(btc), "first_open_utc": btc.timestamp.iloc[0].isoformat(),
        "last_open_utc": btc.timestamp.iloc[-1].isoformat(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    provenance["availability_policy"] = "Daily session date becomes eligible at next calendar day 00:01 America/New_York; first subsequent 4h BTC boundary. Historical vendor revisions are not point-in-time vintages."
    (out / "data_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main():
    p = argparse.ArgumentParser(description="Download public historical research inputs; no orders")
    p.add_argument("--directory", default="data/market/cross-asset")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", help="Exclusive UTC endpoint; cannot exceed now")
    a = p.parse_args()
    print(json.dumps(download(a.directory, a.start, a.end), indent=2))


if __name__ == "__main__":
    main()
