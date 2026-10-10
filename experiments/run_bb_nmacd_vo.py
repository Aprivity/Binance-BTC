"""Public Binance USD-M archives -> independent BTC 15m three-exit research report.

Never opens orders; only public HTTPS archive GET and local files. Requires a
complete, contiguous candle series and actual historical funding records.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import requests
from experiments.bb_nmacd_vo import BAR_MS, MODES, indicators, simulate

ARCHIVE = "https://data.binance.vision/data/futures/um"
SYMBOL = "BTCUSDT"


def utc_ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def timestamps(values) -> np.ndarray:
    x = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.int64)
    if len(x) and np.median(x) > 1e14:
        x = x // 1000  # Binance Vision 2025+ files can be in microseconds.
    return x


def retrieve(session: requests.Session, rel: str, cache: Path) -> pd.DataFrame:
    target = cache / rel
    if target.exists():
        blob = target.read_bytes()
    else:
        url = f"{ARCHIVE}/{rel}"
        for attempt in range(3):
            try:
                response = session.get(url, timeout=90)
                response.raise_for_status()
                blob = response.content
                if len(blob) > 32_000_000:
                    raise ValueError("Archive exceeded 32 MB cap")
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                import time
                time.sleep(2 ** attempt)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        entries = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(entries) != 1:
            raise ValueError(f"Unexpected archive members: {rel}")
        with archive.open(entries[0]) as stream:
            return pd.read_csv(stream, header=None, low_memory=False)


def parse_klines(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[1] < 6:
        raise ValueError("Malformed Kline")
    if str(raw.iloc[0, 0]).strip().lower() in ("open_time", "opentime"):
        raw = raw.iloc[1:].reset_index(drop=True)
    return pd.DataFrame({"open_time_ms": timestamps(raw.iloc[:, 0]),
                         "open": pd.to_numeric(raw.iloc[:, 1], errors="raise"),
                         "high": pd.to_numeric(raw.iloc[:, 2], errors="raise"),
                         "low": pd.to_numeric(raw.iloc[:, 3], errors="raise"),
                         "close": pd.to_numeric(raw.iloc[:, 4], errors="raise")})


def parse_funding(raw: pd.DataFrame) -> pd.DataFrame:
    first = [str(v).strip().lower() for v in raw.iloc[0].tolist()]
    headers = any("time" in v or "rate" in v for v in first)
    if headers:
        time_col = next((i for i, v in enumerate(first)
                         if v in ("calc_time", "fundingtime", "funding_time", "time")), None)
        rate_col = next((i for i, v in enumerate(first)
                         if v in ("last_funding_rate", "fundingrate", "funding_rate", "rate")), None)
        if time_col is None or rate_col is None:
            raise ValueError(f"Unsupported funding header {first}")
        raw = raw.iloc[1:].reset_index(drop=True)
    else:
        if raw.shape[1] < 3:
            raise ValueError("Unrecognized headerless funding CSV")
        time_col, rate_col = 1, 3 if raw.shape[1] >= 4 else 2
    result = pd.DataFrame({"funding_ms": timestamps(raw.iloc[:, time_col]),
                           "rate": pd.to_numeric(raw.iloc[:, rate_col], errors="raise")})
    # Binance Vision funding calc_time is sometimes settlement+1 ms.
    # Normalize ONLY <= 1 second precision offsets; reject genuine mismatch.
    rounded = ((result.funding_ms.astype("int64") + BAR_MS // 2) // BAR_MS) * BAR_MS
    offset = (result.funding_ms.astype("int64") - rounded).abs()
    if (offset > 1000).any():
        raise ValueError(f"Funding timestamp >1s from 15m boundary: {result.loc[offset > 1000, 'funding_ms'].head().tolist()}")
    result["funding_ms"] = rounded
    if result.funding_ms.duplicated().any():
        raise ValueError("Duplicate funding settlement after timestamp normalization")
    if (not np.isfinite(result.rate).all()) or (result.rate.abs() > 0.03).any():
        raise ValueError("Invalid funding rates")
    return result


def gather(start: str, end: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    begin = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    finish = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if not begin < finish <= datetime.now(timezone.utc) + timedelta(days=1):
        raise ValueError("Invalid date range")
    first = begin - timedelta(days=5)  # indicator warmup
    last = finish - timedelta(days=1)
    candles, funding = [], []
    with requests.Session() as session:
        for month in pd.period_range(first.strftime("%Y-%m"), last.strftime("%Y-%m"), freq="M"):
            ym = str(month)
            month_begin = month.start_time.tz_localize("UTC").to_pydatetime()
            next_month = (month + 1).start_time.tz_localize("UTC").to_pydatetime()
            # Binance does not publish monthly archives for the current month.
            if next_month <= datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0):
                specs = [("monthly", ym)]
            else:
                day_start = max(first, month_begin)
                day_end = min(finish, next_month)
                specs = [("daily", d.strftime("%Y-%m-%d"))
                         for d in pd.date_range(day_start.date(), (day_end - timedelta(days=1)).date())]
            for bucket, stamp in specs:
                k = (f"{bucket}/klines/{SYMBOL}/15m/"
                     f"{SYMBOL}-15m-{stamp}.zip")
                f = (f"{bucket}/fundingRate/{SYMBOL}/"
                     f"{SYMBOL}-fundingRate-{stamp}.zip")
                candles.append(parse_klines(retrieve(session, k, cache_dir)))
                funding.append(parse_funding(retrieve(session, f, cache_dir)))
            print(f"Fetched {ym} ({len(specs)} archives per series)", flush=True)
    price = pd.concat(candles, ignore_index=True).drop_duplicates("open_time_ms")
    price = price.sort_values("open_time_ms").reset_index(drop=True)
    price = price[(price.open_time_ms >= int(first.timestamp() * 1000))
                  & (price.open_time_ms < int(finish.timestamp() * 1000))].reset_index(drop=True)
    rates = pd.concat(funding, ignore_index=True).drop_duplicates("funding_ms")
    rates = rates.sort_values("funding_ms").reset_index(drop=True)
    rates = rates[(rates.funding_ms >= int(first.timestamp() * 1000))
                  & (rates.funding_ms < int(finish.timestamp() * 1000))].reset_index(drop=True)
    return price, rates


def validate_market(prices: pd.DataFrame, rates: pd.DataFrame, start_ms: int, end_ms: int) -> None:
    if len(prices) < 201 or not {"open_time_ms", "open", "high", "low", "close"}.issubset(prices.columns):
        raise ValueError("Insufficient candles")
    stamps = prices.open_time_ms.to_numpy(np.int64)
    if (np.any(np.diff(stamps) != BAR_MS) or len(np.unique(stamps)) != len(stamps)
            or stamps[0] > start_ms - 200 * BAR_MS or stamps[-1] < end_ms - BAR_MS):
        raise ValueError("Missing/misaligned BTC 15m candles or warmup")
    raw = prices[["open", "high", "low", "close"]].to_numpy(float)
    if (not np.isfinite(raw).all() or np.any(raw <= 0)
            or np.any(raw[:, 1] < raw[:, [0, 2, 3]].max(axis=1))
            or np.any(raw[:, 2] > raw[:, [0, 1, 3]].min(axis=1))):
        raise ValueError("Bad OHLC values")
    if rates.empty or rates.funding_ms.duplicated().any():
        raise ValueError("Missing or duplicate historical funding")
    times = rates.funding_ms.to_numpy(dtype=np.int64)
    if (len(times) > 1 and np.any(np.diff(times) > 24 * 3600_000)):
        raise ValueError("Funding archival gap longer than 24h")
    # Require surrounding settlements, but not necessarily at midnight.
    in_window = times[(times >= start_ms) & (times < end_ms)]
    if (not len(in_window) or in_window[0] - start_ms > 24 * 3600_000
            or end_ms - in_window[-1] > 24 * 3600_000):
        raise ValueError("Insufficient funding coverage")


def main() -> None:
    p = argparse.ArgumentParser(description="BTC 15m BB+NMACD+VO backtest; PAPER ONLY")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-10-01", help="UTC exclusive; completed months recommended")
    p.add_argument("--initial-cash", type=float, default=1000)
    p.add_argument("--direction", choices=("both", "long", "short"), default="both")
    p.add_argument("--fee-bps", type=float, default=5)
    p.add_argument("--slippage-bps", type=float, default=2)
    p.add_argument("--risk-pct", type=float, default=1)
    p.add_argument("--candles-csv", type=Path, help="Optional local futures 15m OHLC CSV")
    p.add_argument("--funding-csv", type=Path, help="Required with --candles-csv: funding_ms,rate")
    p.add_argument("--cache-dir", type=Path, default=Path("data/btc-um-archive"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/bb-nmacd-vo-15m"))
    args = p.parse_args()
    if bool(args.candles_csv) != bool(args.funding_csv):
        p.error("--candles-csv and --funding-csv must be provided together")
    if utc_ms(args.end) <= utc_ms(args.start):
        p.error("End must be later than start")
    if args.candles_csv:
        prices = pd.read_csv(args.candles_csv)
        rates = pd.read_csv(args.funding_csv)
    else:
        prices, rates = gather(args.start, args.end, args.cache_dir)
    prices = prices.sort_values("open_time_ms").reset_index(drop=True)
    rates = rates.sort_values("funding_ms").reset_index(drop=True)
    start_ms, end_ms = utc_ms(args.start), utc_ms(args.end)
    validate_market(prices, rates, start_ms, end_ms)
    features = indicators(prices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in MODES:
        trades, curve, summary = simulate(
            features, rates, mode=mode, direction=args.direction,
            initial_cash=args.initial_cash, fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps, risk_pct=args.risk_pct,
            start_ms=start_ms, end_ms=end_ms)
        trades.to_csv(args.output_dir / f"trades_{mode}.csv", index=False)
        curve.to_csv(args.output_dir / f"equity_{mode}.csv", index=False)
        summary["signal_count_long"] = int(features.loc[
            (features.open_time_ms >= start_ms) & (features.open_time_ms < end_ms), "long_signal"].sum())
        summary["signal_count_short"] = int(features.loc[
            (features.open_time_ms >= start_ms) & (features.open_time_ms < end_ms), "short_signal"].sum())
        results.append(summary)
    doc = {"symbol": "BTCUSDT USD-M perpetual (public archive)",
           "timeframe": "15m", "start_utc": args.start, "end_utc_exclusive": args.end,
           "candles": len(prices), "funding_events": len(rates),
           "live_orders": False, "leverage_cap": "1x gross notional; no liquidation model",
           "funding_convention": "event rate * bar-open notional, carried position only",
           "slippage_convention": "adverse fill per entry and exit, stop first if OHLC ambiguous",
           "video_rule_assumptions": "strict body crossing BB middle; key candle low/high stop; close-bar signals",
           "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(doc, indent=2), encoding="utf8")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
