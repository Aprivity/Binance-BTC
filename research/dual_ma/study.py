"""Reproducible, non-trading 200 USDT BTC spot dual MA experiment.

Example: python -m research.dual_ma.study --interval 15m --days 365
All variants and risk rules are researcher-defined; original video
parameters have not been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from btc_quant.config import Config, STEPS
from btc_quant.core import backtest
from btc_quant.data import fetch, load_csv
from btc_quant.report import metrics, market_analysis
from research_strategies.dual_ma import CANDIDATES

COST_CASES = {
    "zero_upper_bound": (0.0, 0.0),
    "base": (10.0, 5.0),
    "doubled": (20.0, 10.0),
}
SOURCE_VIDEO = "https://www.youtube.com/watch?v=F2FJeRsUzNE"


def analyze(frame: pd.DataFrame, interval: str, output_dir: str | Path, *,
            days_requested: int | None = None) -> pd.DataFrame:
    """Fixed rules × costs × chronological split, no fitted parameters."""
    if interval not in STEPS:
        raise ValueError("Unsupported timeframe")
    if len(frame) < 1100:
        raise ValueError("Need 1100+ bars for meaningful study")
    stamps = frame.open_time_ms.astype("int64").reset_index(drop=True)
    if stamps.diff().dropna().ne(STEPS[interval]).any():
        raise ValueError("Candle gaps are not allowed; no interpolation")
    if not stamps.mod(STEPS[interval]).eq(0).all():
        raise ValueError("Misaligned candle timestamps")
    start_idx = 200
    split_idx = start_idx + int((len(frame) - start_idx) * 0.7)
    if split_idx >= len(frame) - 20:
        raise ValueError("Insufficient held-out test candles")
    train_start = int(stamps.iloc[start_idx])
    split = int(stamps.iloc[split_idx])
    end = int(stamps.iloc[-1] + STEPS[interval])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades_dir = out / "trades"
    trades_dir.mkdir(exist_ok=True)
    rows = []
    hist = {"BTC/USDT": frame}
    for name, signals in CANDIDATES.items():
        for cost_name, (fee, slip) in COST_CASES.items():
            config = Config(
                timeframe=interval, starting_usdt=200.0,
                fee_bps=fee, slippage_bps=slip,
                risk_pct=1.0, max_open_risk_pct=2.0,
                max_symbol_pct=100.0, max_total_pct=100.0,
                max_drawdown_pct=12.0,
                stop_atr=2.0, take_atr=3.0,
                atr_period=14, min_notional=10.0, warmup_bars=200,
            )
            for stage, first, last in (
                    ("development", train_start, split),
                    ("held_out", split, end)):
                book = backtest(hist, config, signal_fn=signals,
                                start_ms=first, end_ms=last)
                result = metrics(book, config)
                benchmark = market_analysis(hist, config, book)
                rows.append({
                    "candidate": name,
                    "cost_case": cost_name,
                    "stage": stage,
                    "from_utc": pd.to_datetime(first, unit="ms", utc=True).isoformat(),
                    "until_exclusive_utc": pd.to_datetime(last, unit="ms", utc=True).isoformat(),
                    **{k: v for k, v in result.items()
                       if k not in ("pnl_by_symbol", "open_quantities", "note")},
                    "benchmark_return_pct": benchmark.get("benchmark_return_pct"),
                    "strategy_minus_benchmark_pct_points": benchmark.get(
                        "strategy_minus_benchmark_pct_points"),
                })
                pd.DataFrame(book.trades).to_csv(
                    trades_dir / f"{name}_{cost_name}_{stage}.csv", index=False)
    report = pd.DataFrame(rows).sort_values(
        ["stage", "candidate", "cost_case"]).reset_index(drop=True)
    report.to_csv(out / "costed_results.csv", index=False)
    input_sha256 = hashlib.sha256(
        frame[["open_time_ms", "open", "high", "low", "close", "base_volume"]]
        .to_csv(index=False, float_format="%.12g").encode("utf-8")
    ).hexdigest()
    methodology = {
        "source": SOURCE_VIDEO,
        "source_fidelity": "UNVERIFIED video settings. All moving average periods, confirmation, ATR exit rules and risk parameters here are researcher hypotheses.",
        "market": "BTC/USDT Binance SPOT ONLY, long-only. No futures, shorts, leverage or real trades.",
        "timeframe": interval,
        "days_requested": days_requested,
        "candles": len(frame),
        "data_sha256": input_sha256,
        "first_candle_utc": pd.to_datetime(int(stamps.iloc[0]), unit="ms", utc=True).isoformat(),
        "last_candle_utc": pd.to_datetime(int(stamps.iloc[-1]), unit="ms", utc=True).isoformat(),
        "holdout_start_utc": pd.to_datetime(split, unit="ms", utc=True).isoformat(),
        "split": "70 percent chronological development / 30 percent held-out AFTER 200 warmup candles; independent fresh account each fold; no tuning.",
        "starting_cash_usdt": 200,
        "candidates": list(CANDIDATES),
        "fees_slippage_bps_per_side": {
            key: {"fee": fee, "slippage": slip}
            for key, (fee, slip) in COST_CASES.items()
        },
        "risk_model": "1pct equity risk, 2pct open risk cap, up to 100pct cash exposure, 12pct peak DD stop-new-entry, ATR14 stop=2x and take=3x, minimum order 10 USDT",
        "entry": "Finished signal bar -> NEXT candle open; no lookahead",
        "exit": "MA reversal next open OR OHLC stop/take. When stop and take both touched, stop is assumed first.",
        "biases": [
            "Held-out sample becomes inspected after this run and cannot be reused as untouched confirmation.",
            "Multiple candidate/timeframe comparisons can overfit; no trading edge is established.",
            "Any position still open at end is marked to market, not liquidated; unrealized exit fees are omitted.",
            "Spot excludes margin, futures funding, and liquidation mechanics.",
            "Gaps through stops, market impact, liquidity and taxes are not fully simulated.",
            "BTC buy/hold reference applies matching cash cap, entry costs, and no terminal exit fee.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8")
    focus = report[
        (report.stage == "held_out") & (report.cost_case == "base")
    ].sort_values("net_return_pct", ascending=False)
    text = [
        "# 200U BTC spot dual moving average study",
        "",
        "**Caution:** Researcher-defined MA parameters; NOT a verified replication of the source video.",
        "Historical simulation only, not proof of profitability.",
        f"Interval: {interval}; bars: {len(frame)}; held-out begins: {methodology['holdout_start_utc']}.",
        "Model: 200 USDT unlevered spot cash; 10bps fee + 5bps slippage EACH SIDE for table.",
        "Stops: ATR14 x2; targets: ATR14 x3; next-open fills.",
        "",
        "| Rule | OOS net % | Capped BTC hold % | Trades closed | Max DD % | Fee USDT |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in focus.iterrows():
        bench = row.benchmark_return_pct
        bench_txt = f"{bench:+.2f}" if pd.notna(bench) else "N/A"
        text.append(
            f"| {row.candidate} | {row.net_return_pct:+.2f} | {bench_txt} "
            f"| {int(row.closed_trades)} | {row.max_drawdown_pct:.2f} "
            f"| {row.fees_paid_usdt:.2f} |")
    text += [
        "",
        "Full comparisons: costed_results.csv. Assumptions/hash: methodology.json.",
        "Closed trades per experiment: trades/.",
        "Not ranked for deployment; require future unseen data and conservative cost robustness.",
    ]
    (out / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description="200 USDT BTC spot dual-MA research; public data / NO ORDERS")
    p.add_argument("--interval", choices=tuple(STEPS), default="15m")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--csv-dir")
    p.add_argument("--source", choices=("binance", "ccxt"), default="binance")
    p.add_argument("--output-dir")
    args = p.parse_args(argv)
    if not 30 <= args.days <= 3000:
        p.error("--days must be in 30..3000")
    frame = (load_csv(Path(args.csv_dir) / f"BTCUSDT_{args.interval}.csv",
                      "BTC/USDT", args.interval)
             if args.csv_dir else fetch("BTC/USDT", args.interval,
                                        days=args.days, source=args.source))
    dest = args.output_dir or f"outputs/dual-ma-200u/{args.interval}"
    result = analyze(frame, args.interval, dest, days_requested=args.days)
    print(result[["candidate", "cost_case", "stage", "net_return_pct",
                  "closed_trades", "max_drawdown_pct"]].to_string(index=False))
    print(f"Analysis written to {dest}")


if __name__ == "__main__":
    main()
