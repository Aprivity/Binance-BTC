"""CLI for data, backtest, demo, paper and report. No live trading commands."""
import argparse
import json
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from .config import Config, SYMBOLS, STEPS
from .data import fetch, load_csv, save_csv, save_parquet, load_parquet
from .core import backtest, Account
from .paper import run_paper, load_state
from .report import write_report, market_analysis
from .strategies import STRATEGIES
from .research import compare
from .intraday import intraday_study, ARMS


def synthetic(symbol, timeframe, n=550):
    """Deterministic synthetic OHLCV strictly for testing—not performance claims."""
    idx = np.arange(n)
    price = (100 + (0.011 if symbol == "BTC/USDT" else 0.004) * idx
             + 8 * np.sin(idx / 13 + (0 if symbol == "BTC/USDT" else 1.2)))
    start = 1_700_000_000_000 // STEPS[timeframe] * STEPS[timeframe]
    x = pd.DataFrame({"open_time_ms": start + idx * STEPS[timeframe],
                      "open": price, "high": price + 1.5, "low": price - 1.5,
                      "close": price + 0.2 * np.sin(idx / 3),
                      "base_volume": np.ones(n) * 100})
    x["close_time_ms"] = x.open_time_ms + STEPS[timeframe] - 1
    return x


def main(argv=None):
    p = argparse.ArgumentParser(description="BTC/ETH spot research + paper only; never places live orders")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--symbols", nargs="+", default=list(SYMBOLS), choices=SYMBOLS)
        s.add_argument("--interval", choices=tuple(STEPS), default="4h")
        s.add_argument("--initial-cash", type=float, default=1000)
        s.add_argument("--fee-bps", type=float, default=10)
        s.add_argument("--slippage-bps", type=float, default=5)
        s.add_argument("--risk-pct", type=float, default=1)
        s.add_argument("--max-symbol-pct", type=float, default=30)
        s.add_argument("--max-total-pct", type=float, default=50)
        s.add_argument("--max-open-risk-pct", type=float, default=2)
        s.add_argument("--max-drawdown-pct", type=float, default=12)

    demo = sub.add_parser("demo", help="Synthetic data smoke test only")
    common(demo)
    demo.add_argument("--output-dir", default="outputs/demo-v2")
    bt = sub.add_parser("backtest", help="Historical synchronized spot backtest")
    common(bt)
    bt.add_argument("--days", type=int, default=365)
    bt.add_argument("--source", choices=["binance", "ccxt"], default="binance")
    bt.add_argument("--csv-dir", help="Offline CSV directory: BTCUSDT_4h.csv, ETHUSDT_4h.csv")
    bt.add_argument("--parquet-dir", help="Offline Parquet directory: BTCUSDT_4h.parquet, ETHUSDT_4h.parquet")
    bt.add_argument("--mode", choices=["portfolio", "independent"], default="portfolio")
    bt.add_argument("--output-dir", default="outputs/backtest-v2")
    cmp = sub.add_parser("compare", help="Research only: four strategies with untouched time holdout")
    common(cmp)
    cmp.add_argument("--days", type=int, default=730)
    cmp.add_argument("--source", choices=["binance", "ccxt"], default="binance")
    cmp.add_argument("--csv-dir")
    cmp.add_argument("--parquet-dir")
    cmp.add_argument("--scope", choices=["all", "portfolio", "independent"], default="all")
    cmp.add_argument("--train-ratio", type=float, default=.7)
    cmp.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    cmp.add_argument("--output-dir", default="outputs/compare-v021")
    intraday = sub.add_parser("intraday-study", help="Research-only BTC 4h A-E experiments, walk-forward")
    common(intraday)
    intraday.add_argument("--csv-dir", required=True, help="CSV directory containing BTCUSDT_4h.csv")
    intraday.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    intraday.add_argument("--train-days", type=int, default=180)
    intraday.add_argument("--test-days", type=int, default=60)
    intraday.add_argument("--step-days", type=int, default=60)
    intraday.add_argument("--output-dir", default="outputs/btc-intraday-v022")
    paper = sub.add_parser("paper", help="Forward paper simulation, no real orders")
    common(paper)
    paper.add_argument("--state", default="outputs/paper-v2/state.json")
    paper.add_argument("--once", action="store_true")
    paper.add_argument("--poll-seconds", type=int, default=60)
    report = sub.add_parser("report", help="Existing v2 paper-account report")
    report.add_argument("--state", default="outputs/paper-v2/state.json")
    report.add_argument("--output-dir", default="outputs/paper-v2/report")
    data = sub.add_parser("data", help="Download historical OHLCV to CSV files")
    data.add_argument("--symbols", nargs="+", default=list(SYMBOLS), choices=SYMBOLS)
    data.add_argument("--interval", choices=tuple(STEPS), default="4h")
    data.add_argument("--days", type=int, default=365)
    data.add_argument("--source", choices=["binance", "ccxt"], default="binance")
    data.add_argument("--format", choices=["csv", "parquet"], default="csv")
    data.add_argument("--output-dir", default="data/market")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "report":
        raw = json.loads(Path(args.state).read_text(encoding="utf-8"))
        cfg = Config.from_dict(raw["config"])
        if raw["schema_version"] != 2:
            raise ValueError("Only v2 paper states are supported")
        book = Account.restore(raw["account"])
        print(json.dumps(write_report(book, cfg, args.output_dir), indent=2))
        return
    if args.cmd == "data":
        for s in args.symbols:
            df = fetch(s, args.interval, args.days, args.source)
            path = Path(args.output_dir) / f"{s.replace('/', '')}_{args.interval}.{args.format}"
            (save_csv if args.format == "csv" else save_parquet)(df, path)
            print(f"{s}: {len(df)} closed candles -> {path}")
        return
    if args.cmd in ("backtest", "compare") and args.csv_dir and args.parquet_dir:
        p.error("Choose either --csv-dir or --parquet-dir")
    cfg = Config(symbols=tuple(args.symbols), timeframe=args.interval,
                 starting_usdt=args.initial_cash, fee_bps=args.fee_bps,
                 slippage_bps=args.slippage_bps, risk_pct=args.risk_pct,
                 max_symbol_pct=args.max_symbol_pct, max_total_pct=args.max_total_pct,
                 max_open_risk_pct=args.max_open_risk_pct, max_drawdown_pct=args.max_drawdown_pct)
    if args.cmd == "intraday-study":
        if args.symbols != ["BTC/USDT"] or args.interval != "4h":
            p.error("intraday-study requires --symbols BTC/USDT --interval 4h")
        history = load_csv(Path(args.csv_dir) / "BTCUSDT_4h.csv", "BTC/USDT", "4h")
        fold, aggregate, info = intraday_study(history, cfg, arms=tuple(args.arms),
                                             train_days=args.train_days, test_days=args.test_days,
                                             step_days=args.step_days, output_dir=args.output_dir)
        print(f"BTC-only 4h historical walk-forward: {info['folds']} reset folds")
        print(aggregate.to_string(index=False))
        print(f"Saved diagnostics, fold summaries and plot to {args.output_dir}")
        return
    if args.cmd == "paper":
        log_path = Path(args.state).with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.getLogger().addHandler(RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3))
        run_paper(cfg, args.state, once=args.once, poll_seconds=args.poll_seconds)
        return
    if args.cmd == "demo":
        history = {s: synthetic(s, cfg.timeframe) for s in cfg.symbols}
        days = None
        print("[DEMO] SYNTHETIC DATA ONLY. NO INVESTMENT SIGNIFICANCE.")
    else:
        history = {s: (load_csv(Path(args.csv_dir) / f"{s.replace('/', '')}_{cfg.timeframe}.csv", s, cfg.timeframe)
                        if args.csv_dir else load_parquet(Path(args.parquet_dir) / f"{s.replace('/', '')}_{cfg.timeframe}.parquet", s, cfg.timeframe)
                        if args.parquet_dir else fetch(s, cfg.timeframe, args.days, args.source))
                   for s in cfg.symbols}
        days = args.days
    if args.cmd == "compare":
        summary, methodology = compare(history, cfg, strategies=tuple(args.strategies),
                                       train_ratio=args.train_ratio, scope=args.scope,
                                       days=args.days, output_dir=args.output_dir)
        print(f"Holdout starts {methodology['split_utc']}")
        cols = ["scope", "strategy", "stage", "net_return_pct", "max_drawdown_pct",
                "closed_trades", "profit_factor", "policy_hold_return_pct", "avg_exposure_pct"]
        print(summary[cols].to_string(index=False))
        print(f"Saved CSV, methodology and plots to {args.output_dir}")
        return
    if args.cmd == "backtest" and args.mode == "independent":
        for s in cfg.symbols:
            per = Config(**{**cfg.to_dict(), "symbols": (s,)})
            book = backtest({s: history[s]}, per, days)
            target = Path(args.output_dir) / s.replace("/", "")
            result = write_report(book, per, target, plot=True)
            analysis = market_analysis({s: history[s]}, per, book)
            (target / "market_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            print(s, json.dumps({**result, **analysis}, indent=2))
    else:
        book = backtest(history, cfg, days)
        result = write_report(book, cfg, args.output_dir)
        analysis = market_analysis(history, cfg, book)
        (Path(args.output_dir) / "market_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(json.dumps({**result, **analysis}, indent=2))


if __name__ == "__main__":
    main()
