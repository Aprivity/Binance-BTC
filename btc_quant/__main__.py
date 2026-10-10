"""Explicit strategy research, BTC-only price data and observer-first paper CLI."""
import argparse
import json
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from .config import Config, STEPS
from .core import Account, backtest
from .data import fetch, load_csv, load_parquet, save_csv, save_parquet
from .paper import run_paper
from .report import write_report, market_analysis
from .research import rolling_validate
from .strategy import no_trades, resolve_strategy


def synthetic(timeframe="4h", n=560):
    """Offline synthetic prices. For smoke tests only; no embedded trading strategy."""
    t=np.arange(n)
    price=100+0.02*t+5*np.sin(t/15)
    start=1_700_000_000_000//STEPS[timeframe]*STEPS[timeframe]
    times=start+t*STEPS[timeframe]
    return pd.DataFrame({"open_time_ms":times,"open":price,"high":price+1,
                         "low":price-1,"close":price+0.1*np.sin(t/3),
                         "base_volume":np.ones(n)*100,
                         "close_time_ms":times+STEPS[timeframe]-1})


def _cfg(args):
    return Config(timeframe=args.interval,starting_usdt=args.initial_cash,
                  fee_bps=args.fee_bps,slippage_bps=args.slippage_bps,
                  risk_pct=args.risk_pct,max_symbol_pct=args.max_symbol_pct,
                  max_total_pct=args.max_total_pct,max_open_risk_pct=args.max_open_risk_pct,
                  max_drawdown_pct=args.max_drawdown_pct)


def main(argv=None):
    p=argparse.ArgumentParser(description="BTC spot strategy-neutral research and paper only; no live orders")
    sub=p.add_subparsers(dest="cmd",required=True)

    def common(s):
        s.add_argument("--interval",choices=tuple(STEPS),default="4h")
        s.add_argument("--initial-cash",type=float,default=1000)
        s.add_argument("--fee-bps",type=float,default=10)
        s.add_argument("--slippage-bps",type=float,default=5)
        s.add_argument("--risk-pct",type=float,default=1)
        s.add_argument("--max-symbol-pct",type=float,default=30)
        s.add_argument("--max-total-pct",type=float,default=50)
        s.add_argument("--max-open-risk-pct",type=float,default=2)
        s.add_argument("--max-drawdown-pct",type=float,default=12)

    data=sub.add_parser("data",help="Read-only BTC spot OHLCV downloader")
    data.add_argument("--interval",choices=tuple(STEPS),default="4h")
    data.add_argument("--days",type=int,default=365)
    data.add_argument("--source",choices=("binance","ccxt"),default="binance")
    data.add_argument("--format",choices=("csv","parquet"),default="csv")
    data.add_argument("--output-dir",default="data/market")

    bt=sub.add_parser("backtest",help="Explicit plugin required; NO DEFAULT strategy")
    common(bt)
    bt.add_argument("--strategy",required=True,help="trusted module:callable or path.py:callable")
    bt.add_argument("--days",type=int,default=365)
    bt.add_argument("--csv-dir")
    bt.add_argument("--parquet-dir")
    bt.add_argument("--source",choices=("binance","ccxt"),default="binance")
    bt.add_argument("--output-dir",default="outputs/backtest-framework")

    rolling=sub.add_parser("rolling",help="Chronological research for an explicitly loaded strategy")
    common(rolling)
    rolling.add_argument("--strategy",required=True)
    rolling.add_argument("--days",type=int,default=730)
    rolling.add_argument("--csv-dir")
    rolling.add_argument("--parquet-dir")
    rolling.add_argument("--source",choices=("binance","ccxt"),default="binance")
    rolling.add_argument("--train-days",type=int,default=180)
    rolling.add_argument("--test-days",type=int,default=60)
    rolling.add_argument("--step-days",type=int,default=60)
    rolling.add_argument("--output-dir",default="outputs/rolling-framework")

    paper=sub.add_parser("paper",help="No --strategy => flat observer, no trading")
    common(paper)
    paper.add_argument("--strategy",help="optional trusted explicit plugin; paper-only")
    paper.add_argument("--state",default="outputs/paper-framework-v3/state.json")
    paper.add_argument("--once",action="store_true")
    paper.add_argument("--poll-seconds",type=int,default=60)

    report=sub.add_parser("report",help="Report a v3 paper account state")
    report.add_argument("--state",default="outputs/paper-framework-v3/state.json")
    report.add_argument("--output-dir",default="outputs/paper-framework-v3/report")

    demo=sub.add_parser("demo",help="Offline flat-observer backtest smoke test; no trade signals")
    common(demo)
    demo.add_argument("--output-dir",default="outputs/demo-framework")
    args=p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd=="report":
        raw=json.loads(Path(args.state).read_text(encoding="utf-8"))
        if raw.get("schema_version")!=3:
            raise ValueError("Only v3 paper states supported; archive old states")
        cfg=Config.from_dict(raw["config"])
        print(json.dumps(write_report(Account.restore(raw["account"]),cfg,args.output_dir),indent=2))
        return
    if args.cmd=="data":
        df=fetch("BTC/USDT",args.interval,args.days,args.source)
        path=Path(args.output_dir)/f"BTCUSDT_{args.interval}.{args.format}"
        (save_csv if args.format=="csv" else save_parquet)(df,path)
        print(f"BTC/USDT: {len(df)} closed candles -> {path}")
        return
    cfg=_cfg(args)
    if args.cmd=="paper":
        strategy=resolve_strategy(args.strategy) if args.strategy else None
        log=Path(args.state).with_suffix(".log")
        log.parent.mkdir(parents=True,exist_ok=True)
        logging.getLogger().addHandler(RotatingFileHandler(log,maxBytes=1_000_000,backupCount=3))
        run_paper(cfg,args.state,signal_fn=strategy,strategy_ref=args.strategy,
                  once=args.once,poll_seconds=args.poll_seconds)
        return
    if args.cmd=="demo":
        print("SYNTHETIC FLAT OBSERVER SMOKE TEST - NO INVESTMENT SIGNIFICANCE")
        book=backtest({"BTC/USDT":synthetic(cfg.timeframe)},cfg,signal_fn=no_trades)
        print(json.dumps(write_report(book,cfg,args.output_dir),indent=2))
        return
    if args.csv_dir and args.parquet_dir:
        p.error("Choose either --csv-dir or --parquet-dir")
    strategy=resolve_strategy(args.strategy)
    frame=(load_csv(Path(args.csv_dir)/f"BTCUSDT_{cfg.timeframe}.csv","BTC/USDT",cfg.timeframe)
           if args.csv_dir else
           load_parquet(Path(args.parquet_dir)/f"BTCUSDT_{cfg.timeframe}.parquet","BTC/USDT",cfg.timeframe)
           if args.parquet_dir else fetch("BTC/USDT",cfg.timeframe,args.days,args.source))
    hist={"BTC/USDT":frame}
    if args.cmd=="backtest":
        book=backtest(hist,cfg,signal_fn=strategy,days=args.days)
        analysis=market_analysis(hist,cfg,book)
        result=write_report(book,cfg,args.output_dir)
        dest=Path(args.output_dir)
        (dest/"market_analysis.json").write_text(json.dumps(analysis,indent=2),encoding="utf-8")
        print(json.dumps({**result,**analysis},indent=2))
    elif args.cmd=="rolling":
        _,summary=rolling_validate(hist,cfg,signal_fn=strategy,strategy_ref=args.strategy,
                                   train_days=args.train_days,test_days=args.test_days,
                                   step_days=args.step_days,output_dir=args.output_dir)
        print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()