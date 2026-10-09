# Binance-BTC — BTC/ETH Spot Research v0.2.0

Python-based **read-only** BTC/USDT and ETH/USDT spot OHLCV research, EMA crossover backtests and forward-only simulated trading. **No live trading, order APIs, credentials or withdrawals.** The application reads public Binance spot endpoints only; CCXT is an optional market-data adapter.

## Ubuntu 24.04 quick start

```bash
sudo apt update && sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m btc_quant demo
python -m btc_quant backtest --symbols BTC/USDT ETH/USDT --days 365
python -m btc_quant backtest --mode independent --days 365
python -m btc_quant paper --once
python -m btc_quant paper --poll-seconds 60
python -m btc_quant report
```

To cache historical closed candles, run `python -m btc_quant data --days 365`. For offline backtesting, pass `--csv-dir data/market` after downloading. Optional columnar caching: `pip install pyarrow`, then `python -m btc_quant data --format parquet --days 365` and `python -m btc_quant backtest --parquet-dir data/market --days 365`. Files are `BTCUSDT_4h.csv` / `ETHUSDT_4h.csv`. Add optional CCXT adapter with `pip install ccxt` and `--source ccxt` (market data only). API access depends on your country, VPS network and Binance availability.

## Strategy assumptions

* Long-only cash spot; no leverage, shorting, perpetuals or funding fees.
* EMA9 cross EMA21 entry/exit using **previous completed candle**, executed at **next candle open**.
* ATR14 stop at 2×ATR, take profit at 3×ATR; pessimistic stop-first resolution if both crossed in the same OHLC bar. At open, gap-stop is filled at the gap opening price.
* Per-trade planned risk 1% equity; max per-symbol exposure 30%; max aggregate exposure 50%; max aggregate planned stop-risk 2%; portfolio 12% peak-to-trough drawdown halt **new entries only**. Protective exits stay active.
* Default maker/taker-agnostic assumed fees **10 bps each side** and slippage **5 bps each side**. Exchange-specific minimum order sizes, depth-dependent slippage, spread dynamics and outages are not fully modeled. No guarantee of fills or loss caps.
* On same timestamp BTC and ETH entry signals, reserve equal shares of available cash before sizing; symbols never double-spend cash.
* The shared-cash portfolio simulator marks both asset prices on the same clock; missing or incomplete candles cause rejection, not interpolation.
* The first 200 candles are warmup only. Independent mode uses separate 1,000 USDT accounts and is **not** a portfolio P&L measure.

## Paper safety and recovery

Paper mode uses public Binance closed candles and public current prices. It only generates forward signals: first tick initializes cursors, repeated ticks do not execute again, missed bars skip stale signals. A ticker observation is not a guaranteed execution price; prices may gap between 60-second polls, including past stops. Market-data failure prevents state commits; stale feeds fail closed. The v2 state is distinct from legacy `outputs/paper/state.json`, and changes to configuration require a new state path. Keep a copy of `outputs/paper-v2/state.json` before changing versions. One instance per state path is enforced with a lock file.

## Output and testing

Each backtest produces `metrics.json`, `trades.csv`, `events.csv`, `equity.csv`, `equity_drawdown.png` under `outputs/`. Paper reports use `python -m btc_quant report`. Metrics include net P&L, peak drawdown, win rate, profit factor, Sharpe/Sortino estimates, annualized volatility, trading fees and realized P&L by asset. Historical backtests also produce `market_analysis.json` with close-to-close return correlations and a fee/slippage-adjusted, 50%-invested equal-weight buy-and-hold benchmark (no exit fee). Unrealized P&L is included in portfolio equity but not realized trade statistics. Synthetic demo is solely a smoke test and is not a profit forecast.

Run `python -m pytest -q` before deploying. This project is educational research, **not investment advice**. Backtests are hypothetical and may not resemble executable results.
