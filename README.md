# Binance-BTC — BTC-only strategy-neutral research framework

A read-only BTC/USDT **spot** OHLCV downloader, deterministic historical backtest engine,
risk/fee/slippage model, reports, chronological rolling validation and **paper-only** account.
**This version bundles no EMA, ADX, Donchian, SuperTrend or other trading strategies.**

No exchange credentials, authenticated APIs, futures, leverage, shorting, withdrawals or **live order submission** exist.

## Set up (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m btc_quant demo                         # inert smoke test: 0 orders
python -m btc_quant data --interval 4h --days 730
python -m btc_quant paper --once                 # no strategy => observer, ZERO orders
```

## Explicit plugin contract

There is **no default strategy**. `backtest` and `rolling` refuse to start
unless the caller supplies a trusted Python strategy using `--strategy`.
A strategy is a function `signals(frame: pandas.DataFrame, cfg: Config) -> pandas.DataFrame`
which must return exactly one row per input candle with **boolean** `buy_signal`
and `sell_signal` columns, indexed identically to the input. It may compute
any indicators internally, but must not peek at future candles.

For a trusted strategy in `/path/to/new_strategy.py` exporting `signals`:

```bash
python -m btc_quant backtest \
  --strategy /path/to/new_strategy.py:signals \
  --csv-dir data/market --interval 4h --days 730
python -m btc_quant rolling \
  --strategy /path/to/new_strategy.py:signals \
  --csv-dir data/market --train-days 180 --test-days 60 --step-days 60
# OPTIONAL forward simulated trading; NEVER sends exchange orders:
python -m btc_quant paper --strategy /path/to/new_strategy.py:signals --once \
  --state outputs/paper-framework-v3/new-strategy-state.json
```

**External strategy files are executable Python. Review their code first.** The
plugin contract validates types/alignment, but cannot mathematically certify
an arbitrary plugin's absence of future-data leakage. A fresh code review and
untouched validation period are required for every strategy.

The 4h engine uses **previous closed-bar** signals and next-bar opening fills,
14-bar ATR protective stop/take (default 2x/3x), and 200 warmup bars. Default
long-only position risk is 1% equity; max BTC exposure is 30%, max total exposure
50%, aggregate planned stop risk 2%, 12% peak drawdown permanently blocks new
entries in the account. Exits remain active. Modeled cost: 10bps fee + 5bps
slippage per side. OHLC ambiguous stop vs take assumes stop first. Real fills
and stop limits are not guaranteed.

`rolling` resets accounts for each train/test fold and excludes the signal from
the prior fold at each fresh test start. Its folds **must not be compounded** as
a continuously invested real portfolio. Results on already-inspected historical
periods are exploratory, not independent confirmation. Outputs include
`rolling_folds.csv`, `rolling_summary.json`, and `rolling_returns.png`.

## Paper-state safety

Without `--strategy`, `paper` **only reads public BTC spot data**, marks an empty
account and saves observer timestamps: no positions or trade events. With a
trusted explicit strategy, it may simulate fills locally but never places orders.
Only new **schema v3** states are accepted. Old v2 paper files are **not auto-
migrated**. Archive previous `outputs/paper-v2/` and start a new state path;
a strategy reference and complete risk config are bound to each v3 state.
Forward mode bootstraps the candle cursor, never executes bootstrap signals,
skips offline gaps and prevents stale public feeds from committing state.
Observer mode is not a working live stop-loss service.

Historical BTC data under `data/market/` and reports under `outputs/` remain local
and ignored by Git. Old research PRs were closed, but their Git history is retained
on branches and in the PR records. No history is rewritten.