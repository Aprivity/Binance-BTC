# 200U BTC dual moving average research (not live trading)

Video being investigated: https://www.youtube.com/watch?v=F2FJeRsUzNE

The publicly verified title is a 200 USDT short-term trading video about
"double moving average". The actual numerical MA periods and speaker's exact
execution rules were not recovered. This code is a fully disclosed
researcher-defined BASELINE, NOT a precise reproduction of the presenter.

This branch is separate from the unmerged six-MA video study (PR #7) and is
based on the unmerged BTC-only strategy-neutral framework (PR #6).
Nothing changes main, and there is still NO DEFAULT STRATEGY.

## Candidate rules

Five independently tested, closed-bar, long-only spot hypotheses:
- EMA 9/21 cross-to-long, drop-below-to-flat.
- EMA 9/21 requiring two consecutive confirmed closed candles.
- EMA 12/26.
- EMA 20/60.
- SMA 20/60.

No future data is read. If a new valid long signal is generated after a bar
close, a simulated buy is attempted at the NEXT bar's open. Fast MA below slow
MA creates a flat signal. Protective 2xATR stop and 3xATR take are existing
framework modeling choices, NOT video-sourced rules. OHLC ambiguous bars assume
stop first. Execution prices are hypothetical.

## Capital and risk assumptions

BTC/USDT **spot only**. The 200 USDT amount is CASH, not futures margin.
No shorts, leverage, liquidation engine, borrowing or funding-rate modeling.
No authenticated APIs, no exchange credentials, no live orders.

Model: starting 200 USDT, 1% equity risk per new position, 2% open-stop-risk
cap, up to 100% spot-cash exposure, 12% max equity drawdown blocks new entries,
minimum notional 10 USDT, ATR14, 2ATR stop and 3ATR take.

Compare three cost cases: 0/0 bps (optimistic impossible upper bound),
10/5 bps per side (base fee/slippage), 20/10 bps per side (stress).
Actual user VIP/taker/maker fees may differ. No taxes/market impact modeled.

## How to run

Run from repo root (not production or a trading server):

~~~shell
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Public REST data only; one year on 5m/15m/1h/4h separately
python -m research.dual_ma.study --interval 5m --days 365
python -m research.dual_ma.study --interval 15m --days 365
python -m research.dual_ma.study --interval 1h --days 365
python -m research.dual_ma.study --interval 4h --days 365

# Existing contiguous Binance spot OHLCV CSV works offline:
python -m research.dual_ma.study --interval 15m --csv-dir data/market

# Explicit strategy example; shared CLI risks must be specified independently:
python -m btc_quant backtest --strategy research_strategies.dual_ma:ema_9_21 \
  --interval 15m --initial-cash 200 --max-symbol-pct 100 \
  --max-total-pct 100 --days 365
~~~

If public Binance data access is geoblocked, use an existing strictly
contiguous UTC CSV. The study fails on gaps rather than filling them.

## Evaluation

Each timeframe is tested independently, with 200 warmup bars then a fixed
chronological 70% development / 30% held-out split, with freshly reset
accounts and no carry-in boundary signal. Five fixed rule variants times
three cost cases times two phases produce 30 comparable reports per
timeframe. Parameters are not selected/fitted in the runner.

Generated files in outputs/dual-ma-200u/<interval>:
- costed_results.csv: mark-to-market net return, realized pnl, counts,
  win rate, drawdown, fees, slippage, benchmark at matched cash exposure.
- trades/*.csv: simulated completed trades per exact case and stage.
- methodology.json: BTC source, timestamp boundaries, input-data SHA256,
  cost assumptions, limitations and full configuration.
- README.md: primary OOS 10/5bps scenario results (no strategy selected).

A separate GitHub Actions workflow runs the same four timeframes against
public Binance completed candles and preserves output artifacts.

Interpretation caution: Net equity of a position still open at the
historical cutoff is marked to market without a modeled terminal exit fee.
A fixed target is not guaranteed to be hit. 5 candidates, 4 timeframes and
3 fee cases invite multiple-testing selection bias. Historical held-out
ranges become contaminated by inspection. Results are EXPLORATORY, not
proof of a profitable trading edge. Before any paper forward-test, inspect
trade counts, double-cost robustness, bad regime behavior and genuinely
new future data. Do not merge to main without user review.
