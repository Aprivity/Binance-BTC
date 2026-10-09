# Binance-BTC — BTC/ETH Spot Research v0.2.2

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


## v0.2.1: Four-strategy, fee-aware chronological holdout research (NEW)

**No changes to the default live-running `paper` command**: it still runs the original EMA9/21 signals with precisely the same long-only, no-leverage risk-management execution rules. `compare` is offline/historical research only; it NEVER places orders and does not require authentication.

Four fixed rule sets (not optimized automatically):

| Name | Entry (using completed candle) | Signal exit (protective 2 ATR stop / 3 ATR take remain) |
| --- | --- | --- |
| `ema` | EMA9 crosses over EMA21 | EMA9 crosses below EMA21 (v0.2.0 benchmark) |
| `triple_ema` | Same golden cross AND close > EMA200 | Original EMA9/21 death cross |
| `donchian` | Close crosses **above previous** 20-candle highest high | Close crosses below previous 10-candle lowest low |
| `supertrend_cci` | SuperTrend(ATR14 × 3) bullish and CCI(20) > 0, condition changes false→true | SuperTrend bearish OR CCI(20) < -100, condition changes false→true |

Signals determined on bar `t` are executed at bar `t+1` open. The same existing `Account` and bracket exit engine handles all strategies, with default **0.10% per side fees + 0.05% per side slippage**, individual 30% caps, aggregate 50% caps, 1% planned per-trade risk, 2% planned overall stop risk, and 12% new-entry drawdown halt. These rules do **not** guarantee actual fills or a maximum real-world loss.

### Ubuntu usage

From the project root with its Python virtual environment active:

```bash
python -m pytest -q
# Obtain longer BTC/ETH samples; 730 days is a useful starting exploration window.
python -m btc_quant data --symbols BTC/USDT ETH/USDT --interval 4h --days 730
# Reuse precisely the same data for all comparisons, no repeated online fetches.
python -m btc_quant compare --symbols BTC/USDT ETH/USDT \
  --csv-dir data/market --days 730 --train-ratio 0.7 --scope all
# Or fetch public historical candles automatically (subject to Binance access).
python -m btc_quant compare --days 730 --scope portfolio
```

`--scope all` compares BTC-only, ETH-only and shared-cash BTC+ETH portfolio; each independent account resets to 1,000 USDT. `--scope portfolio` tests just the shared portfolio; `--scope independent` runs each standalone coin. Override fee assumptions with `--fee-bps 20 --slippage-bps 10`; they apply **equally to all strategies**. Compare a subset via `--strategies ema donchian`. All costs and limits are frozen across candidates in a given run. Supported timeframes remain `1h`, `4h`, `1d` (default `4h`).

Output directory defaults to `outputs/compare-v021`:

* `compare.csv`: for each scope × strategy, records train/test/full returns, maximum drawdown, trades, win rate, profit factor, Sharpe/Sortino, realized PnL by coin, fees, estimated slippage, and baseline comparisons.
* `methodology.json`: fixed strategy descriptions, UTC train/test boundary, reporting caveats.
* `oos_comparison.png`: side-by-side out-of-sample returns and max drawdowns, with risk-cap-matched buy-and-hold.
* `oos_equity.png`: test-segment shared-portfolio capital curves (when included).

### Historical test design and benchmark caveats

* Chronological first ~70% **development** and last ~30% **holdout**, after 200 candles of indicator warmup. No shuffle or rolling cross-validation yet. At least 500 aligned candles and 300 after warmup are required.
* Training and test use **separate fresh accounts**, starting at the configured capital, so results are independent runs, NOT returns from one seamlessly invested portfolio. A test cannot execute a signal from the prior training candle. Indicator warmup may read earlier completed bars (which are available at that time).
* `full` is a separate uninterrupted run for descriptive context; **do not combine train + test percentages arithmetically**.
* The fixed buy-and-hold comparison respects **the same max exposure policy** as the strategies (one coin 30%, two coins 25% each). It pays model entry costs but does not pay an unrealized terminal exit fee. As a passive benchmark, it remains invested at its initial weights; it does NOT match the strategy's timing.
* An additional `ex_post_exposure_hold_return_pct` uses the strategy's **realized future average asset exposure**, so it is a diagnostic only and would NOT have been investable in advance. Do not use it for strategy selection.
* This comparison **does not search parameter grids**, implement walk-forward retraining, provide trading recommendations, or validate market performance outside your downloaded data. Reviewing the holdout and repeatedly changing strategies would invalidate that holdout. For further iteration, reserve newer unseen data.
* Data are public spot OHLCV; **no trading keys, order placement, futures, shorting, or leverage**. Trading fees and fills are estimates. Always inspect `trades.csv` and a truly untouched later window before drawing conclusions.


## BTC-only 4h single-factor intraday research (v0.2.2)

This module is **research-only**. Existing `backtest`, `compare`, and `paper` commands
are unchanged; `paper` continues to use the legacy EMA9/21 signals. No live orders,
leverage, shorts, API keys or exchange account access are implemented.

With locally downloaded BTC 4-hour *closed* candles (`data/market/BTCUSDT_4h.csv`):

```bash
python -m btc_quant intraday-study \
  --symbols BTC/USDT --interval 4h \
  --csv-dir data/market \
  --train-days 180 --test-days 60 --step-days 60 \
  --output-dir outputs/btc-intraday-v022
```

Five mutually exclusive research arms, one changed factor versus A each:

- `A_base`: original EMA9/21 cross above EMA200 + original ATR exits.
- `B_slope`: A's entries additionally require `EMA200[t] > EMA200[t-6]`.
- `C_adx`: A's entries additionally require Wilder-style `ADX14 > 20`.
- `D_pullback`: **only the entry** changes to a trend-aligned pullback: EMA9>EMA21,
  price>EMA200, previous bar low touching EMA21, green rising bar closing above EMA21.
  Entries trigger at the next 4h open; exits are identical to A.
- `E_trailing`: A's entry remains unchanged; its fixed 3×ATR take-profit is
  replaced by a **3×ATR trailing stop**, updated *after* each completed candle,
  active no earlier than the next candle. Original 2×ATR initial stop remains.

All five arms share the same `Account` cash/position sizing, entry/exit execution,
0.10% per-side fee, 0.05% per-side slippage, 1%-risk sizing, exposure caps, and
12% drawdown entry halt. Those settings are inherited from the CLI common flags;
not optimized. Simultaneous stop/target touches follow existing pessimistic
bar-order assumptions. Entry/exit fill results are simulated, not guaranteed.

Reports saved to `outputs/btc-intraday-v022/`:

- `fold_metrics.csv`: per-fold train/test return, drawdown, fees and **halted_at_utc**.
- `oos_summary.csv`: per-arm reset-window comparison, worst fold and halt counts.
- `trade_diagnostics.csv`: closed-trade timestamps, actual exit reason, fees-included
  net P&L, holding hours, bar-observable MFE/MAE (entry-relative %).
- `fold_equity.csv`: separately reset account equity and drawdown paths.
- `oos_fold_returns.png`: each arm's chronological *test-fold* net returns.
- `methodology.json`: explicit fixed settings, sampling, guardrails and caveats.

**Interpretation:** each train/test fold starts a *fresh* simulated cash account,
never carries forward cash, risk halt or open positions, and never executes an
entry signal from the preceding window. Folds use prior historical candles only
to warm up indicators. Test windows do not overlap (`step_days >= test_days`).
This is rolling *historical* evaluation, not a single continuously traded equity
curve or an untouched out-of-sample validation (we have already inspected past
periods). Do **not** compound reset-fold returns into a portfolio return.
MFE/MAE use fully completed bars only, excluding unknown path extremes on the
intrabar exit candle; they may therefore understate the true excursions.

Run checks with `python -m pytest -q`. The synthetic smoke test data are not
investment-performance evidence.


## v0.2.3 BTC-only 4h cost robustness (research-only)

Run after v0.2.2 historical data caching. Uses the exact same A/B/C/D/E
rules, **no combination**, and never modifies the `paper` simulator:

```bash
python -m btc_quant.robustness \
  --csv-dir data/market \
  --train-days 180 --test-days 60 --step-days 60 \
  --output-dir outputs/btc-robustness-v023
```

The script runs all five arms on the same non-overlapping chronological
60-day fresh-account test windows under three **pre-specified** *per-side*
fee/slippage assumptions: `zero_cost_upper_bound` (0/0 basis points,
**not executable**), `configured` (default 10/5 bps), and `stress_2x`
(default 20/10 bps). It runs the execution engine anew for each scenario:
account sizing and minimum-notional decisions may change with fees.

Files:
- `cost_fold_comparison.csv`: one row per scenario/fold/arm; original
  mark-to-market net return, hypothetical final-liquidation-adjusted return,
  remaining open position count, separate fees/slippage, drawdown, first
  risk halt, and paired percentage-point improvement versus A in the *same*
  fold and cost scenario.
- `cost_paired_summary.csv`: mean/median liquidation-adjusted return and
  paired differences, improved/worse/tied fold counts, no-closed-trade fold
  count, closed trades, worst fold and worst drawdown.
- `methodology.json`: fixed scenarios, existing sizing/risk settings and
  explicit limitations.

`liquidation_net_return_pct` is a **diagnostic**: it marks an open position at
the last known close, then subtracts assumed exit-side fee and slippage once.
It neither executes liquidation nor accounts for future overnight gaps,
liquidity, variable spreads or unavailable order types. It MUST NOT be
combined with mark-to-market return as additional profit or loss.

v0.2.3 also adds `exit_reason_detail` to `intraday_study`'s detailed trade
records, distinguishing original stop from a subsequently activated ATR
trailing stop (`initial_stop`, `trailing_stop`, with `_gap` when appropriate).
Existing `reason`, `exit_reason`, trade fills, account state and paper behavior
remain unchanged. **All previous trade P&L remains comparable.**

These are **exploratory** tests on repeatedly viewed historical data. There
are only nine ~60-day windows in the user example; those windows are not
independent samples from a stationary return distribution, and pairing does
not establish statistical significance or forward profitability. Do not pick
a winning parameter based on the same test folds and call them untouched
OOS. Retain a wholly future period for forward paper validation. Each fold
starts a fresh account; do not compound the fold returns as a live portfolio.
