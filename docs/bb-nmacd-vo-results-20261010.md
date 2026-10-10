# BTC 15m BB + NMACD + VO — first research results (2026-10-10)

Status: **REJECTED for forward activation** based on initial locked-rule backtest.
History is not an independently untouched test, and video author results were not
verified. No trading orders, leverage, credentials, or live execution.

## Assumptions

- Public Binance Vision BTCUSDT USD-M perpetual 15m bars and historical funding,
  initial 1000 USDT, long+short, gross exposure <=1x, modeled risk <=1% equity
  per trade at specified stop including modeled fees/slippage.
- 5 bps fee / side, 2 bps adverse slippage / side, funding included by signed
  settlement cashflow approximated at bar OPEN notional.
- Signal after confirmed candle; next-bar OPEN fill, signal candle low/high stop.
  Conservative stop-first when intrabar stop and take both touched.
- Three frozen exits: fixed 1.5R, fixed 2R, 50%-50% split at 1.5R & 2R.
  Not equivalent to the video's discretionary partial exits + trailing runner.
- Strict body-crosses-BB-middle interpretation; video NMACD and VO script
  identifications still require side-by-side TradingView confirmation.
- Missing price bars, suspicious funding time alignment (>1s) or funding data
  gaps cause hard failure. Binance Vision funding settlement timestamp
  precision of exactly +1 ms is normalized to candle boundary.

## Historic Jan 2024 smoke test

Source: https://github.com/Aprivity/Binance-BTC/actions/runs/38066251551

Period [2024-01-01, 2024-02-01) UTC. 3456 bars (warmup included),
108 funding events. 40 long signals and 45 short signals.

| Mode | Net return | Trades | Win rate | Profit factor | Max drawdown |
|------|-----------:|-------:|---------:|--------------:|-------------:|
| Fixed 1.5R | -3.3096% | 72 | 48.6111% | 0.85172 | 8.0925% |
| Fixed 2R | -0.8918% | 67 | 41.7910% | 0.96245 | 6.3917% |
| 50/50 split | -2.6425% | 67 | 43.2836% | 0.87993 | 7.2455% |

Jan 2024 fees totaled 65.70 / 62.04 / 61.32 USDT across the modes,
respectively. This is a model result, not actual exchange trading cost.

## Broad 2022–Sep 2026 research run

Source: https://github.com/Aprivity/Binance-BTC/actions/runs/38066347481

Period [2022-01-01, 2026-10-01) UTC. 166944 bars (warmup included),
5217 funding events. 2938 long signals and 2963 short signals.

| Mode | Final USDT | Net return | Trades | Win rate | Profit factor | Max drawdown |
|------|-----------:|-----------:|-------:|---------:|--------------:|-------------:|
| Fixed 1.5R | 9.985680 | -99.0014% | 3908 | 39.6366% | 0.69944 | 99.0032% |
| Fixed 2R | 9.988717 | -99.0011% | 3666 | 32.6241% | 0.72609 | 99.0016% |
| 50/50 split | 9.971587 | -99.0028% | 3687 | 35.6116% | 0.69398 | 99.0040% |

At final equity under 10 USDT, the model can no longer meet its minimum
10 USDT opening notional; differences between final balances are not
meaningful strategy rankings. The unlevered monthly and full-period account
results are both negative. A profit factor <1 across all tested modes does
not support a positive historical gross-to-net trading edge.

## Limitations / next validation gates

- TradingView script values have not been independently matched candle-by-
  candle; this is a precise, reproducible *interpretation*, not proof of
  identical video signals and fills.
- The ledger models synthetic 1x USD-M price P&L, not actual perpetual
  margin, order-book depth, liquidation, mark prices or tick-level fills.
- Funding cashflow uses actual archived rates but bar-open price proxies.
- Serial compounding until the account becomes too small can obscure annual
  trade-regime comparisons. For diagnostic comparisons use independent annual
  resets and per-side net expectancy, not final balance alone.
- Changing parameters repeatedly on already-inspected 2022–2026 history is
  exploratory; genuinely new market data are needed for independent validation.
- **Do not activate in paper or live trading** on these results. If continuing
  research, first validate exact indicator equivalence and run fixed-notional
  or independent-year diagnostic decomposition (long/short, year, costs).

Artifacts with trades, equity series and summary JSON were uploaded to both
GitHub Actions runs (retained for 30 days by workflow settings).
