# BTCUSDT U-M futures: Evening Star short versus Morning Star long, frozen 4h

Status: **historical research only**, not a live recommendation, no orders.
Runtime: https://github.com/Aprivity/Binance-BTC/actions/runs/38055365937
Artifact: \`btc-futures-eveningstar-short-vs-long\`
Study source: \`experiments/eveningstar_short_futures_comparison.py\`.

## Selection result

Across ONE frozen long variant, EIGHT short variants and EIGHT nonoverlapping
combined variants (**17 models**), the highest observed **2020–2023
development** model ACCOUNT price-trading return was **long-only original
three-candle Morning Star ATR14×1.5 / take +5% / max48h, 30%-of-equity notional**:
**+15.6408%** on 51 entries, win rate 60.78%, trade-close drawdown -5.2943%.
With NO retuning this same winner, the **2024–2026 inspected period** had
42 entries, 59.52% win, +12.8048% account price return, trade-close
MDD -2.7172%.

This is the highest-return method in BOTH historical phases amongst these
17 exact methods. The original SPOT long study had 53/44 signals and
+11.2231%/+16.2238%; do NOT conflate spot with futures return or quotes.

## Full 8 short-configuration comparison

All parameters unmentioned in cells unchanged: same inverted original
three-candle Evening Star shape, futures 4h, Wilder ATR14 at signal close,
next-bar-open entry, fixed 30% account notional, fee10bps + slip5bps BOTH
sides, **48h reserved signal cohort even in max24h variant**. 58 development
short entry signals and 53 later short signals for EVERY short configuration.

| Short stop | Short profit | Hold max | Account return 2020–23 | Account return 2024–26 |
|---|---|---|---:|---:|
| ATR×1.5 | down 3% | 24h | -8.7188% | -7.4892% |
| ATR×1.5 | down 3% | 48h | -10.0700% | -9.4017% |
| ATR×1.5 | down 5% | 24h | -6.9614% | -7.4014% |
| ATR×1.5 | down 5% | 48h | -7.3003% | -9.0578% |
| ATR×2.0 | down 3% | 24h | -10.7792% | -7.1473% |
| ATR×2.0 | down 3% | 48h | -9.9003% | -6.2439% |
| ATR×2.0 | down 5% | 24h | -9.0615% | -7.0591% |
| **ATR×2.0** | **down 5%** | **48h** | **-6.3941%** | **-5.6317%** |

Best consistent short *within the eight short-only historical choices*:
ATR14×2.0 STOP ABOVE sell entry, take PROFIT if BTC falls 5%, max48h,
fixed 30% notional, **still loses** in both historical phases.
Historical short wins 37.93% (2020–23) and 43.40% (2024–26), with
close-to-close account MDD -9.5023% and -7.3785%.

Most profitable COMBINED method in 2020–23 used the same frozen long side
plus shorts ATR1.5 / down5% TP / 24h: +5.5472% dev (100 trades),
+1.6621% later (86 trades). Highest *retrospective later-only* combined
uses short ATR2.0/down5%/24h: +2.0378% later but only +2.2574% earlier.
Neither comes close to long-only; a chronological 48h reservation causes
some original long signals to be displaced by earlier shorts.

## Market data, costs and deep restrictions

- **14844 authentic USD-M BTCUSDT 4h futures klines**, Jan1 2020 up to
  Oct10 2026 exclusive UTC, no gaps, 2025+ microsecond normalization
  on archive file timestamps. Futures OHLC refers to traded prices, not
  1-minute executed path nor mark-to-market liquidation path.
- Funding archive download: 7395 past settlement rate rows retrieved,
  but **9 October 2026 daily funding archive entries unavailable/invalid**.
  As completeness is NOT verified, **NO funding-adjusted return is claimed**.
  All headline net returns reflect ONLY 10bps/side fee and 5bps/side
  adverse price slippage; NOT futures funding, maintenance margin,
  liquidation, ADL, bankruptcy fees, real market depth or borrowing costs.
- PnL is 1× synthetic short/long notional of 30% of account equity per trade;
  no simulated levered PnL amplification and no binding margin engine.
  Short risk can exceed cash and is not capped by a non-guaranteed stop.
- Full historical 4h OHLC stop-first for ambiguous TP+SL within one bar,
  hypothetical next-candle open buy/sell; close-trade MDD excludes open-position
  peak-to-trough intratrade volatility.
- Historical model selection over repeatedly inspected 2024–26 LONG data
  is NOT out-of-sample evidence; the 2020–23 ranking is descriptive
  after a designed grid and long strategy had been chosen elsewhere.
  A future untouched period and actual futures funding/liquidation replay
  would be necessary before making any deployability claim.

## Final parameter choice for ongoing RESEARCH

**Keep original spot 4h Morning Star LONG 30% fixed allocation
ATR(14)×1.5 initial STOP, +5% TP, 48h cap, no relative-volume, HVN,
trend filters. DO NOT integrate a short signal into the active local
spot paper simulator.** If short-side research continues, the best
among this deliberately narrow, losing short grid is ATR14×2.0,
down5% take, 48h, 30% notional only as a research benchmark.
