# BTCUSDT spot 4h — six bullish additions and OR-ensemble study

**RESEARCH ONLY — no orders, no defaults, no branch merge.**
Script: \`experiments/bullish_pattern_ensembles.py\`.
Successful CI: https://github.com/Aprivity/Binance-BTC/actions/runs/38061763035.
Artifact \`btc-bullish-pattern-ensembles\` (ID 11674010619):
- \`all_bullish_method_comparison.csv\`
- \`per_trade_audit.csv\`
- \`yearly_performance.csv\`
- \`signal_replacement_diagnostics.csv\`
- \`winners.json\`
- \`methodology.json\`

Original unchanged baseline verified exactly: **53** nonoverlapping 4h
Morning Star events 2020–23; **44** 2024–26 (ending 2026-10-09 UTC).
**14,843** genuine 4h spot OHLCV bars. All candidate signals fully
past-only and full future 48h OHLC available in same continuous segment;
unknown missing bars excluded; original patterns shared from existing
\`experiments.candlestick_study.signals\`.

## Rules held fixed across all 17 arms

Next 4h open after confirmation; 14-bar Wilder ATR measured at signal
confirmation; static initial stop at simulated buy fill minus 1.5×ATR,
+5% take relative to buy fill, max48h, **30% current equity** allocated
for each trade. Fee10bp and slippage5bp at EACH side. Conservative stop
first when high/low crosses stop+take in same bar. All OR-union signals
are processed strictly chronologically with full 48h **reserved**
window, even when position stopped/took profit sooner, and missed signals
are never backfilled. This means an added earlier signal may displace a
later otherwise profitable original Morning Star signal.

Six explicit alternative patterns:
- Existing original study's bullish engulfing and bullish harami.
- Bullish hammer with downtrend, green close, body≤35% full range,
  lower wick≥2×body and≥50% full range; upper wick≤25% range;
  close in top35% of range.
- Bullish piercing line: preceding wide bearish body≥50% range and
  downtrend; confirming green body≥40% range, opens≤previous close,
  closes > previous bearish-body midpoint and < bearish open.
- Three White Soldiers: prior decline, three consecutive strong
  rising green closes, bodies≥40% of own candle ranges, upper
  shadows≤25% and late opens within previous candle body.
- Existing original study's 20-bar high breakout.

## Complete comparison

Cumulative compounded ACCOUNT-price returns (NOT annualized). Column
'2024–26/month' is all 34 months including months without trades:

| Method | 2020–23 n | 2020–23 account % | 2024–26 n | 2024–26 trades/mo | 2024–26 account % |
|---|---:|---:|---:|---:|---:|
| **Original strict Morning Star** | 53 | **+11.2231** | 44 | 1.294 | **+16.2238** |
| Engulfing alone | 171 | +7.2371 | 152 | 4.471 | -15.5754 |
| Harami alone | 273 | -16.0888 | 187 | 5.500 | -10.6752 |
| Hammer alone | 172 | -20.3377 | 121 | 3.559 | -14.5555 |
| Piercing alone | 94 | -6.5790 | 69 | 2.029 | +2.4168 |
| Three White Soldiers alone | 7 | -0.7132 | 6 | 0.176 | +0.9522 |
| 20-bar Breakout alone | 198 | -17.1727 | 136 | 4.000 | -3.5231 |
| Original OR Engulfing | 207 | **+12.1892** | 180 | 5.294 | +0.6511 |
| Original OR Harami | 284 | -13.7278 | 199 | 5.853 | -5.0103 |
| Original OR Hammer | 202 | -20.8098 | 145 | 4.265 | -2.9964 |
| **Original OR Piercing** | 137 | +7.9878 | 105 | **3.088** | **+19.4834** |
| Original OR Three White Soldiers | 59 | +9.1188 | 49 | 1.441 | +15.6862 |
| Original OR Breakout | 225 | -4.7044 | 162 | 4.765 | +7.3803 |
| Original OR Engulfing OR Hammer | 314 | -11.3281 | 237 | 6.971 | -8.0208 |
| Original OR Harami OR Piercing | 323 | -5.8654 | 222 | 6.529 | -4.5918 |
| Original OR Soldiers OR Breakout | 228 | -5.5357 | 165 | 4.853 | +6.7586 |
| **Original OR all six extras** | 482 | -30.1285 | 346 | 10.176 | -20.5619 |

## Does Piercing *really* add value?

2024–26 original OR Piercing has 105 trades, ~3.09/mo, +19.4834%
compounded equity, 57.14% win, position-average net +0.5762% and
trade-close account MDD **-2.8046%** (vs original 44 trades +16.2238%,
61.36% win, net +1.1544%, MDD -2.6807%).

Of the 105 chronological 2024–26 accepted signals:
- 37 are from original independent Morning Star cohort;
- 68 are newly accepted non-original time slots;
- **7 original Morning Star signals are displaced** by preceding
  Piercing events under the shared 48h reservation. Such trade
  displacement cannot be erased via "preserve future original signal"
  without hindsight.
- The later-history total benefit is only +3.2596 percentage points
  cumulative, while development performance was WORSE
  (+7.9878% vs +11.2231%).
- Paired all-calendar-month average log-return improvement in inspected
  later history: **+0.08135 percentage points / month**;
  exploratory bootstrap95 **[-0.32418,+0.46618]** and
  unadjusted month sign-flip **p≈0.6842**. No reliable evidence that
  the improvement is greater than noise. Bootstrap and p-value do
  not correct selection among 17 tested arms; this period repeatedly
  used for prior research and NOT untouched OOS.

## Development selection overfit warning

Selecting **highest compounded account gain in 2020–23** amongst the
17 gives Original OR Engulfing **+12.1892%**, just +0.9661 pp
above Original. Same unchanged method in 2024–26 plunges to only
**+0.6511%**, compared to Original alone +16.2238%.
Engulfing OR new entries displace 8 original Morning Stars in
2024–26, add 144 other signals. Trade-close drawdown worsens
from -2.6807% (original) to -8.9763%. Thus choosing the
best development arm fails for already-inspected subsequent years.

## Recommendation

**DO NOT modify the existing Morning Star forward paper program or
enable any new OR entry pattern yet.**

The only potentially interesting higher-frequency candidate from
this batch is Original OR Piercing, and it is a prospective research
hypothesis, not validated performance. Freeze precise candle definition,
48h reservation, ATR1.5/TP5/30% rule and run a SECOND separate local
virtual forward ledger alongside original when engineering capacity
allows; do not invest real funds or claim reliable improvement.

A better next test is an *unseen forward period* with robust 1m/tick
execution replay and timestamp-gap checks. There is no evidence that
increasing trades by itself improves PnL, and the all-pattern arm
generates ~10/mo but loses >20% in later historical phase.

**Methodological caveats:** Multiple comparisons, full hindsight use
of 2024–26 despite phase labels, possible low individual pattern
sample sizes, 4h OHLC intrabar execution ambiguity, fixed hypothetical
fees/slippage, no unrealized equity-drawdown accounting. This is
not independently verified future trading profitability.
