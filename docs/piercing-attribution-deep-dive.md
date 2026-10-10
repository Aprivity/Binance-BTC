# BTC 4h Piercing Line incremental-return audit (research only)

**Status: analysis only; do not merge or activate on original local spot paper.**
Code: \`experiments/piercing_attribution.py\`
Successful actions: https://github.com/Aprivity/Binance-BTC/actions/runs/38063317069
Artifact \`btc-piercing-attribution\` (ID 11672969856). Contains:
\`pattern_source_performance.csv\`, \`account_log_return_decomposition.csv\`,
\`yearly_source_performance.csv\`, \`prepattern_regime_trade_details.csv\`,
\`prepattern_regime_summary.csv\`, \`fee_slippage_stress.csv\`,
\`signal_source_trade_audit.csv\`, and \`methodology.json\`.

## Definitions and experiment scope

- **14,843** authentic Binance BTCUSDT spot **4h** OHLCV candles, one
  feed gap excluded from signals across it, no interpolation.
- 2020–23 descriptive development vs 2024–Oct9 2026 *previously
  inspected* period; NO truly untouched out-of-sample.
- Original three-candle Morning Star unchanged; extra signal is
  **two-candle BTC-adapted bullish Piercing Line**: previous wide bearish
  real body >=50% bar range with downtrend from 4 bars prior, next
  strong green real body >=40% its range, opens no higher than prior close,
  closes above 50% prior bearish body but below previous open.
- All orders **hypothetical spot only**: next-bar-open simulated entry
  following a fully closed signal bar; static initial Wilder ATR14×1.5
  stop, +5% take, max 48h; fixed 30% of current account equity, fee
  10bps+slippage 5bps per side. OHLC ambiguity stop-first. Single
  chronological 48h reservation even if earlier exit. No live or
  paper orders emitted. Original forward paper program unchanged.

## Historical account results

| Strategy | 2020–23 trades | 2020–23 account % | 2024–Oct2026 trades | 2024–Oct2026 account % |
|---|---:|---:|---:|---:|
| Original Morning Star | 53 | **+11.2231%** | 44 | **+16.2238%** |
| Piercing alone | 94 | **-6.5790%** | 69 | **+2.4168%** |
| Chronological Morning Star OR Piercing | 137 | **+7.9878%** | 105 | **+19.4834%** |

The combo yields ~**3.09 signals/month** in the 34-month recent
history versus 1.29/month baseline. A higher-frequency signal is NOT a
reliable profit multiplier: combo early-history trade-close equity MDD
-17.7862% versus -6.3482% baseline, and the recent +3.2596 percentage
point cum-return advantage is exploratory (the previous all-calendar-month
bootstrap95 paired monthly log growth diff spans 0, unadjusted p≈0.684).

### Exact log-account attribution (mathematically additive)

| Phase | Retained original | New Piercing | Displaced originals | New Piercing log equity pts | Saved/forgone originals log equity pts | Net incremental log equity pts |
|---|---:|---:|---:|---:|---:|---:|
| 2020–23 | 47 | 90 | 6 | **-7.5081** | **+4.5560** (displaced trades were losers) | **-2.9520** |
| 2024–26 | 37 | 68 | 7 | **+4.9005** | **-2.1345** (displaced trades were winners) | **+2.7660** |

Sums reflect account \`sum(log(1+0.3*position_net_return))\`,
NOT directly additive arithmetic return percentages. Compounding
produces +11.2231→+7.9878% and +16.2238→+19.4834% respectively.

**Important detail:** of the 68 recent newly accepted Piercing trades
(not the entire combo), the mean position net return is only **+0.2491%**
and win rate **54.41%**, with **8 TP / 24 stops / 36 time exits**. At
the unchanged 30% account allocation the newly accepted subset on its
own compounds **+5.0225%** and has trade-close MDD **-4.3761%**.
In early history the 90 newly accepted trades had **41.11%** win rate,
mean position net **-0.2616%**, 18 TP / 39 stops / 33 time, and
standalone account compound **-7.2331%**, trade-close MDD **-16.5185%**.
Therefore the newer overall positive combo is NOT evidence of a robust
standalone Piercing alpha.

### Annual model ACCOUNT returns (not annualized)

| Calendar year | Morning Star alone | Combined Piercing OR Morning Star |
|---|---:|---:|
| 2020 | +7.2856% | +9.5369% |
| 2021 | +1.7966% | +5.1600% |
| 2022 | +0.6968% | **-9.7815%** |
| 2023 | +1.1358% | +3.9125% |
| 2024 | **+9.3027%** | +6.4373% |
| 2025 | +3.8384% | +7.0829% |
| 2026 (through 2026-10-09) | +2.4014% | +4.8320% |

2022 bearish extended selloff is the clearest adverse regime in this
specific sample. 2024 combo was ALSO worse than baseline despite the
overall 2024–2026 positive delta; no monotonic annual improvement.

### Price/volatility execution sensitivity

Replayed the **same frozen signals and 4h OHLC stop-first bracket paths**,
recomputing stop/take from simulated entry fill under stress assumptions.
Per-side prices are hypothetical and no actual exchange fills guaranteed:

| Side fee | Side slippage | Original recent return | Combined recent return | Difference combined-minus-original |
|---|---|---:|---:|---:|
| **10bps** | **5bps** | **+16.2238%** | **+19.4834%** | **+3.2596 pp** |
| 10bps | 10bps | +15.1967% | +16.7550% | +1.5583 pp |
| 20bps | 10bps | +12.1755% | +9.6079% | **-2.5676 pp** |

Early-phase combined returns under the same three stress settings were
+7.9878%, +4.4461%, and -3.7954%, vs original
+11.2231%, +10.0490% and +6.5931%. The denser signal stream is
more sensitive to turnover costs.

### Historical context before pattern (NOT an activated filter)

For the *NEW* Piercing signals accepted into the combined strategy,
the **prepattern 7-day BTC close return** is measured as close[t-2] /
close[t-44] -1 (exclude both candles of Piercing; only historical
completed bars). Groups are naturally positive versus nonpositive:

| Phase | Prior 7d BTC trend | New Piercing count | Win rate | Mean POSITION net |
|---|---|---:|---:|---:|
| 2020–23 | Positive | 46 | 45.65% | **+0.2389%** |
| 2020–23 | Flat/negative | 44 | 36.36% | **-0.7849%** |
| 2024–26 | Positive | 31 | 64.52% | **+0.7033%** |
| 2024–26 | Flat/negative | 37 | 45.95% | **-0.1315%** |

**Exploratory hypothesis:** this two-candle rebound may function better
as a short-term *bullish trend pullback entry* rather than a reliable
knife-catching signal in a declining 7-day market. These buckets were
examined AFTER seeing price outcomes, not pre-registered or independently
validated; do not convert the 7-day return sign into a deployed entry
filter just because retrospective subgroup numbers look good. The
30-day trend split DID NOT preserve the same ordering recently, further
limiting causal conclusions.

## Final decision

Keep existing Morning Star forward spot paper intact. Piercing OR
Morning Star is a **new, separate candidate shadow paper portfolio only**
with precisely frozen candlestick and cost parameters and a prospective
start after the historical sample ends. Compare monthly paired net
returns, trade-close AND marked-to-market drawdown, overlapping
signals, network gaps, and realized/quoted slippage on never-seen data.
No leverage, shorting, testnet/live order entry or strategy migration.

2024–26 repeatedly used for model selection; any probability/statistics
above are descriptive and subject to multiple comparisons, execution
ambiguity in 4h OHLC and selection bias.
