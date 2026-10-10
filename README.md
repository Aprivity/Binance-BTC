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

## Optional research plugin: video's six-MA compression and first pullback (NOT default)

Source: [video](https://www.youtube.com/watch?v=vdCdg4MdwBs), study based on
its full ~42-minute transcription. At 02:21 the creator specifies six lines:
**SMA(20,60,120)** and **EMA(20,60,120)**. At 13:27–20:20 they introduce
**two different entry methods**: (1) compression-zone breakout and successful
retest; (2) after dispersion, the **first successful MA20 pullback**. At
20:28–27:29 they describe three take-profit concepts: fixed reward/risk,
previous compression zone, or Fibonacci extension. At 27:29–29:04 they say
stops belong below the entry structure. The demonstrations include **SHORT
perpetual futures and leverage**, which are deliberately **not implemented**.

This branch adds TWO separate, explicitly selected, research-only long BTC spot
plugins. Neither is bundled as a default trading signal. It implements **only
the video's fixed 1:3 reward:risk exit alternative**, with next-bar entries and
hard price-bracket stops. Historical support-zone profit targets, Fibonacci
scale-outs, discretionary exits, shorting, leverage and actual orders are NOT
implemented. The current-price/wait-for-'effective-break' discretion is not
reconstructed: instead, the stop is a conservative executable price barrier.

**Critical interpretation:** The video does *not* define numerical thresholds
for "MA compression", "valid break", "retest holds", or how many bars a zone
remains valid. The following are **researcher-defined assumptions**, not the
creator's formulas or proven-optimal settings:

| Assumption | This exploratory implementation |
|---|---|
| Compression | (max of six MA/EMA − min) / close <= 1.2% for at least 3 bars |
| Zone | Frozen min/max across last 12 compression bars |
| Breakout | completed green candle closes > frozen zone top by 0.5% |
| Zone expiry | 24 bars after breakout |
| Retest | bar low <= zone top or MA20 plus a 0.2% tolerance; green close above reference |
| Initial stop | zone bottom or signal-bar MA20, 0.1% below reference |
| Take profit | 3R from *actual simulated entry fill* and declared stop |
| Execution | signal after fully closed bar, next candle's opening price |

A price gap below the planned long stop cancels the entry. Fees, slippage,
maximum allocation, drawdown halt and stop-first OHLC ambiguity remain under
the **unchanged core account risk rules**. The two plugins are independent:

```bash
# Use a trusted, audited source; does NOT put in real orders.
python -m btc_quant backtest \
  --strategy research_strategies/video_six_ma.py:compression_retest \
  --csv-dir data/market --interval 4h --days 730 \
  --output-dir outputs/video6ma-compression

python -m btc_quant backtest \
  --strategy research_strategies/video_six_ma.py:first_ma20_retest \
  --csv-dir data/market --interval 4h --days 730 \
  --output-dir outputs/video6ma-first-ma20

python -m btc_quant rolling \
  --strategy research_strategies/video_six_ma.py:compression_retest \
  --csv-dir data/market --interval 4h --train-days 180 \
  --test-days 60 --step-days 60 \
  --output-dir outputs/video6ma-compression-rolling
```

***Do not use this as proof of profitability.*** Synthetic tests only verify
signal causality and account behavior. Previously studied BTC price history is
exploratory rather than an untouched holdout; neither past-account screenshots
nor claimed 1:3 opportunity ratio establish positive expectancy. Validate with
real historical BTC candles, realistic fees/slippage, paired walk-forward folds
and new future paper data before considering any trading application.