# BTC 15m BB + NMACD + VO — isolated research

**Paper / historical simulation only.** Not a claim that the video's advertised
70% win rate or 3x balance is reproducible. This branch is forked from
refactor/btc-only-strategy-neutral-framework and deliberately does not change
the existing spot BTC engine, paper-state format, or main.

## Locked starting assumptions

- BTCUSDT **USD-M perpetual futures historical candles**, 15m, no credentials.
- BB middle = SMA(close,20); Pine-style population stdev(close,20).
- Normalized MACD: EMA(close,13) and EMA(close,21), signed relative ratio,
  min-max normalized across trailing 50, WMA9 trigger.
- Volatility Oscillator (veryfid): spike = close-open; compare to +/- population
  stdev(spike,100). Ignore divergence pivots/repainted R markers.
- Long: NMACD > trigger and spike > upper and candle body crosses BB middle
  (open <= middle and close > middle). Short uses the mirrored rules.
- Signal from a **fully closed candle**, order modeled at next candle's OPEN.
- Stop = signal candle low (long) or high (short), no subjective swing search.
- Exit alternatives: fixed 1.5R; fixed 2R; or half at 1.5R and half at 2R
  (no breakeven change and no runner/trailing stop in this initial comparison).
- Starting equity 1000 USDT; 1% equity max modeled loss per trade including
  stop-fill slippage/entry+exit commissions; gross exposure at most 1x equity;
  never more than one open position, no leverage or real borrowing.
- Fee 5 bps/side and adverse slip 2 bps/side, **independent** of funding.
  Funding at recorded settlement timestamp is signed (positive funding: longs
  pay, shorts receive) and approximated using corresponding 15m bar's OPEN.
- When a stop and target are touched in the same 15m bar, execute the STOP first;
  stop-gap orders fill at opening with adverse slippage. No live or paper orders
  are submitted, and there is no margin, mark-price or liquidation engine.

**Reproduction uncertainties:** Video had multiple suggested stop locations,
and didn't specify 50/50 partial take sizes. The strict body crossing,
signal-candle stop and 50/50 exits are declared experimental interpretations,
not verified 1-to-1 original video fills. Confirm TV script variant and compare
a few known timestamped signals before claiming exact replication.

## Ubuntu — install and run

From this branch's repository root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-dev.txt
    python -m pytest -q

Quick historical smoke test — one completed month:

    python -m experiments.run_bb_nmacd_vo \
      --start 2024-01-01 --end 2024-02-01 \
      --initial-cash 1000 --direction both \
      --fee-bps 5 --slippage-bps 2 --risk-pct 1 \
      --output-dir outputs/bb-nmacd-vo-smoke

Longer research period, up through September 2026:

    python -m experiments.run_bb_nmacd_vo \
      --start 2022-01-01 --end 2026-10-01 \
      --initial-cash 1000 --direction both \
      --fee-bps 5 --slippage-bps 2 --risk-pct 1 \
      --output-dir outputs/bb-nmacd-vo-15m

All dates are UTC, end date exclusive. Downloads are from the Binance Vision
public futures/um monthly archives (current incomplete month via daily archives).
Archives are cached under data/btc-um-archive. If there are missing bars or
funding coverage gaps, the script **stops** rather than fabricating prices,
assuming zero funding or silently interpolating.

Offline data path (both files must be supplied):

    python -m experiments.run_bb_nmacd_vo \
      --start 2024-01-01 --end 2024-02-01 \
      --candles-csv data/your_15m.csv \
      --funding-csv data/your_funding.csv

Candle CSV schema: open_time_ms,open,high,low,close (UTC milliseconds,
including >=200 complete warmup candles). Funding CSV: funding_ms,rate
(actual UTC settlement times and signed decimal rate). Don't confuse this
futures feed with spot 15m candles. No API keys or Binance login are involved.

## Outputs

- outputs/<experiment>/summary.json — net return, max drawdown, win rate,
  profit factor, average net-R, trade count, long/short signal count,
  total commissions, realized funding P&L, assumptions for all 3 exits.
- outputs/<experiment>/trades_fixed_1_5.csv, trades_fixed_2.csv,
  trades_split.csv — trade-level side, entry/exit, gross stop distance,
  **net** P&L, and funding attribution.
- outputs/<experiment>/equity_fixed_1_5.csv, equity_fixed_2.csv,
  equity_split.csv — 15m marked equity and drawdown.

## Testing / decision process

1. Unit tests: no lookahead for indicator calculation, next-bar entry,
   stop-first intrabar ambiguity, 1x notional cap, short funding direction,
   partial exits, zero-trade handling. These run fully offline.
2. Run the Jan 2024 smoke test, inspect source coverage and individual signals
   against the named TV indicators.
3. Run 2022–2026 research and compare the **three exits under identical costs**.
   Then make separate long-only and short-only reports with --direction.
4. Use 2022–2024 for exploration and 2025–2026 for stability diagnostics.
   Earlier inspected periods are *not* truly untouched out-of-sample.
5. Only after satisfactory historical robustness, collect genuinely new
   forward-only observations in a separate non-order simulation system.
   This runner does NOT plug into the existing spot-only paper trader.

A positive historical return alone is insufficient: examine cost sensitivity,
trade count, regime-by-regime returns, concentrated winners, drawdown and
parameter robustness. No historical signal implies a guaranteed fill.
