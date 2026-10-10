# BTC 4h Morning Star — local forward PAPER setup (explicit opt-in, NO real orders)

This independent research module does **not** replace the strategy-neutral
framework, does not register a default strategy, and does **not** use private
Binance APIs or trade real money. It will not run just by invoking the old
\`python -m btc_quant paper\` observer.

## Ubuntu

\`\`\`bash
git fetch origin
git switch research/btc-candlestick-baselines-20261010
git pull --ff-only origin research/btc-candlestick-baselines-20261010
source .venv/bin/activate
python -m pytest -q
\`\`\`

First **single read-only snapshot**:

\`\`\`bash
python -m experiments.morningstar_forward_paper --paper --once \\
  --initial-usdt 1000 \\
  --state outputs/morningstar-forward-v1/state.json

python -m experiments.morningstar_forward_paper --status \\
  --state outputs/morningstar-forward-v1/state.json
\`\`\`

Then continuous (while terminal is running, logs to terminal):

\`\`\`bash
python -m experiments.morningstar_forward_paper --paper \\
  --initial-usdt 1000 --poll-seconds 60 \\
  --state outputs/morningstar-forward-v1/state.json
\`\`\`

Press Ctrl+C to stop. The state is JSON, saved atomically and locked against
two concurrent simulators. Never run TWO copies on the same account state.
Protect and back up the state if keeping prospective audit evidence.

## What actually happens

- First run bootstraps and records current last closed BTCUSDT 4h candle,
  but **does not backfill or retroactively buy** any previously completed
  historical signal. A fresh eligible new 4h signal may only enter if
  observed at most 2 minutes after the new bar begins. Any late signal
  is skipped and logged rather than impersonating the next-open fill.
- Only original **3-bar** BTC Morning Star pattern of
  \`experiments.candlestick_study.signals()['morning_star']\`, no trend,
  relative volume or heatmap filters.
- A fully closed 4h signal candle and Wilder 14-bar ATR determine the
  initial **static** stop: simulated buy fill - 1.5 × completed ATR.
  +5% fixed take relative to that same simulated buy fill.
- Fixed **30% of current simulated account equity**, budget includes
  entry fee. No additional nominal 0.9% stop-loss budget as previously
  used in dynamic sizing research. Historical 48-hour reservation starts
  at the next 4h bar boundary, preventing signals overlapping the original
  48h window even after an early sell.
- Exit at first *observed* stop, take or 48-hour deadline; BTC spot-only,
  virtual initial 1,000 USDT, fee 10bps and modeled slippage 5bps per
  trade side.
- Missed more than 10 minutes of ticker observations WHILE a position is
  open: sets halted=true and unobserved_price_gap=true, prevents
  future new buys, and records alert. It can still virtually sell the
  existing position when price is sampled. You should inspect the state,
  not blindly resume after this reliability failure.
- Account cumulative drawdown >=12% from highest observed paper equity:
  halt new buys; continue virtual exits. This guard is separate from
  frozen strategy analysis, recorded in state.

## Differences from backtest, and risks

**The paper simulator samples the public Binance spot ticker. It does not
receive actual exchange fills, does not have a live stop order, and can
MISS price levels touched between 60-second polls, while offline, or
during network/API outages.** Its realized paper PnL should NOT be
compared unqualified to the historical 4h OHLC model, which can check
full intrabar high/low and pessimistically assumes stop-first if both
limits hit during one OHLC bar. Time exits during outage can be late.

The paper entry price is the sampled public ticker at the boundary (not
guaranteed exact next 4h opening price). Entry within 2min is logged with
latency. ATR is calculated from the most recent 30+ days (~380+ closed
4h candles), so Wilder's initial condition can differ minutely from the
historical study's 2020-to-present seed.

This version is an **initial local engineering smoke test**, not a
validated 1m/tick-perfect forward fill engine. Prior to investing real
money, add historical 1m/aggTrades replay and stronger observation
reliability validation. It is not a real-time stop-management system.

The state format is distinct from \`btc_quant paper\` schema v3; do not
mix file paths. If changing code/parameters, use a fresh versioned
state rather than quietly mutating an old prospective track record.

## Non-activation and safety

No public Binance live order route, API key, authenticated trading API
or withdrawal operation is present. Nothing is merged to \`main\`.
This remains a research-only opt-in module.
