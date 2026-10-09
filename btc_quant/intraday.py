"""BTC 4h read-only, single-factor triple-EMA diagnostics and forward windows.

Does not affect the paper engine. All orders are simulated via the unchanged
core.Account execution, fees, slippage and risk checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, STEPS
from .core import Account
from .report import metrics
from .strategies import strategy_signals

ARMS = ("A_base", "B_slope", "C_adx", "D_pullback", "E_trailing")
ARM_RULES = {
    "A_base": "Original EMA9/21 bullish crossover with close > EMA200, original 2xATR stop / 3xATR target / EMA9/21 cross-down exit",
    "B_slope": "A plus EMA200[t] > EMA200[t-6] on completed signal bar (no other change)",
    "C_adx": "A plus Wilder ADX(14) > 20 on completed signal bar (no other change)",
    "D_pullback": "Replace A entry only: EMA9>EMA21, close>EMA200, prior low touches EMA21, current close>EMA21 and rising green candle; original exits",
    "E_trailing": "A entry and EMA exit, original 2xATR initial stop; replace fixed 3xATR take-profit with 3xATR highest-high trailing stop, updated after each fully completed bar",
}


def _wilder_adx(x: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ADX computed using only current/past completed candles."""
    high, low, close = x.high.astype(float), x.low.astype(float), x.close.astype(float)
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    # EWM alpha=1/n is Wilder's recurrence after warmup; outputs are invalid
    # until both DM and DX smoothing periods have elapsed.
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def arm_signals(df: pd.DataFrame, cfg: Config, arm: str) -> pd.DataFrame:
    if arm not in ARMS:
        raise ValueError(f"Unknown research arm: {arm}")
    if cfg.symbols != ("BTC/USDT",) or cfg.timeframe != "4h":
        raise ValueError("v0.2.2 intraday research requires BTC/USDT 4h only")
    out = strategy_signals(df, cfg, "triple_ema")
    if arm == "B_slope":
        out["buy_signal"] = out.buy_signal & (out.ema200 > out.ema200.shift(6))
    elif arm == "C_adx":
        out["adx14"] = _wilder_adx(out)
        out["buy_signal"] = out.buy_signal & (out.adx14 > 20)
    elif arm == "D_pullback":
        ema21 = out.close.ewm(span=cfg.ema_slow, adjust=False,
                               min_periods=cfg.ema_slow).mean()
        ema9 = out.close.ewm(span=cfg.ema_fast, adjust=False,
                              min_periods=cfg.ema_fast).mean()
        eligible = ((ema9 > ema21) & (out.close > out.ema200) &
                    (out.low.shift(1) <= ema21.shift(1)) &
                    (out.close > ema21) & (out.close > out.open) &
                    (out.close > out.close.shift(1)) & out.atr.notna())
        out["buy_signal"] = eligible & ~eligible.shift(1, fill_value=False)
    out["buy_signal"] = out.buy_signal.fillna(False).astype(bool)
    return out


def _iso(ms: int) -> str:
    return pd.to_datetime(ms, unit="ms", utc=True).isoformat()


def run_arm(df: pd.DataFrame, cfg: Config, arm: str, *,
            start_ms: int | None = None, end_ms: int | None = None):
    """Return (unchanged Account, detailed closed-trade records, first halt UTC).

    Baseline is byte-for-byte execution equivalent to core.backtest on the same
    explicit window; new diagnostics do not change the core Account model.
    """
    if cfg.symbols != ("BTC/USDT",) or cfg.timeframe != "4h":
        raise ValueError("BTC-only 4h is required")
    x = arm_signals(df, cfg, arm).set_index("open_time_ms", drop=False)
    ts = list(x.index.astype(int))
    step = STEPS[cfg.timeframe]
    if len(ts) < 220 or any(b - a != step for a, b in zip(ts, ts[1:])):
        raise ValueError("Requires >=220 aligned, gap-free closed 4h candles")
    beginning = ts[0] + 200 * step
    if start_ms is not None:
        beginning = max(beginning, int(start_ms))
    if end_ms is not None and end_ms <= beginning:
        raise ValueError("Empty test window")
    book = Account.create(cfg)
    detail = []
    current = None
    halted_at = None

    def update_excursion(raw: float):
        if current is not None:
            change = 100 * (raw / current["entry_fill"] - 1)
            current["mfe_pct_observed"] = max(current["mfe_pct_observed"], change)
            current["mae_pct_observed"] = min(current["mae_pct_observed"], change)

    def close(when, raw, reason):
        nonlocal current
        update_excursion(float(raw))
        book.sell("BTC/USDT", when, float(raw), cfg, reason)
        if current is None:
            raise AssertionError("Exit with missing diagnostic record")
        t = book.trades[-1]
        detail.append({**t, **{k: v for k, v in current.items() if k != "highest_complete_high"},
                       "hold_hours": (pd.Timestamp(when) - pd.Timestamp(t["entry_time"])).total_seconds() / 3600,
                       "net_trade_return_pct": 100 * t["pnl_usdt"] / current["entry_cost"],
                       "exit_reason": reason, "arm": arm})
        current = None

    for i in range(1, len(ts)):
        t = ts[i]
        if t < beginning:
            continue
        if end_ms is not None and t >= int(end_ms):
            break
        bar, prev = x.iloc[i], x.iloc[i-1]
        when = _iso(t)
        book.prices["BTC/USDT"] = float(bar.open)
        if book.equity() <= book.peak * (1 - cfg.max_drawdown_pct / 100):
            book.halted = True
            if halted_at is None:
                halted_at = when
        pos = book.positions.get("BTC/USDT")
        exited = False
        if pos and float(bar.open) <= pos.stop:
            close(when, float(bar.open), "gap_stop")
            exited = True
        elif pos and bool(prev.sell_signal):
            close(when, float(bar.open), "ema_cross_down")
            exited = True
        # Signal must be from prior candle INSIDE this account window, so that
        # a fresh test account cannot execute a training-period signal.
        if (not book.positions and not exited
                and (start_ms is None or ts[i-1] >= beginning)
                and bool(prev.buy_signal)):
            if book.buy("BTC/USDT", when, float(bar.open), float(prev.atr), cfg):
                pos = book.positions["BTC/USDT"]
                current = {"entry_fill": float(pos.entry_price),
                           "entry_cost": float(pos.entry_cost),
                           "initial_stop": float(pos.stop),
                           "mfe_pct_observed": 0.0, "mae_pct_observed": 0.0,
                           "highest_complete_high": float(bar.open)}
                if arm == "E_trailing":
                    pos.take = float("inf")  # research-only, removes fixed take
        trigger = book.protective_exit("BTC/USDT", float(bar.open), float(bar.high), float(bar.low))
        if trigger:
            close(when, trigger[0], trigger[1])
        elif current is not None:
            # Fully held bar: OHLC extremes are available after the bar closes.
            update_excursion(float(bar.high))
            update_excursion(float(bar.low))
            if arm == "E_trailing":
                pos = book.positions["BTC/USDT"]
                current["highest_complete_high"] = max(current["highest_complete_high"], float(bar.high))
                if pd.notna(bar.atr):
                    # New stop is active only on NEXT bar, never retroactively.
                    pos.stop = max(pos.stop, current["highest_complete_high"] - 3 * float(bar.atr))
        book.prices["BTC/USDT"] = float(bar.close)
        before = book.halted
        book.mark(_iso(t + step - 1), cfg)
        if book.halted and not before and halted_at is None:
            halted_at = _iso(t + step - 1)
    return book, detail, halted_at


def _windows(x: pd.DataFrame, train_days: int, test_days: int, step_days: int):
    step = STEPS["4h"]
    starts = x.open_time_ms.astype(int).to_list()
    first = starts[200]
    upper = starts[-1] + step
    train = train_days * 86_400_000
    test = test_days * 86_400_000
    stride = step_days * 86_400_000
    windows = []
    anchor = first
    while anchor + train + test <= upper:
        windows.append((anchor, anchor + train, anchor + train + test))
        anchor += stride
    if not windows:
        raise ValueError("Insufficient history for requested walk-forward windows")
    return windows


def intraday_study(history: pd.DataFrame, cfg: Config, *, arms=ARMS,
                   train_days=180, test_days=60, step_days=60,
                   output_dir="outputs/btc-intraday-v022", plot=True):
    """Fixed-rule rolling TRAIN/TEST windows. No parameter fitting/selection.

    Each fold and stage starts a fresh 1000-USDT account. Overlapping validation
    windows are prohibited, so fold summaries can be compared without double
    counting OOS observations. Historical results are exploratory, NOT an
    untouched holdout and not a continuously investable equity curve.
    """
    if cfg.symbols != ("BTC/USDT",) or cfg.timeframe != "4h":
        raise ValueError("Research is BTC-only 4h")
    if (not arms or len(set(arms)) != len(arms)
            or any(arm not in ARMS for arm in arms)):
        raise ValueError("Specify unique research arms")
    if not (train_days >= 45 and test_days >= 14 and step_days >= test_days):
        raise ValueError("Require train>=45d, test>=14d, step>=test to avoid overlapping validation")
    ts = history.open_time_ms.astype(int)
    step = STEPS["4h"]
    if len(ts) < 220 or not (ts.diff().dropna() == step).all():
        raise ValueError("Expected sorted, gap-free 4h candles with >=220 rows")
    windows = _windows(history, train_days, test_days, step_days)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, trades, curves = [], [], []
    for fold, (train_start, test_start, test_end) in enumerate(windows, start=1):
        for arm in arms:
            for stage, begin, end in (("train", train_start, test_start),
                                      ("test", test_start, test_end)):
                book, details, halt = run_arm(history, cfg, arm, start_ms=begin, end_ms=end)
                m = metrics(book, cfg)
                row = {"fold": fold, "arm": arm, "stage": stage,
                       "window_start_utc": _iso(begin), "window_end_exclusive_utc": _iso(end),
                       "halted_at_utc": halt, **m}
                rows.append(row)
                trades.extend({"fold": fold, "stage": stage, **r} for r in details)
                curves.extend({"fold": fold, "stage": stage, "arm": arm, **c} for c in book.curve)
    table = pd.DataFrame(rows)
    oos = table[table.stage == "test"]
    agg = oos.groupby("arm", sort=False).agg(
        folds=("fold", "count"),
        median_test_return_pct=("net_return_pct", "median"),
        mean_test_return_pct=("net_return_pct", "mean"),
        positive_folds=("net_return_pct", lambda x: int((x > 0).sum())),
        worst_test_return_pct=("net_return_pct", "min"),
        worst_fold_drawdown_pct=("max_drawdown_pct", "min"),
        total_closed_trades=("closed_trades", "sum"),
        folds_halted=("halted", "sum"),
    ).reset_index()
    table.to_csv(out / "fold_metrics.csv", index=False)
    agg.to_csv(out / "oos_summary.csv", index=False)
    pd.DataFrame(trades).to_csv(out / "trade_diagnostics.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "fold_equity.csv", index=False)
    meta = {"version": "0.2.2", "market": "BTC/USDT", "timeframe": "4h",
            "arms": {k: ARM_RULES[k] for k in arms},
            "train_days": train_days, "test_days": test_days, "step_days": step_days,
            "folds": len(windows),
            "window_policy": "Chronological fixed-rule rolling windows; train/test separately reset simulated account to starting_usdt. No tuning. step>=test prevents overlapping test windows.",
            "fee_slippage_risk": cfg.to_dict(),
            "diagnostic_method": "MFE/MAE relative to entry fill; full-bar high/low used only for fully held candles; intrabar exit candle records exit trigger/price only, because OHLC lacks intrabar path. Thus extrema can be understated; not precision intrabar excursions.",
            "halt_policy": "Existing 12% peak drawdown permanently prevents new entries within each independently reset window; existing positions may close. halt timestamp records first observation.",
            "trailing_policy": "Replaces fixed 3xATR take only. Initial 2xATR stop and all other Account controls unchanged; trail uses completed candle high and ATR, activates no earlier than next candle.",
            "warning": "Historical rolling tests already inspected for strategy development are exploratory, not untouched future out-of-sample evidence. Results across reset windows MUST NOT be compounded as a live equity curve; fees/slippage assumed."}
    (out / "methodology.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        for arm, sub in oos.groupby("arm", sort=False):
            ax.plot(sub.fold, sub.net_return_pct, marker="o", label=arm)
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_xlabel("Chronological OOS fold (fresh simulated account)")
        ax.set_ylabel("Fold net return %")
        ax.legend()
        fig.savefig(out / "oos_fold_returns.png", dpi=150)
        plt.close(fig)
    return table, agg, meta
