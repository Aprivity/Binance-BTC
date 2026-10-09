"""Read-only, walk-forward-style chronological holdout comparisons (no tuning)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, STEPS
from .core import backtest
from .report import metrics
from .strategies import STRATEGIES, STRATEGY_DESCRIPTIONS


def _window_prices(history, book, symbols, interval):
    if not book.curve:
        raise ValueError("No backtest observations")
    step = STEPS[interval]
    first = int(pd.Timestamp(book.curve[0]["time"]).value // 1_000_000) - step + 1
    last = int(pd.Timestamp(book.curve[-1]["time"]).value // 1_000_000) - step + 1
    out = {}
    for sym in symbols:
        x = history[sym]
        x = x[(x.open_time_ms >= first) & (x.open_time_ms <= last)]
        if x.empty:
            raise ValueError(f"No benchmark candles for {sym}")
        out[sym] = x
    return out


def benchmark_metrics(history, book, cfg):
    """Same risk CAP benchmark plus *ex-post* average exposure matched comparison.

    Ex-post exposure uses the completed strategy path, so it is a diagnostic,
    NOT a forward-investable strategy or evidence of predictive ability.
    Both hold benchmarks incur assumed entry fee/slippage, no exit cost.
    """
    bars = _window_prices(history, book, cfg.symbols, cfg.timeframe)
    n = len(cfg.symbols)
    policy_weight = min(cfg.max_symbol_pct / 100, cfg.max_total_pct / 100 / n)
    weights = pd.DataFrame([
        {sym: snapshot.get("weights", {}).get(sym, 0.0) for sym in cfg.symbols}
        for snapshot in book.curve], columns=list(cfg.symbols))
    avg_weights = weights.mean().clip(lower=0).to_dict()

    def hold(alloc):
        cash_share = 1.0 - sum(alloc.values())
        if cash_share < -1e-8:
            raise ValueError("Invalid benchmark allocations")
        value = cfg.starting_usdt * max(cash_share, 0)
        for sym, weight in alloc.items():
            x = bars[sym]
            first_price = float(x.iloc[0].open) * (1 + cfg.slip)
            qty = cfg.starting_usdt * weight / (first_price * (1 + cfg.fee))
            value += qty * float(x.iloc[-1].close)
        return 100 * (value / cfg.starting_usdt - 1)

    policy = {s: policy_weight for s in cfg.symbols}
    hold_policy = hold(policy)
    hold_matched = hold(avg_weights)
    return {
        "policy_hold_return_pct": round(hold_policy, 4),
        "ex_post_exposure_hold_return_pct": round(hold_matched, 4),
        "avg_exposure_pct": round(100 * sum(avg_weights.values()), 4),
        "policy_exposure_cap_pct": round(100 * sum(policy.values()), 4),
        "strategy_minus_policy_pct_points": round(100 * (book.equity() / cfg.starting_usdt - 1) - hold_policy, 4),
        "strategy_minus_ex_post_pct_points": round(100 * (book.equity() / cfg.starting_usdt - 1) - hold_matched, 4),
        "ex_post_comparison_warning": "ex-post analysis uses realized future strategy exposure; descriptive only, not tradable",
    }


def _scope_configs(cfg, scope):
    if scope not in ("all", "portfolio", "independent"):
        raise ValueError("Invalid compare scope")
    results = []
    if scope in ("all", "independent"):
        for sym in cfg.symbols:
            results.append((sym.replace("/", ""),
                            Config.from_dict({**cfg.to_dict(), "symbols": [sym]})))
    if scope in ("all", "portfolio"):
        results.append(("portfolio", cfg))
    return results


def compare(history, cfg, *, strategies=STRATEGIES, train_ratio=0.7,
            scope="all", days=None, output_dir="outputs/compare-v021", plot=True):
    """Immutable shared OHLCV / cost assumptions; separate accounts at train/test split."""
    if not 0.5 <= train_ratio <= 0.85:
        raise ValueError("train_ratio must be between 0.5 and 0.85")
    if not strategies or len(set(strategies)) != len(strategies) or any(s not in STRATEGIES for s in strategies):
        raise ValueError("Choose unique supported strategies")
    if set(history) != set(cfg.symbols):
        raise ValueError("History must contain all requested symbols")
    # Explicitly aligned, complete timestamps. backtest() validates each window too.
    common = sorted(set.intersection(*(set(x.open_time_ms.astype(int)) for x in history.values())))
    if len(common) < 500:
        raise ValueError("Need >= 500 aligned candles for 200-bar warmup + train/test holdout")
    valid = [t for t in common[200:] if days is None or t >= common[-1] - days * 86_400_000]
    if len(valid) < 300:
        raise ValueError("Need >= 300 in-sample/out-of-sample candles after warmup")
    split_i = int(len(valid) * train_ratio)
    split_ms = valid[split_i]
    if split_i < 150 or len(valid) - split_i < 75:
        raise ValueError("Train and test windows are too short")
    start_ms = valid[0]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, paths = [], {}
    for label, subcfg in _scope_configs(cfg, scope):
        subset = {s: history[s] for s in subcfg.symbols}
        for strategy in strategies:
            for stage in ("train", "test", "full"):
                kwargs = ({"start_ms": start_ms, "end_ms": split_ms} if stage == "train" else
                          {"start_ms": split_ms} if stage == "test" else {})
                book = backtest(subset, subcfg, days=days, strategy=strategy, **kwargs)
                result = metrics(book, subcfg)
                baseline = benchmark_metrics(subset, book, subcfg)
                rows.append({"scope": label, "strategy": strategy, "stage": stage,
                             "first_mark_utc": book.curve[0]["time"],
                             "last_mark_utc": book.curve[-1]["time"],
                             "bars": len(book.curve), **result, **{k: v for k, v in baseline.items()
                                                                 if k != "ex_post_comparison_warning"}})
                if stage == "test" and label == "portfolio":
                    paths[strategy] = pd.DataFrame(book.curve)[["time", "equity"]].set_index("time").equity
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "compare.csv", index=False)
    metadata = {"version": "0.2.1", "strategies": list(strategies),
                "strategy_descriptions": {s: STRATEGY_DESCRIPTIONS[s] for s in strategies},
                "symbols": list(cfg.symbols), "timeframe": cfg.timeframe,
                "train_ratio": train_ratio,
                "split_utc": pd.to_datetime(split_ms, unit="ms", utc=True).isoformat(),
                "holdout_design": "Chronological, mutually exclusive. Reset cash for train and test; indicators may read prior candles for warmup, but OOS signals cannot trade at first test candle.",
                "benchmark_design": "Fixed buy/hold allocation respecting identical per-symbol and portfolio exposure caps; entry fee/slippage, no exit fee. Secondary comparison matches average realized strategy exposure ex-post ONLY.",
                "selection_warning": "Research comparisons are exploratory; inspecting test results for strategy selection invalidates their untouched-holdout status."}
    (output / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        chart = summary[(summary.stage == "test") & (summary.scope == ("portfolio" if "portfolio" in summary.scope.values else summary.scope.iloc[0]))]
        if not chart.empty:
            fig, (a, b) = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
            x = np.arange(len(chart))
            a.bar(x, chart.net_return_pct)
            a.plot(x, chart.policy_hold_return_pct, marker="o", linestyle="--", label="Same-cap buy-and-hold")
            a.axhline(0, linewidth=.7)
            a.set_xticks(x, chart.strategy, rotation=15)
            a.set_ylabel("OOS net return %")
            a.legend()
            b.bar(x, chart.max_drawdown_pct)
            b.set_xticks(x, chart.strategy, rotation=15)
            b.set_ylabel("OOS max drawdown %")
            fig.savefig(output / "oos_comparison.png", dpi=150)
            plt.close(fig)
        if paths:
            fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
            for name, values in paths.items():
                ax.plot(pd.to_datetime(values.index, utc=True), values.values,
                        label=name)
            ax.set_title("Portfolio OOS equity, separate fresh 1000 USDT accounts")
            ax.set_ylabel("USDT")
            ax.legend()
            fig.savefig(output / "oos_equity.png", dpi=150)
            plt.close(fig)
    return summary, metadata
