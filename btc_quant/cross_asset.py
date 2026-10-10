"""Fixed-specification daily cross-asset -> BTC 4h/24h predictive research.

No trading engine changes. Forecasts are historical research, never live signals.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_asset_data import available_times, validate_daily
from .data import load_csv, validate

ASSETS = ("ndx", "dxy", "gold", "vix")
BTC_FEATURES = ["btc_r4h", "btc_r24h", "btc_r7d", "btc_vol7d"]
EXTERNAL_SUFFIXES = ("r1", "r5", "vol20", "age_hours")
RIDGE_ALPHA = 10.0


def build_samples(btc, external, horizon_hours):
    """One prediction per NDX session, first 4h boundary AFTER eligible time.

BTC features end strictly before entry; labels use next open prices. External
observations are backward-asof only, maximum 96-hour age. No weekend repetition.
"""
    if horizon_hours not in (4, 24) or set(external) != set(ASSETS):
        raise ValueError("Require 4/24 hours and all four external asset inputs")
    b = validate(btc, "4h", require_closed=True)
    clock = pd.to_datetime(b.open_time_ms, unit="ms", utc=True)
    r = np.log(b.close).diff()
    frame = pd.DataFrame({"entry_utc": clock,
        "btc_r4h": r.shift(1), "btc_r24h": np.log(b.close / b.close.shift(6)).shift(1),
        "btc_r7d": np.log(b.close / b.close.shift(42)).shift(1),
        "btc_vol7d": r.rolling(42).std().shift(1)})
    shift = horizon_hours // 4
    frame["exit_utc"] = clock.shift(-shift)
    frame["actual_return"] = b.open.shift(-shift) / b.open - 1
    anchor = None
    for asset in ASSETS:
        d = validate_daily(external[asset])
        d["available_utc"] = available_times(d.date)
        d[f"{asset}_r1"] = np.log(d.close).diff()
        d[f"{asset}_r5"] = np.log(d.close / d.close.shift(5))
        d[f"{asset}_vol20"] = d[f"{asset}_r1"].rolling(20).std()
        cols = ["available_utc", "date", *(f"{asset}_{s}" for s in EXTERNAL_SUFFIXES[:3])]
        right = d[cols].rename(columns={"date": f"{asset}_session_date", "available_utc": f"{asset}_available_utc"})
        frame = pd.merge_asof(frame.sort_values("entry_utc"), right.sort_values(f"{asset}_available_utc"),
            left_on="entry_utc", right_on=f"{asset}_available_utc", direction="backward",
            tolerance=pd.Timedelta(hours=96), allow_exact_matches=False)
        frame[f"{asset}_age_hours"] = (frame.entry_utc - frame[f"{asset}_available_utc"]).dt.total_seconds() / 3600
        if asset == "ndx":
            anchor = pd.DatetimeIndex(d.available_utc).ceil("4h")
    frame = frame[frame.entry_utc.isin(anchor)].dropna().reset_index(drop=True)
    if frame.empty or frame.entry_utc.duplicated().any():
        raise ValueError("No valid unique aligned samples")
    if (frame.exit_utc - frame.entry_utc != pd.Timedelta(hours=horizon_hours)).any():
        raise ValueError("Invalid forward target endpoints")
    for asset in ASSETS:
        if not (frame[f"{asset}_available_utc"] < frame.entry_utc).all():
            raise ValueError("Future external data detected")
    return frame


def ridge_predict(train, test, features, alpha=RIDGE_ALPHA):
    """Train-only scaling; fixed L2 penalty, intercept unpenalized."""
    x = train[features].to_numpy(float)
    xt = test[features].to_numpy(float)
    y = train.actual_return.to_numpy(float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    z, zt = (x - mean) / scale, (xt - mean) / scale
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return y.mean() + zt @ beta


def model_features():
    specs = {"btc_only": BTC_FEATURES}
    for asset in ASSETS:
        specs[f"btc_plus_{asset}"] = BTC_FEATURES + [f"{asset}_{s}" for s in EXTERNAL_SUFFIXES]
    specs["btc_plus_all"] = BTC_FEATURES + [f"{a}_{s}" for a in ASSETS for s in EXTERNAL_SUFFIXES]
    return specs


def walk_forward(samples, train_days=180, test_days=60, min_train=90, min_test=20):
    """Rolling fixed calendar windows; purge unavailable labels, no test tuning."""
    if train_days < 30 or test_days < 10 or min_train < 2 or min_test < 2:
        raise ValueError("Invalid walk-forward windows")
    samples = samples.sort_values("entry_utc").reset_index(drop=True)
    start = samples.entry_utc.min() + pd.Timedelta(days=train_days)
    records, folds = [], []
    fold = 0
    while start <= samples.entry_utc.max():
        end = start + pd.Timedelta(days=test_days)
        train = samples[(samples.entry_utc >= start - pd.Timedelta(days=train_days)) &
                        (samples.entry_utc < start) & (samples.exit_utc < start)]
        test = samples[(samples.entry_utc >= start) & (samples.entry_utc < end)]
        if len(train) >= min_train and len(test) >= min_test:
            fold += 1
            predictions = {"zero_return": np.zeros(len(test)),
                "training_mean": np.full(len(test), train.actual_return.mean())}
            predictions.update({name: ridge_predict(train, test, cols) for name, cols in model_features().items()})
            for name, pred in predictions.items():
                out = test[["entry_utc", "exit_utc", "actual_return"]].copy()
                out["prediction"] = pred
                out["model"], out["fold"] = name, fold
                records.append(out)
            folds.append({"fold": fold, "test_start_utc": start.isoformat(), "test_end_exclusive_utc": end.isoformat(),
                "train_samples": len(train), "test_samples": len(test),
                "last_train_label_utc": train.exit_utc.max().isoformat(),
                "first_test_entry_utc": test.entry_utc.min().isoformat(),
                "partial_last_window": bool(end > samples.entry_utc.max() + pd.Timedelta(days=1))})
        start = end
    if not records:
        raise ValueError("Insufficient samples for walk-forward evaluation")
    return pd.concat(records, ignore_index=True), pd.DataFrame(folds)


def bootstrap_mean(values, repeats=2000, block=10, seed=20261010):
    """Circular moving-block bootstrap; approximate serial-dependence CI/p-value."""
    v = np.asarray(values, dtype=float)
    if len(v) < 20 or not np.isfinite(v).all() or repeats < 100 or block < 1:
        raise ValueError("Insufficient/invalid bootstrap data")
    block = min(block, len(v))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(v), size=(repeats, int(np.ceil(len(v) / block))))
    idx = (starts[..., None] + np.arange(block)) % len(v)
    means = v[idx.reshape(repeats, -1)[:, :len(v)]].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    # Recenter at the null. Add-one correction avoids a spurious zero p-value.
    p = (1 + (np.abs(means - v.mean()) >= abs(v.mean())).sum()) / (repeats + 1)
    return {"mean": float(v.mean()), "ci_low": float(low), "ci_high": float(high), "p_value": float(p)}


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    previous = 0.0
    for rank, idx in enumerate(order):
        previous = max(previous, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, previous)
    return adjusted


def diagnostic_trades(frame, fee_bps=10, slippage_bps=5, weight=.30):
    """Separate horizon-specific, non-overlapping long/cash toy diagnostic.

Fixed 30% exposure, no ATR stops/risk sizing/halt: NOT the existing EMA engine.
Includes BOTH entry/exit costs; never pools horizons or simultaneous models.
"""
    if fee_bps < 0 or slippage_bps < 0 or fee_bps >= 10000 or slippage_bps >= 10000 or not 0 < weight <= 1:
        raise ValueError("Invalid costs/exposure")
    frame = frame.sort_values("entry_utc")
    if (frame.entry_utc.iloc[1:].to_numpy() < frame.exit_utc.iloc[:-1].to_numpy()).any():
        raise ValueError("Overlapping trades cannot be compounded")
    fee, slip = fee_bps / 1e4, slippage_bps / 1e4
    ratio = (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee))
    take = frame.prediction.to_numpy() > (1 / ratio - 1)
    gross = frame.actual_return.to_numpy()
    net = np.where(take, weight * ((1 + gross) * ratio - 1), 0)
    upper = np.where(take, weight * gross, 0)
    equity = np.r_[1., np.cumprod(1 + net)]
    dd = 1 - equity / np.maximum.accumulate(equity)
    return {"toy_trades": int(take.sum()), "toy_net_return_pct": float((equity[-1] - 1) * 100),
        "toy_same_trades_zero_cost_return_pct": float((np.prod(1 + upper) - 1) * 100),
        "toy_max_drawdown_pct": float(dd.max() * 100), "toy_exposure_pct": weight * 100}


def evaluate(predictions, horizon, fee_bps=10, slippage_bps=5, bootstrap_repeats=2000):
    wide = predictions.pivot(index=["entry_utc", "exit_utc", "fold", "actual_return"], columns="model", values="prediction").reset_index().sort_values("entry_utc")
    y = wide.actual_return.to_numpy()
    btc_error = np.abs(y - wide.btc_only.to_numpy())
    mean_sse = np.sum((y - wide.training_mean.to_numpy()) ** 2)
    rows = []
    for name in ["zero_return", "training_mean", *model_features()]:
        pred = wide[name].to_numpy()
        row = {"horizon_hours": horizon, "model": name, "samples": len(y),
            "mae_bps": float(np.abs(y - pred).mean() * 1e4),
            "rmse_bps": float(np.sqrt(np.mean((y - pred) ** 2)) * 1e4),
            "oos_r2_vs_training_mean": float(1 - np.sum((y - pred) ** 2) / mean_sse),
            "direction_accuracy_pct": float(((pred > 0) == (y > 0)).mean() * 100) if np.any(pred != 0) else np.nan,
            "always_up_accuracy_pct": float((y > 0).mean() * 100),
            "up_rate_pct": float((pred > 0).mean() * 100)}
        row.update(diagnostic_trades(predictions[predictions.model == name], fee_bps, slippage_bps))
        if name.startswith("btc_plus_"):
            diff = bootstrap_mean(btc_error - np.abs(y - pred), repeats=bootstrap_repeats)
            row.update({"mae_improvement_vs_btc_bps": diff["mean"] * 1e4,
                "mae_improvement_ci_low_bps": diff["ci_low"] * 1e4,
                "mae_improvement_ci_high_bps": diff["ci_high"] * 1e4,
                "mae_p_value": diff["p_value"]})
        rows.append(row)
    per_fold = []
    for (fold, model), g in predictions.groupby(["fold", "model"]):
        e = g.actual_return - g.prediction
        per_fold.append({"horizon_hours": horizon, "fold": fold, "model": model, "samples": len(g),
            "mae_bps": float(e.abs().mean() * 1e4),
            "direction_accuracy_pct": float(((g.prediction > 0) == (g.actual_return > 0)).mean() * 100),
            **diagnostic_trades(g, fee_bps, slippage_bps)})
    return pd.DataFrame(rows), pd.DataFrame(per_fold)


def descriptive_correlations(samples, horizon):
    rows = []
    # Contemporaneous window here is trailing BTC 24h up to entry, NOT the
    # precisely synchronized US session return. This is explicitly descriptive.
    for asset in ASSETS:
        x = samples[f"{asset}_r1"]
        rows.append({"horizon_hours": horizon, "asset": asset, "samples": len(samples),
            "corr_with_trailing_btc_24h_logreturn": x.corr(samples.btc_r24h),
            "corr_with_future_btc_return": x.corr(samples.actual_return),
            "interpretation": "Full-sample exploratory correlation, not independent OOS or exact synchronized session correlation"})
    return pd.DataFrame(rows)


def study(directory, output_dir, train_days=180, test_days=60, fee_bps=10, slippage_bps=5, bootstrap_repeats=2000):
    inputs, out = Path(directory), Path(output_dir)
    manifest = json.loads((inputs / "data_manifest.json").read_text()) if (inputs / "data_manifest.json").exists() else None
    files = [inputs / "BTCUSDT_4h.csv", *(inputs / f"{a}_daily.csv" for a in ASSETS)]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if manifest:
        for a, info in manifest["inputs"].items():
            name = "BTCUSDT_4h.csv" if a == "btc" else f"{a}_daily.csv"
            if info["sha256"] != hashes[name]:
                raise ValueError(f"Input changed since download: {name}")
    out.mkdir(parents=True, exist_ok=True)
    btc = load_csv(inputs / "BTCUSDT_4h.csv", "BTC/USDT", "4h")
    external = {a: pd.read_csv(inputs / f"{a}_daily.csv") for a in ASSETS}
    summaries, per_folds, correlations, forecasts, fold_tables, timing = [], [], [], [], [], {}
    for horizon in (4, 24):
        samples = build_samples(btc, external, horizon)
        predictions, folds = walk_forward(samples, train_days, test_days)
        predictions["horizon_hours"] = horizon
        folds["horizon_hours"] = horizon
        summary, per_fold = evaluate(predictions, horizon, fee_bps, slippage_bps, bootstrap_repeats)
        summaries.append(summary); per_folds.append(per_fold); forecasts.append(predictions); fold_tables.append(folds)
        correlations.append(descriptive_correlations(samples, horizon))
        timing[str(horizon)] = {"aligned_samples": len(samples), "folds": len(folds),
            "first_sample_utc": samples.entry_utc.min().isoformat(), "last_sample_utc": samples.entry_utc.max().isoformat(),
            "oos_samples": summary.samples.iloc[0].item(),
            "first_oos_utc": predictions.entry_utc.min().isoformat(), "last_oos_utc": predictions.entry_utc.max().isoformat()}
        # An audit snapshot of information actually available at each entry.
        samples.to_csv(out / f"aligned_samples_{horizon}h.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    is_added = summary.model.str.startswith("btc_plus_")
    summary.loc[is_added, "mae_holm_p_value"] = holm(summary.loc[is_added, "mae_p_value"])
    summary["passes_primary_test"] = (summary.mae_improvement_ci_low_bps > 0) & (summary.mae_holm_p_value < .05)
    summary.to_csv(out / "forecast_summary.csv", index=False)
    pd.concat(per_folds).to_csv(out / "fold_metrics.csv", index=False)
    pd.concat(correlations).to_csv(out / "descriptive_correlations.csv", index=False)
    pd.concat(forecasts).to_csv(out / "oos_predictions.csv", index=False)
    pd.concat(fold_tables).to_csv(out / "fold_boundaries.csv", index=False)
    method = {"research_version": "cross-asset-0.1", "inputs_sha256": hashes, "download_manifest": manifest,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "availability": "Session date eligible next calendar day 00:01 America/New_York; backward-asof max age 96h; one entry on first eligible 4h boundary per NDX session, normally 08:00 UTC. No weekend repeat observations.",
        "targets": "BTC/USDT next open to open after 4h / 24h. BTC predictors end on preceding completed candle.",
        "assets": {"ndx": "Yahoo ^NDX daily index; NOT NQ futures", "dxy": "Yahoo DX-Y.NYB dollar index",
            "gold": "Yahoo GC=F continuous gold futures proxy; NOT XAU spot; rolls can contaminate returns", "vix": "CBOE daily VIX CLOSE"},
        "features": model_features(), "ridge_alpha": RIDGE_ALPHA, "train_days": train_days, "test_days": test_days,
        "training_policy": "Rolling train-only standardization and fixed ridge alpha; purge any training label whose endpoint is >= test boundary. Last partial test window retained only with >=20 observations; >=90 train observations.",
        "primary_endpoint": "Paired OOS absolute error improvement vs BTC-only ridge. Circular 10-observation block bootstrap 95% CI, null-centered two-sided approximate p; Holm correction over all 5 additions x 2 horizons. Other metrics descriptive.",
        "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": 20261010,
        "toy_execution": {"description": "Separate long/cash fixed 30% exposure simulation for each horizon/model; enter only when predicted simple return exceeds modeled roundtrip breakeven. Nonoverlapping open/open trades. No existing EMA stops, risk sizing or halt; no leverage/shorts. Diagnostic only, not a replacement strategy.", "fee_bps_per_side": fee_bps, "slippage_bps_per_side": slippage_bps},
        "timing": timing,
        "limitations": ["Exploratory historical rolling holdouts, not untouched future data; previous development inspected overlapping history.",
            "Current vendor snapshots are not timestamped point-in-time vintages. Availability lag reduces look-ahead but cannot prove historical publication/revision timing.",
            "Daily observations cannot answer whether contemporaneous intraday NQ/DXY/VIX signals lead BTC within the same trading session.",
            "Daily sampling conditional on NDX sessions omits most weekends and uses mainly one UTC hour; not all 24/7 market conditions.",
            "Gold futures roll changes, vendor anomalies and source discrepancies need separate checks before trading use.",
            "Block-bootstrap inference assumes approximate local dependence; structural changes and fold fitting dependence can make intervals optimistic. No causal claim.",
            "Modeling fees/slippage does not guarantee executable fills; toy return is not EMA strategy return. Do not pool horizons/models."]}
    (out / "methodology.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    return summary, method


def main():
    p = argparse.ArgumentParser(description="Cross-asset BTC historical prediction research; never orders")
    p.add_argument("--directory", default="data/market/cross-asset")
    p.add_argument("--output-dir", default="outputs/cross-asset")
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=60)
    p.add_argument("--fee-bps", type=float, default=10)
    p.add_argument("--slippage-bps", type=float, default=5)
    a = p.parse_args()
    summary, _ = study(a.directory, a.output_dir, a.train_days, a.test_days, a.fee_bps, a.slippage_bps)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
