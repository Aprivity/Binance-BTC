"""Explicit, locally trusted strategy plugin protocol. No built-in signals."""
from __future__ import annotations
import importlib
import importlib.util
import hashlib
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def resolve_strategy(reference: str):
    """Resolve `module.path:signals` or `/path/to/file.py:signals` explicitly.

    Strategies are executable Python code. Only use trusted, reviewed plugins.
    No plugin is loaded implicitly and no plugin is installed by this project.
    """
    if not reference or ":" not in reference:
        raise ValueError("Explicit --strategy module:callable or path.py:callable required")
    name, symbol = reference.rsplit(":", 1)
    if not name or not symbol.isidentifier():
        raise ValueError("Invalid strategy reference")
    if name.endswith(".py"):
        path = Path(name).resolve(strict=True)
        # Register under a path-stable module name before executing. Dataclasses,
        # decorators and runtime type hints expect the module in sys.modules.
        name_key = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        module_name = f"_btc_strategy_plugin_{name_key}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load strategy file")
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            if previous is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = previous
            raise
    else:
        module = importlib.import_module(name)
    fn = getattr(module, symbol, None)
    if not callable(fn):
        raise ValueError("Strategy must export a callable")
    return fn


def no_trades(frame, cfg):
    """Inert observer signals. Not registered as a trading strategy."""
    return pd.DataFrame({"buy_signal": False, "sell_signal": False}, index=frame.index)


def prepare_signals(frame: pd.DataFrame, cfg, signal_fn):
    """Validate plugin output and provide ATR for universal bracket risk rules."""
    if signal_fn is None or not callable(signal_fn):
        raise ValueError("No trading strategy registered")
    supplied = signal_fn(frame.copy(deep=True), cfg)
    if not isinstance(supplied, pd.DataFrame) or len(supplied) != len(frame):
        raise ValueError("Plugin must return one signal row per input candle")
    if not supplied.index.equals(frame.index):
        raise ValueError("Plugin changed candle alignment")
    for name in ("buy_signal", "sell_signal"):
        if name not in supplied or supplied[name].isna().any():
            raise ValueError(f"Missing/nonboolean {name}")
        if not pd.api.types.is_bool_dtype(supplied[name].dtype):
            raise ValueError(f"{name} must be boolean dtype")
    out = frame.copy()
    out["buy_signal"] = supplied.buy_signal.astype(bool)
    out["sell_signal"] = supplied.sell_signal.astype(bool)
    if (out.buy_signal & out.sell_signal).any():
        raise ValueError("Contradictory simultaneous signals")
    previous = out.close.shift(1)
    ranges = pd.concat([out.high-out.low, (out.high-previous).abs(),
                        (out.low-previous).abs()], axis=1)
    tr = ranges.max(axis=1)
    out["atr"] = tr.ewm(alpha=1/cfg.atr_period, adjust=False,
                        min_periods=cfg.atr_period).mean()
    # Optional research-only planned price bracket. Both fields are required
    # together. A signal's limits are decided at the candle close and carried
    # to the next open; the engine computes 3R (or other declared R) at entry.
    fields = ("initial_stop_price", "reward_risk")
    if any(field in supplied for field in fields):
        if not all(field in supplied for field in fields):
            raise ValueError("Explicit price bracket requires both initial_stop_price and reward_risk")
        for field in fields:
            try:
                values = pd.to_numeric(supplied[field], errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {field} numbers") from exc
            active = out.buy_signal.to_numpy(dtype=bool)
            if np.any(active & (~np.isfinite(values) | (values <= 0))):
                raise ValueError(f"Positive finite {field} required for every entry signal")
            out[field] = values
    return out