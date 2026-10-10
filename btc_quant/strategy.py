"""Explicit, locally trusted strategy plugin protocol. No built-in signals."""
from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path
import pandas as pd


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
        spec = importlib.util.spec_from_file_location("_btc_explicit_strategy_plugin", path)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load strategy file")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
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
    return out