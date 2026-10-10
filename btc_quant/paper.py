"""Forward paper-only observer and opt-in local-plugin simulator. NEVER submits orders."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import math
import os
import time
import requests
from .config import STEPS
from .core import Account
from .strategy import prepare_signals
from .data import fetch

LOG = logging.getLogger(__name__)
SCHEMA = 3


@contextmanager
def exclusive(path):
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Paper state already running or stale lock: {lock}") from exc
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def atomic_save(path, payload):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, dest)


def state_for(cfg, strategy_ref=None):
    return {"schema_version": SCHEMA, "config": cfg.to_dict(), "strategy_ref":strategy_ref,
            "last_closed_ms": None, "account": Account.create(cfg).serialize()}


def load_state(path, cfg, strategy_ref=None):
    p=Path(path)
    if not p.exists():
        return state_for(cfg,strategy_ref)
    state=json.loads(p.read_text(encoding="utf-8"))
    if (state.get("schema_version") != SCHEMA or state.get("config") != cfg.to_dict()
            or state.get("strategy_ref") != strategy_ref):
        raise ValueError("Incompatible paper state/schema/strategy: start a NEW state file; never migrate v2 implicitly")
    return state


def paper_tick(state, cfg, history, observed_price, when, *, signal_fn=None, strategy_ref=None):
    """Single public-data snapshot, idempotent. Strategy omitted = FLAT observer."""
    if state.get("schema_version")!=SCHEMA or state.get("config")!=cfg.to_dict() or state.get("strategy_ref")!=strategy_ref:
        raise ValueError("Paper state must match config and explicit plugin reference")
    if (signal_fn is None) != (strategy_ref is None):
        raise ValueError("Plugin callable and reference must be provided together")
    if "BTC/USDT" not in history or not math.isfinite(float(observed_price)) or float(observed_price)<=0:
        raise ValueError("Incomplete or invalid public market snapshot")
    frame=history["BTC/USDT"]
    if len(frame)<(cfg.warmup_bars+1 if signal_fn else 2):
        raise ValueError("Insufficient closed public candles")
    last = prepare_signals(frame,cfg,signal_fn).iloc[-1] if signal_fn else frame.iloc[-1]
    cursor=int(last.open_time_ms)
    old=state["last_closed_ms"]
    if old is not None and cursor<old:
        raise ValueError("Candle clock regressed")
    # Reconstruct into fresh objects: a failed tick never mutates caller state.
    book=Account.restore(state["account"])
    symbol="BTC/USDT"
    if signal_fn is None and book.positions:
        raise ValueError("Observer mode cannot inherit a trading position")
    book.prices[symbol]=float(observed_price)
    book.check_halt(when,cfg)
    closed=False
    if symbol in book.positions:
        pos=book.positions[symbol]
        if observed_price<=pos.stop or observed_price>=pos.take:
            reason="observed_stop" if observed_price<=pos.stop else "observed_take"
            book.sell(symbol,when,float(observed_price),cfg,reason)
            closed=True
    if signal_fn and old is not None and cursor==old+STEPS[cfg.timeframe]:
        if symbol in book.positions and bool(last.sell_signal):
            book.sell(symbol,when,float(observed_price),cfg,"plugin_exit")
            closed=True
        elif not closed and symbol not in book.positions and bool(last.buy_signal):
            bracket = ({"stop_price": float(last.initial_stop_price),
                        "reward_risk": float(last.reward_risk)}
                       if "initial_stop_price" in last.index else {})
            book.buy(symbol,when,float(observed_price),float(last.atr),cfg,**bracket)
    elif signal_fn and old is not None and cursor>old+STEPS[cfg.timeframe]:
        LOG.warning("Missed bars; stale plugin signals skipped")
    book.mark(when,cfg)
    return {**state, "last_closed_ms":cursor, "account":book.serialize()}


def tick_market(cfg):
    step=STEPS[cfg.timeframe]
    days=max(30, (cfg.warmup_bars+4)*step//86_400_000 + 3)
    data=fetch("BTC/USDT",cfg.timeframe,days=days)
    response=requests.get("https://data-api.binance.vision/api/v3/ticker/price",
                          params={"symbol":"BTCUSDT"},timeout=20)
    response.raise_for_status()
    price=float(response.json()["price"])
    if not math.isfinite(price) or price<=0:
        raise ValueError("Invalid spot ticker")
    if int(time.time()*1000)-int(data.iloc[-1].close_time_ms)>step+60_000:
        raise RuntimeError("Public candles stale; observer update rejected")
    return {"BTC/USDT":data},price


def run_paper(cfg, path, *, signal_fn=None, strategy_ref=None, once=False, poll_seconds=60):
    if poll_seconds<30:
        raise ValueError("Poll at least every 30 seconds")
    with exclusive(path):
        state=load_state(path,cfg,strategy_ref)
        while True:
            try:
                history,price=tick_market(cfg)
                when=datetime.now(timezone.utc).isoformat()
                new=paper_tick(state,cfg,history,price,when,signal_fn=signal_fn,strategy_ref=strategy_ref)
                atomic_save(path,new)
                state=new
                book=Account.restore(state["account"])
                LOG.info("Paper mode %s, USDT equity %.4f, holdings %s",
                         strategy_ref or "observer-only",book.equity(),list(book.positions))
                if once:
                    return state
            except (KeyboardInterrupt,SystemExit):
                raise
            except Exception:
                LOG.exception("Paper tick failed; state not committed")
                if once:
                    raise
            time.sleep(poll_seconds)