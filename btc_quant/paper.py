"""Paper-only forward execution; no authenticated APIs and no order submission."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import os
import time
import requests
from .config import STEPS
from .core import Account, signals
from .data import fetch

LOG = logging.getLogger(__name__)


@contextmanager
def exclusive(path):
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Already running or stale lock: {lock}") from exc
    try:
        with os.fdopen(fd, "w") as file:
            file.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def atomic_save(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def state_for(cfg):
    return {"schema_version": 2, "config": cfg.to_dict(),
            "last_closed_ms": {s: None for s in cfg.symbols},
            "account": Account.create(cfg).serialize()}


def load_state(path, cfg):
    p = Path(path)
    if not p.exists():
        return state_for(cfg)
    state = json.loads(p.read_text(encoding="utf-8"))
    if state.get("schema_version") != 2 or state.get("config") != cfg.to_dict():
        raise ValueError("State/version/config mismatch; use a NEW state path")
    return state


def paper_tick(state, cfg, history, observed_prices, when):
    """Pure idempotent state update. All feeds must succeed before calling."""
    book = Account.restore(state["account"])
    last = {}
    for sym in cfg.symbols:
        if sym not in history or sym not in observed_prices:
            raise ValueError("Incomplete market snapshot: no trading")
        price = float(observed_prices[sym])
        if not (0 < price < float("inf")):
            raise ValueError("Invalid observed price")
        df = signals(history[sym], cfg)
        if len(df) < max(220, cfg.ema_slow + 2):
            raise ValueError("Insufficient warmup history")
        last[sym] = df.iloc[-1]
    # Validate all cursors and prices before any mutation.
    for sym in cfg.symbols:
        prev = state["last_closed_ms"][sym]
        current = int(last[sym].open_time_ms)
        if prev is not None and current < prev:
            raise ValueError("Candle clock regressed")
        book.prices[sym] = float(observed_prices[sym])
    if book.equity() <= book.peak * (1 - cfg.max_drawdown_pct / 100):
        book.halted = True
    closed_now = set()
    for sym in sorted(cfg.symbols):
        pos = book.positions.get(sym)
        if pos:
            price = book.prices[sym]
            if price <= pos.stop or price >= pos.take:
                reason = "observed_stop" if price <= pos.stop else "observed_take"
                book.sell(sym, when, price, cfg, reason)
                closed_now.add(sym)
    candidates = []
    for sym in sorted(cfg.symbols):
        current = int(last[sym].open_time_ms)
        prev = state["last_closed_ms"][sym]
        if prev is not None and current == prev + STEPS[cfg.timeframe] and sym not in closed_now:
            if bool(last[sym].sell_signal) and sym in book.positions:
                book.sell(sym, when, book.prices[sym], cfg, "ema_cross_down")
            elif bool(last[sym].buy_signal) and sym not in book.positions:
                candidates.append(sym)
        elif prev is not None and current > prev + STEPS[cfg.timeframe]:
            LOG.warning("Skipped %s stale signal after offline gap", sym)
        state["last_closed_ms"][sym] = current
    budget = book.cash / len(candidates) if candidates else 0
    for sym in candidates:
        book.buy(sym, when, book.prices[sym], float(last[sym].atr), cfg, cash_cap=budget)
    book.mark(when, cfg)
    state["account"] = book.serialize()
    return state


def tick_market(cfg):
    history = {s: fetch(s, cfg.timeframe, days=30) for s in cfg.symbols}
    prices = {}
    for s in cfg.symbols:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/price",
                         params={"symbol": s.replace("/", "")}, timeout=20)
        r.raise_for_status()
        prices[s] = float(r.json()["price"])
        # Never generate a signal from a candle that is too old.
        end = int(history[s].iloc[-1].close_time_ms)
        if int(time.time() * 1000) - end > STEPS[cfg.timeframe] + 60_000:
            raise RuntimeError(f"Market data stale: {s}")
    return history, prices


def run_paper(cfg, path, *, once=False, poll_seconds=60):
    if poll_seconds < 30:
        raise ValueError("Poll at least every 30 seconds")
    with exclusive(path):
        state = load_state(path, cfg)
        while True:
            try:
                history, prices = tick_market(cfg)
                when = datetime.now(timezone.utc).isoformat()
                state = paper_tick(state, cfg, history, prices, when)
                atomic_save(path, state)
                book = Account.restore(state["account"])
                LOG.info("Equity=%.4f USDT, holdings=%s, halted=%s", book.equity(),
                         {s: round(p.quantity, 8) for s, p in book.positions.items()}, book.halted)
                if once:
                    return state
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                LOG.exception("Paper tick failed; no new state committed")
                if once:
                    raise
            time.sleep(poll_seconds)
