"""BTC-only simulation assumptions. No default trading strategy."""
from dataclasses import dataclass, asdict

SYMBOLS = ("BTC/USDT",)
STEPS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = SYMBOLS
    timeframe: str = "4h"
    starting_usdt: float = 1000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    risk_pct: float = 1.0
    max_symbol_pct: float = 30.0
    max_total_pct: float = 50.0
    max_open_risk_pct: float = 2.0
    max_drawdown_pct: float = 12.0
    stop_atr: float = 2.0
    take_atr: float = 3.0
    min_notional: float = 10.0
    atr_period: int = 14
    warmup_bars: int = 200

    def __post_init__(self):
        if self.symbols != SYMBOLS:
            raise ValueError("BTC-only spot: symbols must equal ('BTC/USDT',)")
        if self.timeframe not in STEPS or self.starting_usdt <= 0:
            raise ValueError("Invalid timeframe or initial cash")
        if not (0 <= self.fee_bps <= 100 and 0 <= self.slippage_bps <= 100):
            raise ValueError("Invalid fee/slippage basis points")
        if not (0 < self.risk_pct <= 10 and 0 < self.max_symbol_pct <= self.max_total_pct <= 100):
            raise ValueError("Invalid risk or exposure limits")
        if not (0 < self.max_open_risk_pct <= 100 and 0 < self.max_drawdown_pct <= 100):
            raise ValueError("Invalid drawdown/open-risk limits")
        if self.stop_atr <= 0 or self.take_atr <= 0 or self.min_notional <= 0:
            raise ValueError("Invalid stop/target/notional")
        if not (2 <= self.atr_period <= self.warmup_bars and self.warmup_bars >= 20):
            raise ValueError("Invalid ATR/warmup")

    @property
    def fee(self):
        return self.fee_bps / 10000

    @property
    def slip(self):
        return self.slippage_bps / 10000

    def to_dict(self):
        return {**asdict(self), "symbols": list(self.symbols)}

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "symbols": tuple(data["symbols"])})