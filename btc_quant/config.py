from dataclasses import dataclass, asdict

SYMBOLS = ("BTC/USDT", "ETH/USDT")
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
    ema_fast: int = 9
    ema_slow: int = 21
    atr_period: int = 14

    def __post_init__(self):
        if not self.symbols or len(set(self.symbols)) != len(self.symbols) or any(s not in SYMBOLS for s in self.symbols):
            raise ValueError("Only unique BTC/USDT and ETH/USDT spot symbols are supported")
        if self.timeframe not in STEPS or self.starting_usdt <= 0:
            raise ValueError("Invalid timeframe or starting cash")
        if not (0 <= self.fee_bps <= 100 and 0 <= self.slippage_bps <= 100):
            raise ValueError("Invalid fee/slippage bps")
        if not (0 < self.risk_pct <= 10 and 0 < self.max_symbol_pct <= self.max_total_pct <= 100):
            raise ValueError("Invalid position limits")
        if not (0 < self.max_open_risk_pct <= 100 and 0 < self.max_drawdown_pct <= 100):
            raise ValueError("Invalid portfolio risk limits")
        if not (self.stop_atr > 0 and self.take_atr > 0 and self.min_notional > 0):
            raise ValueError("Invalid stop/target/notional")
        if not (2 <= self.ema_fast < self.ema_slow and self.atr_period >= 2):
            raise ValueError("Invalid indicators")

    @property
    def fee(self):
        return self.fee_bps / 10_000

    @property
    def slip(self):
        return self.slippage_bps / 10_000

    def to_dict(self):
        return {**asdict(self), "symbols": list(self.symbols)}

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "symbols": tuple(data["symbols"])})
