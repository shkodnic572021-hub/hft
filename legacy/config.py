import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "key.env")


@dataclass
class ExchangeConfig:
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    testnet: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    leverage: int = 25
    margin_type: str = "ISOLATED"


@dataclass
class ScannerConfig:
    quote_asset: str = "USDT"
    min_volume_24h_usdt: float = 150_000_000
    min_price_change_24h: float = 20.0
    natr_min_pct: float = 1.0
    natr_max_pct: float = 3.5
    exclude_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT",
        "BUSDUSDT", "DAIUSDT", "EURUSDT",
    ])
    scan_interval_sec: int = 300
    fading_check_interval_sec: int = 60
    volume_drop_pct: float = 80.0
    natr_drop_factor: float = 4.0
    better_coin_natr_mult: float = 1.2


@dataclass
class StrategyConfig:
    gamma: float = 0.1
    kappa: float = 1.5
    time_horizon: float = 1.0
    calibration_window: int = 1000
    num_levels: int = 1
    level_spacing_pct: float = 0.20
    level_spacing_ticks: int = 5
    order_size_usdt: float = 6.0
    min_order_lifetime_sec: float = 0.5
    tp_offset_pct: float = 0.2
    max_inventory_usdt: float = 12.0
    # Доля баланса (0.10 = 10%): при таком убытке сессии бот останавливается
    max_drawdown_pct: float = 0.10
    max_latency_ms: float = 500.0

    spread_mode: str = "dynamic"
    spread_base_half_pct: float = 0.10
    spread_k_vol: float = 0.15
    spread_k_market: float = 0.5
    spread_max_half_pct: float = 0.80
    spread_natr_refresh_sec: int = 30

    min_half_spread_pct: float = 0.35


@dataclass
class StormConfig:
    enabled: bool = True

    # --- Дисбаланс стакана ---
    imbalance_threshold: float = 0.08
    imbalance_levels: int = 5
    imbalance_consecutive: int = 15

    # --- Расширение спреда ---
    spread_mult: float = 7.0
    spread_baseline_sec: int = 60
    spread_consecutive: int = 3

    # --- BTC-дёргание ---
    btc_symbol: str = "BTCUSDT"
    btc_drop_pct: float = 0.8
    btc_window_sec: float = 5.0
    btc_consecutive: int = 2

    # --- Длительность storm ---
    duration_sec: float = 15.0
    close_position: bool = True

    # --- Per-Position Stop ---
    position_stop_pct: float = 0.6

    # --- Loss Stop ---
    loss_stop_pct: float = 0.01


@dataclass
class PnLConfig:
    log_interval_sec: int = 60
    csv_path: str = "pnl_log.csv"
    summary_on_exit: bool = True


@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    storm: StormConfig = field(default_factory=StormConfig)
    pnl: PnLConfig = field(default_factory=PnLConfig)
    dry_run: bool = False
    manual_symbol: str = ""
