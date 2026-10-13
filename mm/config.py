"""
Конфигурация маркет-мейкера.

Все величины в bps (базисных пунктах): 1 bps = 0.01%.
Ключи читаются из key.env в корне проекта (BINANCE_API_KEY / BINANCE_API_SECRET).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "key.env")


@dataclass
class ExchangeConfig:
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    testnet: bool = field(
        default_factory=lambda: os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    )

    # Маркет-мейкеру большое плечо не нужно: позиция ограничена
    # max_inventory_usdt, а высокое плечо только приближает ликвидацию.
    leverage: int = 5
    margin_type: str = "ISOLATED"

    # Комиссии VIP0 USDⓈ-M. В live-режиме при старте подтягиваются
    # фактические ставки аккаунта через API (без учёта скидки BNB).
    # Вся стратегия считает прибыль ПОСЛЕ этих комиссий.
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0

    # WebSocket Binance Futures (с 2026 года потоки разделены по путям:
    # /public — стакан и bookTicker, /market — сделки, /private — ордера).
    ws_public: str = "wss://fstream.binance.com/public"
    ws_market: str = "wss://fstream.binance.com/market"
    ws_private: str = "wss://fstream.binance.com/private"
    ws_public_testnet: str = "wss://fstream.binancefuture.com/public"
    ws_market_testnet: str = "wss://fstream.binancefuture.com/market"
    ws_private_testnet: str = "wss://fstream.binancefuture.com/private"

    def ws_urls(self):
        if self.testnet:
            return self.ws_public_testnet, self.ws_market_testnet, self.ws_private_testnet
        return self.ws_public, self.ws_market, self.ws_private


@dataclass
class ScannerConfig:
    quote_asset: str = "USDT"
    exclude_symbols: List[str] = field(default_factory=lambda: [
        "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT", "EURUSDT",
    ])

    # Объём: снизу — чтобы были исполнения, сверху — топ-монеты, где спред
    # 1 тик и очередь занята колоцированными HFT (с 65 мс там не выиграть).
    min_volume_24h_usdt: float = 5_000_000
    max_volume_24h_usdt: float = 1_500_000_000

    # Памп/дамп-монеты = тренд = убыток маркет-мейкера.
    max_abs_change_24h_pct: float = 25.0
    # Доля направленного движения за последний час (efficiency ratio):
    # |сумма доходностей| / сумма |доходностей|. 1 = чистый тренд.
    max_trend_ratio: float = 0.35

    min_spread_bps: float = 3.0         # медианный спред
    min_trades_per_min: float = 10.0    # сделок в минуту (по 1m свечам)
    max_sigma_1m_bps: float = 35.0      # волатильность 1-минутных доходностей
    max_abs_funding: float = 0.0005     # 0.05% за период

    # Оценка: net_bps = spread/2 − maker_fee − adverse_coef·σ_1m
    # score = net_bps · sqrt(trades_per_min)
    adverse_coef: float = 0.10
    min_net_bps: float = 0.5

    spread_samples: int = 5             # сколько снимков bookTicker брать
    spread_sample_interval_sec: float = 1.0
    max_kline_candidates: int = 40      # для скольких монет грузить свечи

    rescan_interval_sec: int = 900
    empty_rescan_sec: int = 60          # если монет нет — повтор не чаще
    rotate_score_mult: float = 1.3      # новая монета должна быть лучше на 30%


@dataclass
class QuoteConfig:
    order_usdt: float = 6.0             # номинал одной заявки на вход

    # Минимальная прибыль на исполнение после комиссии мейкера
    min_edge_bps: float = 1.0
    # Ширина половины спреда ≥ vol_mult · σ_1s · sqrt(vol_horizon_s):
    # защита от движения цены за время нашей реакции (65 мс + обработка).
    vol_horizon_s: float = 2.0
    vol_mult: float = 1.0
    # Половина спреда ≥ fee + min_edge + adverse_mult · измеренный adverse
    adverse_mult: float = 1.0
    max_half_spread_bps: float = 40.0

    # Сдвиг справедливой цены (alpha), bps
    k_micro: float = 1.0                # вес отклонения microprice от mid
    k_depth_imb_bps: float = 1.0        # дисбаланс 5 уровней стакана, [-1..1]
    k_flow_bps: float = 1.5             # дисбаланс потока сделок, [-1..1]
    max_alpha_bps: float = 4.0
    flow_halflife_s: float = 2.0

    # Управление инвентарём
    skew_max_bps: float = 6.0           # сдвиг котировок при полной позиции
    time_skew_bps_per_min: float = 3.0  # доп. сдвиг за каждую минуту удержания
    # Первые N секунд выход не дешевле цены входа + комиссии (дальше — по рынку)
    exit_patience_s: float = 30.0

    # Перекотировка только при заметном изменении — сохраняем место в очереди
    # и не тратим лимиты API.
    requote_bps: float = 1.0
    requote_min_interval_ms: int = 200
    loop_timeout_ms: int = 250

    latency_bad_ms: float = 250.0       # при большей задержке спред ×1.5


@dataclass
class RiskConfig:
    max_inventory_usdt: float = 18.0
    stop_loss_bps: float = 35.0         # против avg_entry → выход по рынку
    max_hold_s: float = 180.0           # дольше держать нельзя → по рынку
    reentry_cooldown_s: float = 10.0    # пауза входов после стопа

    # Шоки
    btc_symbol: str = "BTCUSDT"
    btc_move_bps: float = 40.0
    btc_window_s: float = 3.0
    spread_shock_mult: float = 4.0      # спред > 4× медианы
    vol_shock_mult: float = 4.0         # быстрая σ > 4× медленной
    shock_pause_s: float = 20.0

    # Токсичность потока (markout-контроль)
    markout_horizons_s: List[float] = field(default_factory=lambda: [1.0, 5.0, 30.0])
    toxic_horizon_s: float = 5.0
    toxic_window: int = 40
    toxic_min_fills: int = 15
    toxic_net_bps: float = -1.0         # средний net хуже → монета токсична
    toxic_pause_s: float = 600.0
    adverse_halflife_fills: float = 20.0

    funding_blackout_s: float = 30.0    # не открываться перед funding

    symbol_max_loss_usdt: float = 1.5   # убыток по монете → ротация
    session_max_loss_pct: float = 0.05  # доля баланса → стоп бота

    max_orders_per_10s: int = 200       # лимит Binance 300/10с
    position_sync_s: float = 5.0


@dataclass
class PaperConfig:
    balance_usdt: float = 100.0
    latency_ms: float = 65.0
    # Консервативная модель: заявка исполняется, только если сделка прошла
    # СКВОЗЬ нашу цену (на нашей цене стояла очередь, мы в её конце).
    touch_fill_prob: float = 0.0


@dataclass
class LogConfig:
    dir: str = "logs"
    status_interval_s: int = 60


@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    quote: QuoteConfig = field(default_factory=QuoteConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    log: LogConfig = field(default_factory=LogConfig)

    paper_mode: bool = True             # по умолчанию — бумажная торговля
    symbols: List[str] = field(default_factory=list)  # пусто = сканер
    max_symbols: int = 1
    flatten_on_exit: bool = True
