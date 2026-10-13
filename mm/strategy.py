"""
Расчёт котировок — чистая функция без I/O, поэтому легко тестируется.

Логика на каждое обновление рынка:
1. Справедливая цена fair = mid · (1 + alpha), где alpha — краткосрочный прогноз
   из microprice, дисбаланса стакана и потока агрессивных сделок.
2. Цена резервирования res = fair · (1 − skew): при длинной позиции котировки
   сдвигаются вниз (охотнее продаём, неохотнее покупаем), при короткой — вверх.
   Сдвиг растёт с размером позиции и временем её удержания.
3. Половина спреда = max(
       комиссия + минимальная прибыль + измеренный adverse selection,
       σ за время реакции,
   ), ограничена сверху.
4. Заявка на сокращение позиции — reduceOnly на весь объём позиции.
   Первые exit_patience_s секунд она не дешевле входа + комиссий.
5. Заявки post-only: никогда не пересекают противоположную лучшую цену.
"""

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Optional

from .config import QuoteConfig


@dataclass
class SymbolFilters:
    tick: float
    step: float
    min_qty: float
    min_notional: float


def round_to(value: float, step: float, up: bool) -> float:
    if step <= 0:
        return value
    d = Decimal(repr(value)) / Decimal(repr(step))
    n = d.to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR)
    return float(n * Decimal(repr(step)))


def fmt(value: float, step: float) -> str:
    """Десятичная строка, кратная step (без экспоненты)."""
    d = Decimal(repr(value)).quantize(Decimal(repr(step)).normalize()) \
        if step > 0 else Decimal(repr(value))
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


@dataclass
class QuoteInputs:
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    depth_imb: float
    flow_imb: float
    sigma_1s_bps: float
    adverse_bps: float
    position: float            # количество монет со знаком
    avg_entry: float
    hold_s: float              # сколько держим текущую позицию
    allow_open: bool           # можно ли наращивать позицию
    latency_bad: bool
    maker_fee_bps: float
    max_inventory_usdt: float


@dataclass
class Side:
    price: float
    qty: float
    reduce_only: bool


@dataclass
class Quote:
    bid: Optional[Side]
    ask: Optional[Side]
    fair: float
    res: float
    half_bps: float
    alpha_bps: float
    skew_bps: float


def compute_quote(inp: QuoteInputs, f: SymbolFilters, cfg: QuoteConfig) -> Optional[Quote]:
    if inp.bid <= 0 or inp.ask <= inp.bid:
        return None
    mid = (inp.bid + inp.ask) / 2.0

    # 1. alpha
    tot = inp.bid_qty + inp.ask_qty
    micro = (inp.bid * inp.ask_qty + inp.ask * inp.bid_qty) / tot if tot > 0 else mid
    alpha = (
        cfg.k_micro * (micro - mid) / mid * 1e4
        + cfg.k_depth_imb_bps * inp.depth_imb
        + cfg.k_flow_bps * inp.flow_imb
    )
    alpha = max(-cfg.max_alpha_bps, min(cfg.max_alpha_bps, alpha))
    fair = mid * (1 + alpha / 1e4)

    # 2. инвентарь
    inv_usdt = inp.position * mid
    inv_ratio = 0.0
    if inp.max_inventory_usdt > 0:
        inv_ratio = max(-1.0, min(1.0, inv_usdt / inp.max_inventory_usdt))
    skew = cfg.skew_max_bps * inv_ratio
    if inp.position != 0:
        skew += math.copysign(cfg.time_skew_bps_per_min * inp.hold_s / 60.0, inp.position)
    skew = max(-3 * cfg.skew_max_bps, min(3 * cfg.skew_max_bps, skew))
    res = fair * (1 - skew / 1e4)

    # 3. ширина
    fee_floor = inp.maker_fee_bps + cfg.min_edge_bps + cfg.adverse_mult * max(0.0, inp.adverse_bps)
    vol_floor = cfg.vol_mult * inp.sigma_1s_bps * math.sqrt(cfg.vol_horizon_s)
    half = max(fee_floor, vol_floor)
    if inp.latency_bad:
        half *= 1.5
    half = min(half, cfg.max_half_spread_bps)

    bid_px = res * (1 - half / 1e4)
    ask_px = res * (1 + half / 1e4)

    # 4. защита выхода в первые exit_patience_s секунд
    if inp.position > 0 and inp.avg_entry > 0 and inp.hold_s < cfg.exit_patience_s:
        ask_px = max(ask_px, inp.avg_entry * (1 + (2 * inp.maker_fee_bps + cfg.min_edge_bps) / 1e4))
    if inp.position < 0 and inp.avg_entry > 0 and inp.hold_s < cfg.exit_patience_s:
        bid_px = min(bid_px, inp.avg_entry * (1 - (2 * inp.maker_fee_bps + cfg.min_edge_bps) / 1e4))

    # 5. post-only: не пересекать противоположную сторону
    bid_px = min(bid_px, inp.ask - f.tick)
    ask_px = max(ask_px, inp.bid + f.tick)
    bid_px = round_to(bid_px, f.tick, up=False)
    ask_px = round_to(ask_px, f.tick, up=True)

    # Размеры
    base_qty = cfg.order_usdt / mid
    pos = inp.position
    pos_abs = abs(pos)

    def opening(px: float, ratio_same_side: float) -> Optional[Side]:
        if not inp.allow_open or px <= 0:
            return None
        scale = 1.0 - max(0.0, ratio_same_side)
        q = round_to(base_qty * scale, f.step, up=False)
        if q < f.min_qty or q * px < f.min_notional:
            return None
        return Side(px, q, False)

    def reducing(px: float) -> Optional[Side]:
        q = round_to(pos_abs, f.step, up=False)
        if q < f.min_qty or px <= 0:
            return None
        return Side(px, q, True)

    # Для размера берём позицию по цене входа, а не по mid: иначе объём
    # менялся бы на каждом тике и заявка постоянно перевыставлялась.
    size_ratio = 0.0
    if inp.max_inventory_usdt > 0:
        ref = inp.avg_entry if inp.avg_entry > 0 else mid
        size_ratio = max(-1.0, min(1.0, pos * ref / inp.max_inventory_usdt))

    if pos > 0:
        bid = opening(bid_px, size_ratio)
        ask = reducing(ask_px)
    elif pos < 0:
        bid = reducing(bid_px)
        ask = opening(ask_px, -size_ratio)
    else:
        bid = opening(bid_px, 0.0)
        ask = opening(ask_px, 0.0)

    return Quote(bid, ask, fair, res, half, alpha, skew)


def needs_requote(cur_px: float, cur_qty: float, cur_reduce: bool,
                  new: Side, tick: float, requote_bps: float) -> bool:
    if cur_reduce != new.reduce_only:
        return True
    if abs(cur_qty - new.qty) > 1e-12:
        return True
    threshold = max(tick, cur_px * requote_bps / 1e4)
    return abs(cur_px - new.price) >= threshold - 1e-12
