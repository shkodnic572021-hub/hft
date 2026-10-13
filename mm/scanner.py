"""
Отбор монет для маркет-мейкинга.

Маркет-мейкер зарабатывает спред и теряет на направленных движениях.
Поэтому ищем НЕ «самые волатильные/растущие» монеты (это ловушка: тренд
съедает спред), а монеты, где:
  - спред широкий относительно комиссии (есть что зарабатывать);
  - много сделок (заявки исполняются);
  - волатильность умеренная и без тренда (меньше adverse selection);
  - объём не топовый (на BTC/ETH/SOL очередь у лучшей цены занята
    колоцированными HFT, с задержкой 65 мс там не выиграть);
  - funding не экстремальный.

Оценка: net_bps = spread/2 − maker_fee − adverse_coef·σ_1m,
        score   = net_bps · sqrt(trades_per_min).
Это априорный фильтр. Реальную прибыльность монеты показывает markout
после исполнений — по нему бот сам уходит с плохих монет.
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List

import numpy as np

from .config import BotConfig

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    symbol: str
    volume_24h: float = 0.0
    change_24h: float = 0.0
    spread_bps: float = 0.0
    spread_ticks: float = 0.0
    trades_per_min: float = 0.0
    sigma_1m_bps: float = 0.0
    trend: float = 0.0
    funding: float = 0.0
    net_bps: float = 0.0
    score: float = 0.0

    def __str__(self):
        return (f"{self.symbol:14s} score={self.score:6.2f} net={self.net_bps:5.2f}bps "
                f"spread={self.spread_bps:5.2f}bps ({self.spread_ticks:.1f}t) "
                f"trades/min={self.trades_per_min:6.0f} σ1m={self.sigma_1m_bps:5.1f}bps "
                f"trend={self.trend:.2f} vol=${self.volume_24h/1e6:.0f}M "
                f"chg={self.change_24h:+.1f}% fund={self.funding*100:+.3f}%")


def score_candidate(c: Candidate, maker_fee_bps: float, adverse_coef: float) -> float:
    c.net_bps = c.spread_bps / 2 - maker_fee_bps - adverse_coef * c.sigma_1m_bps
    c.score = c.net_bps * math.sqrt(max(c.trades_per_min, 0.0)) if c.net_bps > 0 else 0.0
    return c.score


def kline_stats(klines) -> Dict[str, float]:
    """σ 1-минутных лог-доходностей (bps), тренд и сделок в минуту по 1m свечам."""
    closes = np.array([float(k[4]) for k in klines])
    trades = np.array([float(k[8]) for k in klines])
    if len(closes) < 10 or np.any(closes <= 0):
        return {"sigma": 0.0, "trend": 1.0, "tpm": 0.0}
    r = np.diff(np.log(closes)) * 1e4
    abs_sum = float(np.sum(np.abs(r)))
    trend = abs(float(np.sum(r))) / abs_sum if abs_sum > 0 else 1.0
    return {"sigma": float(np.std(r)), "trend": trend, "tpm": float(np.median(trades))}


class Scanner:
    def __init__(self, client, cfg: BotConfig, filters):
        self.client = client
        self.cfg = cfg
        self.filters = filters      # callable symbol -> SymbolFilters | None

    async def scan(self) -> List[Candidate]:
        s = self.cfg.scanner
        maker = self.cfg.exchange.maker_fee_bps
        tickers = await self.client.futures_ticker()
        premium = await self.client.futures_mark_price()
        funding = {p["symbol"]: float(p.get("lastFundingRate", 0) or 0) for p in premium}

        # Воронка: сколько монет отсеяно на каждом шаге — чтобы было видно,
        # какой фильтр «душит» отбор на текущем рынке.
        funnel: Dict[str, int] = {}

        def drop(reason: str):
            funnel[reason] = funnel.get(reason, 0) + 1

        cands: Dict[str, Candidate] = {}
        total = 0
        for t in tickers:
            sym = t["symbol"]
            if not sym.endswith(s.quote_asset) or sym in s.exclude_symbols:
                continue
            if self.filters(sym) is None:
                continue
            total += 1
            try:
                vol = float(t["quoteVolume"])
                chg = float(t["priceChangePercent"])
            except (KeyError, ValueError):
                continue
            if vol < s.min_volume_24h_usdt:
                drop("объём мал")
                continue
            if vol > s.max_volume_24h_usdt:
                drop("объём велик (топ)")
                continue
            if abs(chg) > s.max_abs_change_24h_pct:
                drop("памп/дамп 24ч")
                continue
            f = funding.get(sym, 0.0)
            if abs(f) > s.max_abs_funding:
                drop("funding")
                continue
            cands[sym] = Candidate(sym, vol, chg, funding=f)

        # Медианный спред по нескольким снимкам bookTicker
        samples: Dict[str, List[float]] = {k: [] for k in cands}
        ticks: Dict[str, List[float]] = {k: [] for k in cands}
        for i in range(s.spread_samples):
            if i:
                await asyncio.sleep(s.spread_sample_interval_sec)
            for bt in await self.client.futures_orderbook_ticker():
                sym = bt.get("symbol")
                if sym not in cands:
                    continue
                try:
                    b, a = float(bt["bidPrice"]), float(bt["askPrice"])
                except (KeyError, ValueError):
                    continue
                if b > 0 and a > b:
                    samples[sym].append((a - b) / ((a + b) / 2) * 1e4)
                    ticks[sym].append((a - b) / self.filters(sym).tick)
        spread_ok: List[Candidate] = []
        near_miss: List[Candidate] = []
        for sym, c in cands.items():
            if not samples[sym]:
                continue
            c.spread_bps = median(samples[sym])
            c.spread_ticks = median(ticks[sym])
            if c.spread_bps < s.min_spread_bps:
                drop(f"спред < {s.min_spread_bps:g}bps")
                near_miss.append(c)
            else:
                spread_ok.append(c)

        # Свечи — только для лучших по предварительной оценке (лимиты API)
        pre = sorted(spread_ok, key=lambda c: c.spread_bps, reverse=True)
        for c in pre[s.max_kline_candidates:]:
            drop("вне лимита свечей")
        pre = pre[: s.max_kline_candidates]
        sem = asyncio.Semaphore(8)

        async def enrich(c: Candidate):
            async with sem:
                try:
                    k = await self.client.futures_klines(symbol=c.symbol, interval="1m", limit=61)
                except Exception as e:
                    logger.debug(f"{c.symbol}: klines {e}")
                    return
            st = kline_stats(k[:-1])
            c.sigma_1m_bps, c.trend, c.trades_per_min = st["sigma"], st["trend"], st["tpm"]

        await asyncio.gather(*[enrich(c) for c in pre])

        out = []
        rejected: List[tuple] = []
        for c in pre:
            score_candidate(c, maker, s.adverse_coef)
            if c.trades_per_min < s.min_trades_per_min:
                reason = f"сделок/мин < {s.min_trades_per_min:g}"
            elif c.sigma_1m_bps <= 0 or c.sigma_1m_bps > s.max_sigma_1m_bps:
                reason = f"σ1m > {s.max_sigma_1m_bps:g}bps"
            elif c.trend > s.max_trend_ratio:
                reason = f"тренд > {s.max_trend_ratio:g}"
            elif c.net_bps < s.min_net_bps:
                reason = "спред не окупает комиссию"
            else:
                out.append(c)
                continue
            drop(reason)
            rejected.append((c, reason))
        out.sort(key=lambda c: c.score, reverse=True)

        logger.info(
            f"🔍 Сканер: {total} монет → прошли {len(out)} | отсеяно: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(funnel.items(), key=lambda x: -x[1]))
        )
        for c in out[:10]:
            logger.info(f"   ✅ {c}")
        if not out:
            # Показать, кто был ближе всего — по этим данным видно,
            # какой порог мешает на текущем рынке.
            for c, reason in sorted(rejected, key=lambda x: -x[0].spread_bps)[:5]:
                logger.info(f"   ✗ {reason}: {c}")
            for c in sorted(near_miss, key=lambda c: -c.spread_bps)[:5]:
                logger.info(f"   ✗ спред: {c.symbol} {c.spread_bps:.2f}bps "
                            f"({c.spread_ticks:.1f} тика) vol=${c.volume_24h/1e6:.0f}M")
        return out
