"""
Сканер монет для маркет-мейкинга.

Критерии:
- NATR 5m: 2.5–5%
- Volume 24h ≥ $200M
- Price change 24h ≥ +20%
"""

import logging
import asyncio
from typing import List, Dict
import numpy as np

from binance import AsyncClient
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class SymbolInfo:
    __slots__ = ("symbol", "volume_24h", "price_change_pct",
                 "last_price", "natr_5m", "score")

    def __init__(self):
        self.symbol = ""
        self.volume_24h = 0.0
        self.price_change_pct = 0.0
        self.last_price = 0.0
        self.natr_5m = 0.0
        self.score = 0.0

    def __repr__(self):
        return (f"{self.symbol:14s} | vol=${self.volume_24h/1e6:6.0f}M | "
                f"chg={self.price_change_pct:+6.1f}% | "
                f"NATR5m={self.natr_5m:.2f}% | score={self.score:.2f}")


class SymbolScanner:
    def __init__(self, client: AsyncClient, config):
        self.client = client
        self.cfg = config

    async def scan(self) -> List[SymbolInfo]:
        logger.info("🔍 Сканирую рынок...")
        tickers = await self._fetch_tickers_24h()
        if not tickers:
            return []

        candidates = self._filter_candidates(tickers)
        logger.info(f"После фильтра (vol≥${self.cfg.scanner.min_volume_24h_usdt/1e6:.0f}M, "
                    f"chg≥{self.cfg.scanner.min_price_change_24h}%): "
                    f"{len(candidates)} кандидатов")

        await self._enrich_natr(candidates)

        s = self.cfg.scanner
        valid = [c for c in candidates
                 if s.natr_min_pct <= c.natr_5m <= s.natr_max_pct]
        logger.info(
            f"Прошли NATR-фильтр ({s.natr_min_pct}-{s.natr_max_pct}%): "
            f"{len(valid)}"
        )

        for c in valid:
            c.score = c.natr_5m
        valid.sort(key=lambda x: x.score, reverse=True)

        logger.info("=== Топ кандидатов ===")
        for c in valid[:10]:
            logger.info(f"  {c}")

        return valid

    async def _fetch_tickers_24h(self) -> Dict[str, dict]:
        try:
            tickers = await self.client.futures_ticker()
            return {t["symbol"]: t for t in tickers}
        except BinanceAPIException as e:
            logger.error(f"ticker error: {e}")
            return {}

    def _filter_candidates(self, tickers: Dict[str, dict]) -> List[SymbolInfo]:
        cfg = self.cfg.scanner
        out = []
        for sym, t in tickers.items():
            if not sym.endswith(cfg.quote_asset):
                continue
            if sym in cfg.exclude_symbols:
                continue
            try:
                vol = float(t["quoteVolume"])
                chg = float(t["priceChangePercent"])
                price = float(t["lastPrice"])
            except (KeyError, ValueError):
                continue
            if vol < cfg.min_volume_24h_usdt:
                continue
            if chg < cfg.min_price_change_24h:
                continue
            info = SymbolInfo()
            info.symbol = sym
            info.volume_24h = vol
            info.price_change_pct = chg
            info.last_price = price
            out.append(info)
        return out

    async def _enrich_natr(self, candidates: List[SymbolInfo]) -> None:
        sem = asyncio.Semaphore(8)

        async def fetch(info: SymbolInfo):
            async with sem:
                try:
                    k = await self.client.futures_klines(
                        symbol=info.symbol, interval="5m", limit=20
                    )
                    if len(k) < 2:
                        return
                    # исключаем текущую незавершённую свечу
                    completed = k[:-1]
                    info.natr_5m = self.compute_natr_5m(completed)
                except Exception as e:
                    logger.debug(f"NATR {info.symbol}: {e}")

        await asyncio.gather(*[fetch(c) for c in candidates])

    @staticmethod
    def compute_natr_5m(klines) -> float:
        """
        NATR = (ATR(14) / Close) * 100.
        klines — список completed 5m-свечей (>=15 штук).
        """
        if len(klines) < 15:
            return 0.0
        k = klines[-15:]
        highs = np.array([float(x[2]) for x in k])
        lows = np.array([float(x[3]) for x in k])
        closes = np.array([float(x[4]) for x in k])

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr = float(np.mean(tr))
        if closes[-1] <= 0:
            return 0.0
        return (atr / closes[-1]) * 100.0
