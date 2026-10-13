import csv
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Порог, ниже которого остаток позиции считаем нулём (накопление
# ошибок float при сложении дробных количеств).
QTY_EPS = 1e-9


@dataclass
class SymbolPnL:
    symbol: str = ""
    position_qty: float = 0.0
    avg_entry: float = 0.0
    last_mid: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees: float = 0.0
    total_fills: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    started_at: float = field(default_factory=time.time)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_pnl(self) -> float:
        return self.total_pnl - self.total_fees

    @property
    def runtime_min(self) -> float:
        return (time.time() - self.started_at) / 60.0


class PnLTracker:
    def __init__(self, cfg, maker_fee_pct: float = 0.02):
        self.cfg = cfg
        self.maker_fee_pct = maker_fee_pct
        self.states: Dict[str, SymbolPnL] = {}
        self._last_log_ts = 0.0
        self._init_csv()

    def _init_csv(self):
        path = self.cfg.csv_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "symbol", "position_qty", "avg_entry",
                    "mid_price", "realized_pnl", "unrealized_pnl",
                    "fees", "net_pnl", "total_fills",
                    "buy_volume", "sell_volume"
                ])
            logger.info(f"📄 P&L CSV создан: {path}")

    def get_state(self, symbol: str) -> SymbolPnL:
        if symbol not in self.states:
            self.states[symbol] = SymbolPnL(symbol=symbol)
        return self.states[symbol]

    def sync_position(self, symbol: str, position_qty: float, entry_price: float):
        """Выравнивает позицию и цену входа по данным биржи."""
        s = self.get_state(symbol)
        if abs(position_qty) < QTY_EPS:
            s.position_qty = 0.0
            s.avg_entry = 0.0
            s.unrealized_pnl = 0.0
            return
        s.position_qty = position_qty
        if entry_price > 0:
            s.avg_entry = entry_price

    def on_fill(
        self, symbol: str, side: str, fill_qty: float, fill_price: float,
        fee: Optional[float] = None,
    ) -> float:
        """
        Учитывает одну сделку. fill_qty/fill_price — объём и цена именно
        этой сделки (поля l/L в ORDER_TRADE_UPDATE). fee — фактическая
        комиссия в USDT; если None, оценивается по maker_fee_pct.
        """
        s = self.get_state(symbol)
        signed_qty = fill_qty if side == "BUY" else -fill_qty
        s.total_fills += 1

        notional = fill_qty * fill_price
        if fee is None:
            fee = notional * (self.maker_fee_pct / 100.0)
        s.total_fees += fee

        if side == "BUY":
            s.buy_volume += notional
        else:
            s.sell_volume += notional

        realized_now = 0.0

        if abs(s.position_qty) < QTY_EPS:
            s.avg_entry = fill_price
        elif (s.position_qty > 0 and signed_qty > 0) or \
             (s.position_qty < 0 and signed_qty < 0):
            old_abs = abs(s.position_qty)
            new_abs = old_abs + fill_qty
            s.avg_entry = (s.avg_entry * old_abs + fill_price * fill_qty) / new_abs
        else:
            close_qty = min(abs(s.position_qty), fill_qty)
            if s.position_qty > 0:
                realized_now = (fill_price - s.avg_entry) * close_qty
            else:
                realized_now = (s.avg_entry - fill_price) * close_qty
            s.realized_pnl += realized_now
            remaining = fill_qty - close_qty
            if remaining > QTY_EPS:
                s.avg_entry = fill_price

        s.position_qty += signed_qty
        if abs(s.position_qty) < QTY_EPS:
            s.position_qty = 0.0
            s.avg_entry = 0.0
            s.unrealized_pnl = 0.0

        logger.debug(
            f"PnL {symbol}: realized_now={realized_now:+.4f} | "
            f"total_realized={s.realized_pnl:+.4f} | pos={s.position_qty:.6f}"
        )
        return realized_now

    def update_unrealized(self, symbol: str, mid_price: float):
        s = self.get_state(symbol)
        if mid_price > 0:
            s.last_mid = mid_price
        if abs(s.position_qty) < QTY_EPS or s.avg_entry <= 0 or mid_price <= 0:
            s.unrealized_pnl = 0.0
            return
        if s.position_qty > 0:
            s.unrealized_pnl = (mid_price - s.avg_entry) * s.position_qty
        else:
            s.unrealized_pnl = (s.avg_entry - mid_price) * abs(s.position_qty)

    def session_net_pnl(self) -> float:
        return sum(s.net_pnl for s in self.states.values())

    def log_if_due(self, prices: Dict[str, float]):
        now = time.time()
        if now - self._last_log_ts < self.cfg.log_interval_sec:
            return
        self._last_log_ts = now

        for symbol, mid in prices.items():
            self.update_unrealized(symbol, mid)
            self._log_symbol(symbol)

    def _log_symbol(self, symbol: str):
        s = self.states.get(symbol)
        if not s:
            return

        logger.info(
            f"📊 {symbol} | pos={s.position_qty:+.6f} | "
            f"entry={s.avg_entry:.4f} | "
            f"realized={s.realized_pnl:+.4f} | "
            f"unrealized={s.unrealized_pnl:+.4f} | "
            f"fees={s.total_fees:.4f} | "
            f"NET={s.net_pnl:+.4f} | "
            f"fills={s.total_fills}"
        )

        try:
            with open(self.cfg.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    int(time.time()), symbol, f"{s.position_qty:.8f}",
                    f"{s.avg_entry:.6f}", f"{s.last_mid:.6f}",
                    f"{s.realized_pnl:.6f}", f"{s.unrealized_pnl:.6f}",
                    f"{s.total_fees:.6f}", f"{s.net_pnl:.6f}",
                    s.total_fills, f"{s.buy_volume:.2f}", f"{s.sell_volume:.2f}"
                ])
        except Exception as e:
            logger.error(f"CSV write error: {e}")

    def force_log(self, symbol: str, mid_price: float):
        self.update_unrealized(symbol, mid_price)
        self._log_symbol(symbol)

    def print_summary(self):
        if not self.states:
            logger.info("P&L: нет данных.")
            return

        logger.info("=" * 70)
        logger.info("📈 ИТОГОВЫЙ P&L СЕССИИ")
        logger.info("=" * 70)

        total_realized = 0.0
        total_unrealized = 0.0
        total_fees = 0.0
        total_fills = 0
        total_buy_vol = 0.0
        total_sell_vol = 0.0

        for symbol, s in self.states.items():
            logger.info(
                f"  {symbol:12s} | fills={s.total_fills:4d} | "
                f"realized={s.realized_pnl:+9.4f} | "
                f"unrealized={s.unrealized_pnl:+9.4f} | "
                f"fees={s.total_fees:7.4f} | "
                f"NET={s.net_pnl:+9.4f} | "
                f"runtime={s.runtime_min:.1f}м"
            )
            total_realized += s.realized_pnl
            total_unrealized += s.unrealized_pnl
            total_fees += s.total_fees
            total_fills += s.total_fills
            total_buy_vol += s.buy_volume
            total_sell_vol += s.sell_volume

        logger.info("-" * 70)
        logger.info(f"  Всего fills:      {total_fills}")
        logger.info(f"  Buy volume:      ${total_buy_vol:.2f}")
        logger.info(f"  Sell volume:     ${total_sell_vol:.2f}")
        logger.info(f"  Realized P&L:    {total_realized:+.4f} USDT")
        logger.info(f"  Unrealized P&L:  {total_unrealized:+.4f} USDT")
        logger.info(f"  Fees:            {total_fees:.4f} USDT")
        logger.info(f"  ══════════════════════════════════")
        logger.info(f"  NET P&L:         {total_realized + total_unrealized - total_fees:+.4f} USDT")
        logger.info("=" * 70)
