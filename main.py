"""
Маркет-мейкер Binance USDⓈ-M.

  python main.py                      # бумажная торговля, монеты выбирает сканер
  python main.py --symbols DOGEUSDT   # бумажная торговля на заданной монете
  python main.py --scan               # только показать рейтинг монет
  python main.py --live               # реальная торговля (нужен key.env)
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler

from mm.bot import Bot
from mm.config import BotConfig


def setup_logging(level: str, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(os.path.join(log_dir, "bot.log"), maxBytes=50_000_000,
                             backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="Binance USDⓈ-M market maker")
    p.add_argument("--live", action="store_true", help="реальная торговля (по умолчанию paper)")
    p.add_argument("--scan", action="store_true", help="только просканировать рынок и выйти")
    p.add_argument("--symbols", default="", help="монеты через запятую (иначе сканер)")
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--order-usdt", type=float, default=None)
    p.add_argument("--max-inventory-usdt", type=float, default=None)
    p.add_argument("--paper-balance", type=float, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def run_scan(cfg: BotConfig):
    from mm.exchange import PaperExchange
    from mm.scanner import Scanner
    ex = PaperExchange(cfg, {})
    await ex.start()
    try:
        await Scanner(ex.client, cfg, ex.filters).scan()
    finally:
        await ex.close()


async def main():
    args = parse_args()
    cfg = BotConfig()
    cfg.paper_mode = not args.live
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.max_symbols:
        cfg.max_symbols = args.max_symbols
    elif cfg.symbols:
        cfg.max_symbols = len(cfg.symbols)
    if args.order_usdt:
        cfg.quote.order_usdt = args.order_usdt
    if args.max_inventory_usdt:
        cfg.risk.max_inventory_usdt = args.max_inventory_usdt
    if args.paper_balance:
        cfg.paper.balance_usdt = args.paper_balance

    setup_logging(args.log_level, cfg.log.dir)
    log = logging.getLogger("main")

    if args.scan:
        await run_scan(cfg)
        return

    if not cfg.paper_mode and not (cfg.exchange.api_key and cfg.exchange.api_secret):
        log.error("❌ Для --live нужны BINANCE_API_KEY и BINANCE_API_SECRET в key.env")
        return

    e, q, r = cfg.exchange, cfg.quote, cfg.risk
    log.info("=" * 70)
    log.info(f"MARKET MAKER | режим: {'LIVE' if not cfg.paper_mode else 'PAPER'}"
             f"{' (testnet)' if e.testnet else ''}")
    log.info(f"Монеты: {', '.join(cfg.symbols) if cfg.symbols else 'сканер'} | max {cfg.max_symbols}")
    log.info(f"Заявка ${q.order_usdt} | макс. позиция ${r.max_inventory_usdt} | плечо {e.leverage}x")
    log.info(f"Комиссии: maker {e.maker_fee_bps}bps taker {e.taker_fee_bps}bps | "
             f"мин. прибыль {q.min_edge_bps}bps/fill")
    log.info(f"Стоп {r.stop_loss_bps}bps | max hold {r.max_hold_s:.0f}с | "
             f"стоп сессии {r.session_max_loss_pct*100:.0f}% баланса")
    log.info("=" * 70)

    bot = Bot(cfg)
    loop = asyncio.get_running_loop()

    def _stop():
        log.warning("🛑 Сигнал остановки")
        bot.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except (NotImplementedError, RuntimeError):
            pass      # Windows: Ctrl+C придёт как KeyboardInterrupt
    try:
        await bot.run()
    except asyncio.CancelledError:
        pass
    finally:
        if not bot.stopping:
            await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Остановлено пользователем")
