import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MarketCalibrator:
    """
    Оценка параметров модели Авельянеды–Стойкова по потоку стакана.

    sigma — std лог-доходностей mid между соседними обновлениями стакана.
    kappa — 1 / средний ОТНОСИТЕЛЬНЫЙ спред (spread / mid). Относительные
    единицы нужны, чтобы kappa не зависела от цены монеты: модель считает
    ln(1 + gamma / kappa) и потом умножает на mid.
    """

    MIN_SAMPLES = 100

    def __init__(self, window_size: int = 1000):
        self.prices = deque(maxlen=window_size)
        self.rel_spreads = deque(maxlen=window_size)
        self.sigma = 0.0
        self.kappa = 0.0

    @property
    def ready(self) -> bool:
        return self.sigma > 0 and self.kappa > 0

    def update(self, mid_price: float, spread: float):
        if mid_price <= 0:
            return
        self.prices.append(mid_price)
        if spread > 0:
            self.rel_spreads.append(spread / mid_price)

        if len(self.prices) > self.MIN_SAMPLES:
            prices_arr = np.array(self.prices)
            log_returns = np.diff(np.log(prices_arr))
            self.sigma = float(np.std(log_returns))

        if len(self.rel_spreads) > self.MIN_SAMPLES:
            avg_spread = float(np.mean(self.rel_spreads))
            if avg_spread > 0:
                self.kappa = 1.0 / avg_spread

    def get_params(self):
        return self.sigma, self.kappa
