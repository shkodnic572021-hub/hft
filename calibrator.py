import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MarketCalibrator:
    def __init__(self, window_size: int = 1000):
        self.prices = deque(maxlen=window_size)
        self.spreads = deque(maxlen=window_size)
        self.sigma = 0.0
        self.kappa = 1.5

    def update(self, mid_price: float, spread: float):
        self.prices.append(mid_price)
        if spread > 0:
            self.spreads.append(spread)

        if len(self.prices) > 100:
            prices_arr = np.array(self.prices)
            log_returns = np.diff(np.log(prices_arr))
            self.sigma = np.std(log_returns)

        if len(self.spreads) > 100:
            avg_spread = np.mean(self.spreads)
            if avg_spread > 0:
                self.kappa = 1.0 / avg_spread

    def get_params(self):
        return self.sigma, self.kappa
