import threading
from config import CTRADER_CONFIG

_broker = None
_broker_lock = threading.Lock()


class BrokerWrapper:
    """cTrader broker wrapper with unified API."""

    TIMEFRAME_D1 = 16408
    TIMEFRAME_H4 = 16420
    TIMEFRAME_H1 = 16388
    TIMEFRAME_M15 = 16385
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(self):
        self._impl = None

    def initialize(self) -> bool:
        from ctrader_client import CTraderClient
        self._impl = CTraderClient(
            host=CTRADER_CONFIG["host"],
            port=CTRADER_CONFIG["port"],
            client_id=CTRADER_CONFIG["client_id"],
            client_secret=CTRADER_CONFIG["client_secret"],
            access_token=CTRADER_CONFIG["access_token"],
            account_id=int(CTRADER_CONFIG["account_id"]),
        )
        return self._impl.initialize()

    def shutdown(self):
        if self._impl is not None:
            self._impl.shutdown()
            self._impl = None

    def account_info(self):
        return self._impl.account_info()

    def symbol_info(self, symbol: str):
        return self._impl.symbol_info(symbol)

    def symbol_info_tick(self, symbol: str):
        return self._impl.symbol_info_tick(symbol)

    def positions_get(self, symbol: str = None):
        return self._impl.positions_get(symbol=symbol)

    def copy_rates_from_pos(self, symbol: str, timeframe, start_pos: int, count: int):
        return self._impl.copy_rates_from_pos(symbol, timeframe, start_pos, count)

    def order_send(self, request: dict):
        return self._impl.order_send(request)

    def close_position(self, position_ticket, symbol=None):
        return self._impl.close_position(position_ticket, symbol=symbol)

    def modify_position(self, position_ticket, sl=None, tp=None):
        return self._impl.modify_position(position_ticket, sl=sl, tp=tp)

    def get_symbol_digits(self, symbol: str) -> int:
        info = self.symbol_info(symbol)
        if info and isinstance(info, dict):
            return info.get("digits", 5)
        return 5

    def get_symbol_point(self, symbol: str) -> float:
        digits = self.get_symbol_digits(symbol)
        if digits == 3:
            return 0.01
        return 0.0001

    def get_pip_value(self, symbol: str) -> float:
        digits = self.get_symbol_digits(symbol)
        if digits == 3:
            return 0.01
        return 0.0001

    @property
    def is_connected(self):
        return self._impl is not None and self._impl.is_connected


def init_broker() -> bool:
    global _broker
    with _broker_lock:
        if _broker is not None and _broker._impl is not None:
            return True
        _broker = BrokerWrapper()
        return _broker.initialize()


def shutdown_broker():
    global _broker
    with _broker_lock:
        if _broker is not None:
            _broker.shutdown()
            _broker = None


def get_broker() -> BrokerWrapper:
    global _broker
    if _broker is None or _broker._impl is None:
        raise RuntimeError("Broker not initialized. Call init_broker() first.")
    return _broker