import threading
import time
import datetime
import logging
import json
import re
import numpy as np

import requests

from config import CTRADER_CONFIG

try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOATraderReq,
        ProtoOAGetTrendbarsReq,
        ProtoOASymbolsListReq,
        ProtoOASymbolByIdReq,
        ProtoOASubscribeSpotsReq,
        ProtoOAUnsubscribeSpotsReq,
        ProtoOAReconcileReq,
        ProtoOANewOrderReq,
        ProtoOAClosePositionReq,
        ProtoOAAmendPositionSLTPReq,
        ProtoOACancelOrderReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOATrendbarPeriod,
        ProtoOAOrderType,
        ProtoOATradeSide,
    )
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
        ProtoHeartbeatEvent,
    )
    HAS_CTRADER = True
except ImportError:
    HAS_CTRADER = False

from twisted.internet import reactor

log = logging.getLogger(__name__)

HEARTBEAT_PT = ProtoHeartbeatEvent().payloadType if HAS_CTRADER else 53


class CTraderClient:
    def __init__(self, host=None, port=None, client_id=None,
                 client_secret=None, access_token=None, account_id=None):
        if not HAS_CTRADER:
            raise ImportError("ctrader-open-api not installed")
        cfg = CTRADER_CONFIG
        self.host = host or cfg.get("host", EndPoints.PROTOBUF_DEMO_HOST)
        self.port = port or cfg.get("port", EndPoints.PROTOBUF_PORT)
        self.client_id = client_id or cfg.get("client_id", "")
        self.client_secret = client_secret or cfg.get("client_secret", "")
        self.access_token = access_token or cfg.get("access_token", "")
        self.account_id = int(account_id or cfg.get("account_id", 0))
        self._client = None
        self._connected = threading.Event()
        self._authed = False
        self._lock = threading.RLock()
        self._responses = {}
        self._events = {}
        self._symbols = {}
        self._timeout = 15
        self._account_info_cache = None
        self._subscribed_symbols = set()
        self._last_spots = {}

    def initialize(self) -> bool:
        with self._lock:
            try:
                return self._connect()
            except Exception as e:
                log.error(f"CTrader init error: {e}")
                import traceback
                traceback.print_exc()
                return False

    def _send_and_wait(self, request, response_name, timeout=None):
        ev = threading.Event()
        self._events[response_name] = ev
        self._events.setdefault("ProtoOAErrorRes_backup", threading.Event())
        self._events["ProtoOAErrorRes"] = ev
        if reactor.running:
            reactor.callFromThread(self._client.send, request)
        else:
            self._client.send(request)
        ev.wait(timeout=timeout or self._timeout)
        self._events.pop(response_name, None)
        self._events.pop("ProtoOAErrorRes", None)
        return self._responses.get(response_name)

    def _refresh_access_token(self) -> bool:
        try:
            cfg = CTRADER_CONFIG
            refresh_token = cfg.get("refresh_token", "")
            if not refresh_token:
                log.error("No refresh_token in config — cannot refresh")
                return False

            token_url = "https://demo.ctraderapi.com/token" if "demo" in self.host else "https://live.ctraderapi.com/token"
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
            log.info("Refreshing access token...")
            resp = requests.post(token_url, data=data, timeout=15)
            if resp.status_code != 200:
                log.error(f"Token refresh failed: HTTP {resp.status_code} {resp.text[:200]}")
                return False

            token_data = resp.json()
            new_access = token_data.get("access_token")
            new_refresh = token_data.get("refresh_token", refresh_token)
            if not new_access:
                log.error(f"Token refresh response missing access_token: {token_data}")
                return False

            self.access_token = new_access
            CTRADER_CONFIG["access_token"] = new_access
            if new_refresh:
                CTRADER_CONFIG["refresh_token"] = new_refresh

            self._update_config_file(new_access, new_refresh)
            log.info("Access token refreshed successfully")
            return True
        except Exception as e:
            log.error(f"Token refresh exception: {e}")
            return False

    def _update_config_file(self, access_token, refresh_token):
        try:
            import os
            config_path = os.path.join(os.path.dirname(__file__), "config.py")
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()

            content = re.sub(
                r'("access_token":\s*")[^"]*("\s*,?)',
                rf'\g<1>{access_token}\g<2>',
                content,
            )
            if refresh_token:
                content = re.sub(
                    r'("refresh_token":\s*")[^"]*("\s*,?)',
                    rf'\g<1>{refresh_token}\g<2>',
                    content,
                )
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(content)
            log.info("config.py tokens updated")
        except Exception as e:
            log.warning(f"Could not update config.py: {e}")

    def _connect(self) -> bool:
        host = self.host
        if host in ("demo.ctraderapi.com", ""):
            host = EndPoints.PROTOBUF_DEMO_HOST
        elif host in ("live.ctraderapi.com",):
            host = EndPoints.PROTOBUF_LIVE_HOST

        if not reactor.running:
            reactor_thread = threading.Thread(target=reactor.run, args=(False,), daemon=True)
            reactor_thread.start()
            for _ in range(30):
                if reactor.running:
                    break
                time.sleep(0.1)
            else:
                log.error("Twisted reactor failed to start")
                return False
            time.sleep(0.5)
            log.info("Twisted reactor started in daemon thread")

        log.info(f"Connecting to cTrader: {host}:{self.port}")
        self._client = Client(host, self.port, TcpProtocol)

        self._client.setConnectedCallback(self._on_connect)
        self._client.setDisconnectedCallback(self._on_disconnect)
        self._client.setMessageReceivedCallback(self._on_message)

        reactor.callFromThread(self._client.startService)

        log.info("Waiting for TCP connection...")
        if not self._connected.wait(timeout=30):
            log.error("TCP connection timeout after 30s")
            log.error(f"  host={host} port={self.port}")
            log.error(f"  reactor.running={reactor.running}")
            return False

        log.info("TCP connected, authenticating app...")
        app_auth = ProtoOAApplicationAuthReq()
        app_auth.clientId = self.client_id
        app_auth.clientSecret = self.client_secret
        app_res = self._send_and_wait(app_auth, "ProtoOAApplicationAuthRes", timeout=15)
        if app_res is None:
            log.error("App auth failed — no response")
            log.error("Check: client_id, client_secret are correct in CTRADER_CONFIG")
            return False
        log.info("App auth OK")

        self.account_id = self._resolve_account_id()

        log.info(f"Account auth with ctidTraderAccountId={self.account_id}...")
        acc_auth = ProtoOAAccountAuthReq()
        acc_auth.ctidTraderAccountId = self.account_id
        acc_auth.accessToken = self.access_token
        acc_res = self._send_and_wait(acc_auth, "ProtoOAAccountAuthRes", timeout=15)
        if acc_res is None:
            err = self._responses.get("ProtoOAErrorRes")
            err_code = getattr(err, 'errorCode', '') if err else ''
            err_desc = getattr(err, 'description', '') if err else ''
            if err:
                log.error(f"Account auth error: {err_code} {err_desc}")
            else:
                log.error("Account auth failed — no response. Check access_token and account_id.")

            if 'UNAUTHORIZED' in str(err_code).upper() or 'NOT_AUTHORIZED' in str(err_desc).upper() or 'INVALID_REQUEST' in str(err_code).upper():
                log.info("Token appears expired — attempting refresh...")
                if self._refresh_access_token():
                    log.info("Retrying account auth with new token...")
                    acc_auth.accessToken = self.access_token
                    acc_res2 = self._send_and_wait(acc_auth, "ProtoOAAccountAuthRes", timeout=15)
                    if acc_res2 is not None:
                        log.info("Account auth OK after refresh")
                        self._authed = True
                        self._load_symbols()
                        self._load_account_info()
                        self._subscribe_main_symbols()
                        log.info(f"cTrader ready: account={self.account_id}, symbols={len(self._symbols)}")
                        return True
                log.error("Token refresh did not resolve auth failure")
            return False
        log.info("Account auth OK")
        self._authed = True

        self._load_symbols()
        self._load_account_info()
        self._subscribe_main_symbols()

        log.info(f"cTrader ready: account={self.account_id}, symbols={len(self._symbols)}")
        return True

    def _resolve_account_id(self):
        acct_list_req = ProtoOAGetAccountListByAccessTokenReq()
        acct_list_req.accessToken = self.access_token
        log.info("Discovering accounts from access token...")
        result = self._send_and_wait(acct_list_req, "ProtoOAGetAccountListByAccessTokenRes", timeout=10)
        if result is None:
            log.warning("Could not discover accounts, using config account_id as-is")
            return self.account_id

        configured = self.account_id
        for acc in result.ctidTraderAccount:
            log.info(f"  Found account: ctid={acc.ctidTraderAccountId} login={getattr(acc, 'traderLogin', '?')} live={acc.isLive}")
            if acc.ctidTraderAccountId == configured:
                return configured
            if getattr(acc, 'traderLogin', 0) == configured:
                resolved = acc.ctidTraderAccountId
                log.info(f"  Resolved traderLogin {configured} -> ctidTraderAccountId {resolved}")
                return resolved

        if result.ctidTraderAccount:
            first = result.ctidTraderAccount[0]
            resolved = first.ctidTraderAccountId
            log.info(f"  Using first account: ctid={resolved} login={getattr(first, 'traderLogin', '?')}")
            return resolved

        return configured

    def _on_connect(self, client):
        log.info(f"cTrader TCP connected to {self.host}:{self.port}")
        self._connected.set()

    def _on_disconnect(self, client, reason):
        self._connected.clear()
        log.warning(f"cTrader disconnected: {reason}")

    def _on_message(self, client, message):
        pt = message.payloadType
        if pt == HEARTBEAT_PT:
            return
        try:
            extracted = Protobuf.extract(message)
            name = extracted.__class__.__name__
        except Exception:
            name = f"pt_{pt}"
            extracted = message

        self._responses[name] = extracted

        if name == "ProtoOAErrorRes":
            code = getattr(extracted, 'errorCode', '')
            if code == 'ALREADY_SUBSCRIBED':
                pass
            else:
                log.warning(f"cTrader error: code={code} desc={getattr(extracted, 'description', '?')}")

        if name == "ProtoOASpotEvent":
            sym_id = getattr(extracted, 'symbolId', 0)
            if sym_id:
                for sname, info in self._symbols.items():
                    if info["id"] == sym_id:
                        digits = info.get("digits", 5)
                        bid = self._price_from_raw(getattr(extracted, 'bid', 0), digits)
                        ask = self._price_from_raw(getattr(extracted, 'ask', 0), digits)
                        ts = datetime.datetime.fromtimestamp(getattr(extracted, 'timestamp', int(time.time() * 1000)) / 1000)
                        self._last_spots[sym_id] = (bid, ask, time.time(), ts)
                        break

        ev = self._events.get(name)
        if ev:
            ev.set()

    _KEY_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "EUR_USD", "GBP_USD", "USD_JPY"]

    def _load_symbols(self):
        # Step 1: Get light symbols list
        
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.account_id
        req.includeArchivedSymbols = False
        result = self._send_and_wait(req, "ProtoOASymbolsListRes", timeout=15)
        if result is None:
            log.error("Failed to load symbols list")
            return

        light_map = {}
        id_to_name = {}
        for sym in result.symbol:
            self._symbols[sym.symbolName] = {
                "id": sym.symbolId,
                "name": sym.symbolName,
                "digits": 5,
                "pip_size": 0.0001,
                "contract_size": 100000,
                "_loaded": False,
            }
            light_map[sym.symbolName] = sym.symbolId
            id_to_name[sym.symbolId] = sym.symbolName
        log.info(f"Loaded {len(self._symbols)} light symbols")

        
        for key in self._KEY_SYMBOLS:
            if key not in light_map:
                continue
            sym_id = light_map[key]
            detail_req = ProtoOASymbolByIdReq()
            detail_req.ctidTraderAccountId = self.account_id
            detail_req.symbolId.append(sym_id)
            detail = self._send_and_wait(detail_req, "ProtoOASymbolByIdRes", timeout=10)
            if detail and hasattr(detail, 'symbol'):
                for s in detail.symbol:
                    sname = id_to_name.get(s.symbolId, key)
                    self._symbols[sname] = {
                        "id": s.symbolId,
                        "name": sname,
                        "digits": s.digits,
                        "pip_size": 10 ** (-s.pipPosition) if hasattr(s, 'pipPosition') else 0.0001,
                        "contract_size": getattr(s, 'lotSize', 100000),
                        "_loaded": True,
                    }
                    log.info(f"  {sname}: id={s.symbolId} digits={s.digits} lotSize={getattr(s, 'lotSize', '?')}")
            else:
                log.warning(f"  Failed to load full details for {key}")

    def _load_account_info(self):
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        result = self._send_and_wait(req, "ProtoOATraderRes", timeout=10)
        if result and hasattr(result, 'trader'):
            t = result.trader
            bal = float(t.balance) / 100.0
            lev = getattr(t, 'leverage', 100)
            cur = getattr(t, 'currencyId', 'N/A')
            log.info(f"Account: balance=${bal:,.2f} leverage=1:{lev} currency={cur}")
            self._account_info_cache = result
        else:
            log.error("Failed to load account info")

    def _subscribe_main_symbols(self):
        sym_ids = set()
        for name in ["EURUSD", "GBPUSD", "USDJPY"]:
            if name in self._symbols:
                sym_ids.add(self._symbols[name]["id"])
        if sym_ids:
            try:
                req = ProtoOASubscribeSpotsReq()
                req.ctidTraderAccountId = self.account_id
                for sid in sym_ids:
                    req.symbolId.append(sid)
                reactor.callFromThread(self._client.send, req)
                self._subscribed_symbols.update(sym_ids)
                log.info(f"Subscribed to spots for {len(sym_ids)} main symbols")
            except Exception as e:
                log.debug(f"Subscribe main symbols error: {e}")

    def shutdown(self):
        with self._lock:
            if self._client:
                try:
                    if reactor.running:
                        reactor.callFromThread(self._client.stopService)
                    else:
                        self._client.stopService()
                except Exception:
                    pass
            self._connected.clear()
            self._authed = False

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and self._authed

    def _get_symbol_id(self, symbol_name: str) -> int:
        clean = symbol_name.replace("/", "").replace(" ", "").upper()
        if clean in self._symbols:
            return self._symbols[clean]["id"]
        
        log.warning(f"Symbol '{clean}' not in cache, available: {list(self._symbols.keys())[:10]}")
        return 1

    def _tf_to_period(self, timeframe):
        mapping = {
            16408: ProtoOATrendbarPeriod.D1,
            16420: ProtoOATrendbarPeriod.H4,
            16388: ProtoOATrendbarPeriod.H1,
            16385: ProtoOATrendbarPeriod.M15,
            16386: ProtoOATrendbarPeriod.M5,
            16384: ProtoOATrendbarPeriod.M1,
        }
        return mapping.get(timeframe, ProtoOATrendbarPeriod.D1)

    def _seconds_for_tf(self, timeframe):
        return {16408: 86400, 16420: 14400, 16388: 3600, 16385: 900, 16386: 300, 16384: 60}.get(timeframe, 86400)

    def _get_symbol_digits(self, symbol_name: str) -> int:
        clean = symbol_name.replace("/", "").replace(" ", "").upper()
        if clean in self._symbols and self._symbols[clean].get("_loaded"):
            return self._symbols[clean]["digits"]
        return 5

    def _price_from_raw(self, raw_val, digits=None):
        if digits is None:
            digits = 5
        divisor = 10 ** digits
        return float(raw_val) / divisor

    def account_info(self):
        with self._lock:
            result = _AccountInfo()
            result.server = f"cTrader ({self.host})"
            try:
                data = self._account_info_cache
                if data is None or not hasattr(data, 'trader'):
                    req = ProtoOATraderReq()
                    req.ctidTraderAccountId = self.account_id
                    data = self._send_and_wait(req, "ProtoOATraderRes", timeout=10)
                    if data:
                        self._account_info_cache = data
                if data and hasattr(data, 'trader'):
                    t = data.trader
                    money_digits = int(getattr(t, 'moneyDigits', 2))
                    divisor = 10 ** money_digits
                    result.balance = float(t.balance) / divisor
                    result.equity = float(t.balance) / divisor
                    result.profit = 0.0
                    result.margin = float(getattr(t, 'usedMargin', 0)) / divisor
                    result.margin_free = result.equity - result.margin
                    result.leverage = int(getattr(t, 'leverageInCents', 10000)) // 100
                    result.server = f"cTrader ({self.host})"
            except Exception as e:
                log.debug(f"account_info error: {e}")
            return result

    def copy_rates_from_pos(self, symbol_name, timeframe, start_pos, count):
        """Fetch OHLCV bars. start_pos=0 is most recent, increasing goes back."""
        with self._lock:
            try:
                sym_id = self._get_symbol_id(symbol_name)
                period = self._tf_to_period(timeframe)
                sec_per_bar = self._seconds_for_tf(timeframe)
                digits = self._get_symbol_digits(symbol_name)

                
                end_ts = int(time.time() * 1000) - (start_pos * sec_per_bar * 1000)
                from_ts = max(0, end_ts - (count * sec_per_bar * 1000))

                
                total_needed = count
                all_bars = []
                batch_start = from_ts
                batch_end = end_ts

                while total_needed > 0:
                    batch = min(total_needed, 5000)

                    req = ProtoOAGetTrendbarsReq()
                    req.ctidTraderAccountId = self.account_id
                    req.symbolId = sym_id
                    req.period = period
                    req.fromTimestamp = int(batch_start)
                    req.toTimestamp = int(batch_end)

                    result = self._send_and_wait(req, "ProtoOAGetTrendbarsRes", timeout=15)
                    if result is None or not hasattr(result, 'trendbar') or len(result.trendbar) == 0:
                        if not all_bars:
                            log.warning(f"No bars returned for {symbol_name} TF={timeframe}")
                        break

                    bars = list(result.trendbar)
                    all_bars.extend(bars)
                    total_needed -= batch

                    if len(bars) < batch:
                        break

                    earliest = min(b.utcTimestampInMinutes for b in bars) * 60 * 1000
                    batch_end = earliest - 1
                    batch_start = max(1, batch_end - (total_needed * sec_per_bar * 1000))

                if not all_bars:
                    log.warning(f"No bars returned for {symbol_name} TF={timeframe}")
                    return None

                all_bars.sort(key=lambda b: b.utcTimestampInMinutes)

                
                seen = set()
                unique_bars = []
                for b in all_bars:
                    key = b.utcTimestampInMinutes
                    if key not in seen:
                        seen.add(key)
                        unique_bars.append(b)
                all_bars = unique_bars

                
                if len(all_bars) > count:
                    all_bars = all_bars[-count:]

                dtype = np.dtype([
                    ('time', 'i8'), ('open', 'f8'), ('high', 'f8'),
                    ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
                    ('spread', 'i4'), ('real_volume', 'i8'),
                ])
                arr = np.empty(len(all_bars), dtype=dtype)

                for i, b in enumerate(all_bars):
                    low_raw = b.low
                    open_raw = low_raw + b.deltaOpen
                    high_raw = low_raw + b.deltaHigh
                    close_raw = low_raw + b.deltaClose
                    ts_seconds = int(b.utcTimestampInMinutes) * 60
                    arr[i] = (
                        ts_seconds,
                        self._price_from_raw(open_raw, digits),
                        self._price_from_raw(high_raw, digits),
                        self._price_from_raw(low_raw, digits),
                        self._price_from_raw(close_raw, digits),
                        int(getattr(b, 'volume', 0)),
                        0, 0,
                    )

                
                for i in range(len(arr)):
                    if arr[i]['open'] == 0:
                        arr[i]['open'] = arr[i]['low']
                    if arr[i]['high'] == 0:
                        arr[i]['high'] = arr[i]['close']
                    if arr[i]['low'] == 0:
                        arr[i]['low'] = arr[i]['close']
                    if arr[i]['close'] == 0:
                        arr[i]['close'] = arr[i]['open']

                return arr

            except Exception as e:
                log.error(f"copy_rates error: {e}")
                import traceback
                traceback.print_exc()
                return None

    def symbol_info(self, symbol_name):
        clean = symbol_name.replace("/", "").replace(" ", "").upper()
        return self._symbols.get(clean)

    def symbol_info_tick(self, symbol_name):
        with self._lock:
            try:
                sym_id = self._get_symbol_id(symbol_name)
                digits = self._get_symbol_digits(symbol_name)

                self._subscribe_symbols({sym_id})

                cached = self._last_spots.get(sym_id)
                if cached and (time.time() - cached[2]) < 30:
                    tick = _TickInfo()
                    tick.bid = cached[0]
                    tick.ask = cached[1]
                    tick.time = cached[3]
                    return tick

                ev = threading.Event()
                self._events["ProtoOASpotEvent"] = ev
                ev.wait(timeout=5)
                data = self._responses.get("ProtoOASpotEvent")
                if data and getattr(data, 'symbolId', 0) == sym_id:
                    tick = _TickInfo()
                    tick.bid = self._price_from_raw(getattr(data, 'bid', 0), digits)
                    tick.ask = self._price_from_raw(getattr(data, 'ask', 0), digits)
                    tick.time = datetime.datetime.fromtimestamp(
                        getattr(data, 'timestamp', int(time.time() * 1000)) / 1000
                    )
                    self._last_spots[sym_id] = (tick.bid, tick.ask, time.time(), tick.time)
                    return tick

                if cached:
                    tick = _TickInfo()
                    tick.bid = cached[0]
                    tick.ask = cached[1]
                    tick.time = cached[3]
                    return tick
            except Exception as e:
                log.debug(f"symbol_info_tick error: {e}")
            return None

    def _subscribe_symbols(self, sym_ids):
        """Subscribe to spot prices for given symbol IDs. Skip if already subscribed."""
        new_ids = sym_ids - self._subscribed_symbols
        if not new_ids:
            return
        try:
            req = ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = self.account_id
            for sid in new_ids:
                req.symbolId.append(sid)
            reactor.callFromThread(self._client.send, req)
            self._subscribed_symbols.update(new_ids)
            log.debug(f"Subscribed to spots: {new_ids}")
        except Exception as e:
            log.debug(f"Subscribe error: {e}")

    def positions_get(self, symbol=None):
        with self._lock:
            try:
                req = ProtoOAReconcileReq()
                req.ctidTraderAccountId = self.account_id

                result = self._send_and_wait(req, "ProtoOAReconcileRes", timeout=self._timeout)
                if result is None or not hasattr(result, 'position'):
                    return None

                positions = []
                for p in result.position:
                    pos = _PositionInfo()
                    pos.ticket = int(p.positionId)
                    pos.type = 0 if p.tradeData.tradeSide == ProtoOATradeSide.BUY else 1
                    pos.volume = float(p.tradeData.volume) / 100.0
                    money_digits = int(getattr(p, 'moneyDigits', 2))
                    money_divisor = 10 ** money_digits
                    pos.price_open = float(p.price) / money_divisor
                    pos.price_current = float(p.price) / money_divisor
                    pos.sl = float(p.stopLoss) / money_divisor if getattr(p, 'stopLoss', 0) else 0.0
                    pos.tp = float(p.takeProfit) / money_divisor if getattr(p, 'takeProfit', 0) else 0.0
                    pos.profit = 0.0
                    pos.time = datetime.datetime.utcfromtimestamp(
                        int(p.tradeData.openTimestamp) / 1000
                    ) if hasattr(p.tradeData, 'openTimestamp') and p.tradeData.openTimestamp else datetime.datetime.utcnow()

                    
                    sym_name = ""
                    sym_digits = 5
                    for name, info in self._symbols.items():
                        if info["id"] == p.tradeData.symbolId:
                            sym_name = name
                            sym_digits = info.get("digits", 5)
                            break
                    pos.symbol = sym_name

                    if symbol:
                        clean = symbol.replace("/", "").replace(" ", "").upper()
                        if clean == sym_name.upper().replace("/", ""):
                            positions.append(pos)
                    else:
                        positions.append(pos)

                
                if positions:
                    self._update_position_prices(positions)

                return positions if positions else None
            except Exception as e:
                log.error(f"positions_get error: {e}")
                return None

    def _update_position_prices(self, positions):
        """Update price_current and profit for positions using cached spot data."""
        try:
            sym_ids = set()
            for pos in positions:
                if pos.symbol and pos.symbol in self._symbols:
                    sym_ids.add(self._symbols[pos.symbol]["id"])

            if not sym_ids:
                return

            self._subscribe_symbols(sym_ids)

            time.sleep(0.5)

            for pos in positions:
                if pos.symbol and pos.symbol in self._symbols:
                    sym_id = self._symbols[pos.symbol]["id"]
                    cached = self._last_spots.get(sym_id)
                    if cached:
                        bid, ask, _, _ = cached
                        mid = (bid + ask) / 2.0
                        pos.price_current = mid
                        if pos.type == 0:
                            pos.profit = (bid - pos.price_open) * pos.volume * 100000
                        else:
                            pos.profit = (pos.price_open - ask) * pos.volume * 100000
        except Exception as e:
            log.debug(f"update_position_prices error: {e}")

    def close_position(self, position_ticket, symbol=None):
        """Close a position by its ticket (positionId in cTrader)."""
        with self._lock:
            try:
                req = ProtoOAClosePositionReq()
                req.ctidTraderAccountId = self.account_id
                req.positionId = int(position_ticket)
                req.volume = 0  

                ev = threading.Event()
                self._events["ProtoOAExecutionEvent"] = ev
                self._events["ProtoOAOrderErrorEvent"] = ev
                reactor.callFromThread(self._client.send, req)
                ev.wait(timeout=self._timeout)

                result = _OrderResult()
                data = self._responses.get("ProtoOAExecutionEvent")
                if data:
                    result.retcode = 1
                    result.order = int(getattr(data, 'orderId', getattr(data, 'order', 0)))
                    result.comment = "Closed"
                else:
                    err = self._responses.get("ProtoOAOrderErrorEvent")
                    if err:
                        result.retcode = 0
                        result.comment = f"Error: {getattr(err, 'errorCode', '?')} {getattr(err, 'description', '?')}"
                    else:
                        result.retcode = 0
                        result.comment = "Timeout"
                return result
            except Exception as e:
                result = _OrderResult()
                result.retcode = 0
                result.comment = str(e)
                return result

    def modify_position(self, position_ticket, sl=None, tp=None):
        with self._lock:
            try:
                req = ProtoOAAmendPositionSLTPReq()
                req.ctidTraderAccountId = self.account_id
                req.positionId = int(position_ticket)

                if sl is not None:
                    req.stopLoss = float(sl)
                if tp is not None:
                    req.takeProfit = float(tp)

                ev = threading.Event()
                self._events["ProtoOAExecutionEvent"] = ev
                self._events["ProtoOAOrderErrorEvent"] = ev
                reactor.callFromThread(self._client.send, req)
                ev.wait(timeout=self._timeout)

                result = _OrderResult()
                data = self._responses.get("ProtoOAExecutionEvent")
                if data:
                    result.retcode = 1
                    result.order = int(position_ticket)
                    result.comment = "Modified"
                else:
                    err = self._responses.get("ProtoOAOrderErrorEvent")
                    if err:
                        result.retcode = 0
                        result.comment = f"Error: {getattr(err, 'errorCode', '?')} {getattr(err, 'description', '?')}"
                    else:
                        result.retcode = 0
                        result.comment = "Timeout"
                return result
            except Exception as e:
                result = _OrderResult()
                result.retcode = 0
                result.comment = str(e)
                return result

    def order_send(self, request):
        with self._lock:
            try:
                sym_id = self._get_symbol_id(request.get("symbol", "EURUSD"))
                trade_side = ProtoOATradeSide.BUY if request.get("type", 0) == 0 else ProtoOATradeSide.SELL

                req = ProtoOANewOrderReq()
                req.ctidTraderAccountId = self.account_id
                req.symbolId = sym_id
                req.orderType = ProtoOAOrderType.MARKET
                req.tradeSide = trade_side
                req.volume = int(request.get("volume", 0.01) * 100)

                if request.get("sl", 0):
                    req.stopLoss = float(request["sl"])
                if request.get("tp", 0):
                    req.takeProfit = float(request["tp"])

                ev = threading.Event()
                self._events["ProtoOAExecutionEvent"] = ev
                self._events["ProtoOAOrderErrorEvent"] = ev
                reactor.callFromThread(self._client.send, req)

                ev.wait(timeout=self._timeout)

                result = _OrderResult()
                data = self._responses.get("ProtoOAExecutionEvent")
                if data:
                    result.retcode = 1
                    result.order = int(getattr(data, 'orderId', getattr(data, 'order', 0)))
                    result.comment = "Done"
                else:
                    err = self._responses.get("ProtoOAOrderErrorEvent")
                    if err:
                        result.retcode = 0
                        result.comment = f"Error: {getattr(err, 'errorCode', '?')} {getattr(err, 'description', '?')}"
                    else:
                        result.retcode = 0
                        result.comment = "Timeout"
                return result
            except Exception as e:
                result = _OrderResult()
                result.retcode = 0
                result.comment = str(e)
                return result

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TIMEFRAME_D1 = 16408
    TIMEFRAME_H4 = 16420
    TIMEFRAME_H1 = 16388
    TIMEFRAME_M15 = 16385
    TRADE_ACTION_DEAL = 1
    TRADE_RETCODE_DONE = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1


class _AccountInfo:
    balance = 0.0
    equity = 0.0
    profit = 0.0
    margin = 0.0
    margin_free = 0.0
    leverage = 100
    server = ""


class _TickInfo:
    bid = 0.0
    ask = 0.0
    time = None


class _PositionInfo:
    ticket = 0
    symbol = ""
    type = 0
    volume = 0.0
    price_open = 0.0
    price_current = 0.0
    sl = 0.0
    tp = 0.0
    profit = 0.0
    time = None
    magic = 0


class _OrderResult:
    retcode = 0
    order = 0
    comment = ""