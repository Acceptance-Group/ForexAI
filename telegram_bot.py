import threading
import time
import logging
import hashlib
import json
import os
import requests

from config import TELEGRAM_CONFIG

log = logging.getLogger(__name__)

_bot_token = None
_known_chats = set()
_lock = threading.Lock()
_polling_thread = None
_last_signal_hash = None
_last_signal_text = None

# Reference to dashboard's cached signal (injected from dashboard.py)
_cached_signal_ref = None

_CHATS_FILE = "telegram_chats.json"
_rate_limit_until = 0


def _load_chats():
    global _known_chats
    if os.path.exists(_CHATS_FILE):
        try:
            with open(_CHATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _known_chats = set(data)
            log.info(f"Loaded {len(_known_chats)} chats from {_CHATS_FILE}")
        except Exception as e:
            log.warning(f"Could not load {_CHATS_FILE}: {e}")


def _save_chats():
    try:
        with open(_CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(_known_chats), f)
    except Exception as e:
        log.warning(f"Could not save {_CHATS_FILE}: {e}")


def _add_chat(chat_id):
    with _lock:
        if chat_id not in _known_chats:
            _known_chats.add(chat_id)
            _save_chats()
            log.info(f"Telegram: new chat added {chat_id}")


def set_cached_signal_ref(ref):
    global _cached_signal_ref
    _cached_signal_ref = ref


def init_telegram():
    global _bot_token
    _bot_token = TELEGRAM_CONFIG.get("bot_token", "")
    if not _bot_token:
        log.warning("Telegram bot token not configured — skipping Telegram integration")
        return False
    _load_chats()
    t = threading.Thread(target=_polling_loop, daemon=True)
    t.start()
    log.info("Telegram bot polling started")
    return True


def _polling_loop():
    # Start with offset=-1 to skip old updates on restart
    offset = -1
    first_request = True
    while True:
        try:
            if not _bot_token:
                time.sleep(10)
                continue
            url = f"https://api.telegram.org/bot{_bot_token}/getUpdates"
            params = {"limit": 100, "timeout": 5}
            if offset >= 0:
                params["offset"] = offset
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if not data.get("ok"):
                err_desc = data.get("description", "")
                if "Retry after" in err_desc or "retry after" in err_desc:
                    log.warning(f"Telegram rate limited: {err_desc}")
                    time.sleep(10)
                else:
                    log.warning(f"Telegram getUpdates error: {data}")
                    time.sleep(5)
                continue
            updates = data.get("result", [])
            if first_request and offset == -1 and not updates:
                # After offset=-1, switch to normal offset=0 for new updates
                offset = 0
                first_request = False
                continue
            first_request = False
            for update in updates:
                offset = update["update_id"] + 1
                # Process each update in a background thread so polling never blocks
                threading.Thread(target=_process_update, args=(update,), daemon=True).start()
        except Exception as e:
            log.error(f"Telegram polling error: {e}")
            time.sleep(5)


def _process_update(update):
    try:
        # Handle regular messages
        msg = update.get("message")
        if msg:
            chat_id = msg["chat"]["id"]
            _add_chat(chat_id)

            # Handle bot added to group via new_chat_members
            new_members = msg.get("new_chat_members", [])
            for m in new_members:
                if m.get("is_bot") and m.get("username"):
                    _add_chat(chat_id)
                    log.info(f"Bot added to group {chat_id}")

            text = msg.get("text", "")
            if text and text.startswith("/"):
                _handle_command(chat_id, text, msg)
            return

        # Handle my_chat_member (bot added to group / DM started)
        my_chat = update.get("my_chat_member")
        if my_chat:
            chat = my_chat["chat"]
            chat_id = chat["id"]
            new_status = my_chat.get("new_chat_member", {}).get("status", "")
            if new_status in ("member", "administrator"):
                _add_chat(chat_id)
                log.info(f"Bot became member of chat {chat_id} ({chat.get('type', '?')})")
                if chat.get("type") == "private":
                    _send_message(chat_id,
                        "🤖 *ForexAI* бот активирован!\n\n"
                        "Я буду присылать обновления сигналов и уведомления о сделках.",
                        parse_mode="Markdown"
                    )
            return

        # Handle callback_query
        cb = update.get("callback_query")
        if cb:
            chat_id = cb["message"]["chat"]["id"]
            _add_chat(chat_id)
    except Exception as e:
        log.error(f"Process update error: {e}")


def _handle_command(chat_id, text, msg):
    text = text.strip().lower()
    if text == "/start":
        _send_message(chat_id,
            "🤖 *ForexAI* бот активирован!\n\n"
            "Я буду присылать обновления сигналов и уведомления о сделках.\n\n"
            "*Команды:*\n"
            "`/status` — текущий статус\n"
            "`/signal` — текущий сигнал\n"
            "`/chats` — активных чатов\n"
            "`/help` — справка",
            parse_mode="Markdown"
        )
    elif text == "/status":
        _send_status(chat_id)
    elif text == "/signal":
        _send_signal(chat_id)
    elif text == "/chats":
        with _lock:
            count = len(_known_chats)
        _send_message(chat_id, f"📡 Активных чатов: {count}")
    elif text == "/help":
        _send_message(chat_id,
            "*ForexAI Telegram Bot*\n\n"
            "`/start` — активация\n"
            "`/status` — текущий статус системы\n"
            "`/signal` — текущий прогноз\n"
            "`/chats` — сколько чатов подключено\n"
            "`/help` — эта справка",
            parse_mode="Markdown"
        )


def _send_status(chat_id):
    try:
        # Import here to avoid circular dependency at module load
        from broker import get_broker
        broker = get_broker()
        info = broker.account_info()
        bal = info.balance if info else 0
        eq = info.equity if info else 0
        text = (
            f"📊 *ForexAI Статус*\n\n"
            f"`Баланс   : ${bal:,.2f}`\n"
            f"`Эквити   : ${eq:,.2f}`\n"
            f"`P&L      : ${eq - bal:+.2f}`\n\n"
            f"⏱️ `{_utc_now()}`"
        )
    except Exception as e:
        text = f"📊 *ForexAI Статус*\n\nБрокер недоступен.\n`{e}`"
    _send_message(chat_id, text)


def _send_signal(chat_id):
    sig = None
    if _cached_signal_ref:
        try:
            sig = _cached_signal_ref()
            if not sig or not isinstance(sig, dict):
                sig = None
        except Exception:
            sig = None
    if not sig:
        text = "📡 *ForexAI Сигнал*\n\nСигнал ещё не рассчитан. Попробуй через минуту."
        _send_message(chat_id, text)
        return

    signal = sig.get("signal", "HOLD")
    prob = sig.get("prob_up", 0.5)
    adx = sig.get("adx", 0)
    vol_high = bool(sig.get("vol_high", False))
    atr = sig.get("atr_pips", 0)
    sl = sig.get("sl_pips", 0)
    tp = sig.get("tp_pips", 0)

    if signal == "BUY":
        emoji = "🟢"
        sig_text = "*BUY*"
    elif signal == "SELL":
        emoji = "🔴"
        sig_text = "*SELL*"
    else:
        emoji = "⚪"
        sig_text = "*HOLD*"

    text = (
        f"📡 *ForexAI Сигнал*\n\n"
        f"{emoji} {sig_text}\n"
        f"`P(UP) = {prob:.4f}`\n"
        f"`ADX   = {adx:.1f}`\n"
        f"`Vol   = {'✅' if vol_high else '❌'}`\n"
        f"`ATR   = {atr:.1f}p`\n"
        f"`SL/TP = {sl:.1f}p / {tp:.1f}p`\n\n"
        f"⏱️ `{sig.get('timestamp', '')}`"
    )
    _send_message(chat_id, text)


def _utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _send_message(chat_id, text, parse_mode="Markdown"):
    global _rate_limit_until
    if not _bot_token:
        return False
    if time.time() < _rate_limit_until:
        log.warning(f"Rate limited, skipping message to {chat_id}")
        return False
    try:
        url = f"https://api.telegram.org/bot{_bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=10)
        result = resp.json()
        if not result.get("ok"):
            err_code = result.get("error_code", 0)
            err_desc = result.get("description", "")
            if err_code == 429:
                retry_after = result.get("parameters", {}).get("retry_after", 30)
                _rate_limit_until = time.time() + retry_after
                log.warning(f"Telegram rate limit: retry after {retry_after}s")
            else:
                log.warning(f"Telegram send failed ({err_code}): {err_desc[:200]}")
            # Fallback: try without parse_mode if markdown parsing failed
            if "parse" in err_desc.lower() or "entities" in err_desc.lower():
                try:
                    payload.pop("parse_mode")
                    resp2 = requests.post(url, json=payload, timeout=10)
                    return resp2.json().get("ok", False)
                except Exception:
                    pass
            return False
        return True
    except Exception as e:
        log.error(f"Telegram send error: {e}")
        return False


def _broadcast(text, parse_mode="Markdown"):
    with _lock:
        chats = list(_known_chats)
    if not chats:
        log.debug("No Telegram chats to broadcast to")
        return
    # Run broadcast in background thread so caller is never blocked
    threading.Thread(target=_do_broadcast, args=(chats, text, parse_mode), daemon=True).start()


def _do_broadcast(chats, text, parse_mode):
    for chat_id in chats:
        try:
            _send_message(chat_id, text, parse_mode)
        except Exception as e:
            log.error(f"Broadcast to {chat_id} failed: {e}")
        time.sleep(0.1)  # gentle rate limiting between chats


def _header():
    return (
        "```\n"
        "🤖  F O R E X A I  🤖\n"
        "```"
    )


def _hash_sig(sig):
    s = sig.get("signal", "HOLD")
    p = f"{sig.get('prob_up', 0):.4f}"
    a = f"{sig.get('adx', 0):.1f}"
    v = "1" if sig.get("vol_high") else "0"
    raw = f"{s}|{p}|{a}|{v}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def send_thinking_update(sig):
    global _last_signal_hash, _last_signal_text
    h = _hash_sig(sig)
    if _last_signal_hash == h:
        return
    _last_signal_hash = h

    signal = sig.get("signal", "HOLD")
    prob = sig.get("prob_up", 0.5)
    pxgb = sig.get("prob_xgb", 0)
    plgbm = sig.get("prob_lgbm", 0)
    pcb = sig.get("prob_cb", 0)
    adx = sig.get("adx", 0)
    vol = sig.get("vol_pred", 0)
    vol_high = bool(sig.get("vol_high", False))
    vh = "✅ Прошел" if vol_high else "❌ Отфильтрован"
    atr = sig.get("atr_pips", 0)
    sl = sig.get("sl_pips", 0)
    tp = sig.get("tp_pips", 0)
    dyn = sig.get("dyn_tp_sl", 3.5)

    if signal == "BUY":
        emoji = "🟢"
        sig_text = "*BUY* (покупка)"
    elif signal == "SELL":
        emoji = "🔴"
        sig_text = "*SELL* (продажа)"
    else:
        emoji = "⚪"
        sig_text = "*HOLD* (вне рынка)"

    text = (
        f"{_header()}\n"
        f"{emoji} *Обновление размышлений*\n\n"
        f"*Сигнал:* {sig_text}\n"
        f"`P(UP) = {prob:.4f}`\n\n"
        f"*Вероятности моделей:*\n"
        f"`XGB      : {pxgb:.4f}`\n"
        f"`LightGBM : {plgbm:.4f}`\n"
        f"`CatBoost : {pcb:.4f}`\n\n"
        f"*Фильтры:*\n"
        f"`ADX      : {adx:.1f} / 30.0`\n"
        f"`Vol      : {vh} ({vol:.5f})`\n"
        f"`ATR      : {atr:.1f} pips`\n\n"
        f"*Риск-менеджмент:*\n"
        f"`SL       : {sl:.1f} pips`\n"
        f"`TP       : {tp:.1f} pips`\n"
        f"`TP/SL    : {dyn:.2f}x`\n\n"
        f"⏱️ `{sig.get('timestamp', '')}`"
    )
    _last_signal_text = text
    _broadcast(text)


def send_trade_open(sig, order_info):
    side = sig.get("signal", "HOLD")
    if side == "BUY":
        emoji = "🟢🚀"
        side_text = "*BUY* (покупка)"
    elif side == "SELL":
        emoji = "🔴🚀"
        side_text = "*SELL* (продажа)"
    else:
        emoji = "⚪"
        side_text = "*HOLD*"

    entry = order_info.get("entry_price", 0)
    sl = order_info.get("sl", 0)
    tp = order_info.get("tp", 0)
    lots = order_info.get("volume", 0)
    ticket = order_info.get("ticket", 0)

    text = (
        f"{_header()}\n"
        f"{emoji} *ОТКРЫТИЕ ПОЗИЦИИ*\n\n"
        f"*Направление:* {side_text}\n"
        f"`Тикет    : #{ticket}`\n"
        f"`Объем    : {lots:.2f} лот`\n"
        f"`Вход     : {entry:.5f}`\n"
        f"`SL       : {sl:.5f}`\n"
        f"`TP       : {tp:.5f}`\n\n"
        f"⏱️ `{sig.get('timestamp', '')}`"
    )
    _broadcast(text)


def send_trade_close(trade_data):
    side = trade_data.get("side", "HOLD")
    pnl = trade_data.get("pnl", 0)
    reason = trade_data.get("reason", "N/A")
    ticket = trade_data.get("ticket", 0)
    duration = trade_data.get("duration_hours", 0)

    if pnl > 0:
        emoji = "✅"
        pnl_text = f"*+${pnl:.2f}* 📈"
    elif pnl < 0:
        emoji = "❌"
        pnl_text = f"*-${abs(pnl):.2f}* 📉"
    else:
        emoji = "➖"
        pnl_text = f"`${pnl:.2f}`"

    if side == "BUY":
        side_text = "*BUY*"
    elif side == "SELL":
        side_text = "*SELL*"
    else:
        side_text = "*HOLD*"

    text = (
        f"{_header()}\n"
        f"{emoji} *ЗАКРЫТИЕ ПОЗИЦИИ*\n\n"
        f"*Направление:* {side_text}\n"
        f"`Тикет    : #{ticket}`\n"
        f"`P&L      : {pnl_text}`\n"
        f"`Причина  : {reason}`\n"
        f"`Удержание: {duration:.1f}ч`\n\n"
        f"⏱️ `{trade_data.get('timestamp', '')}`"
    )
    _broadcast(text)
