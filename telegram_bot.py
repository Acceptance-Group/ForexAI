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

_CHATS_FILE = "telegram_chats.json"


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
    global _known_chats
    offset = 0
    while True:
        try:
            if not _bot_token:
                time.sleep(10)
                continue
            url = f"https://api.telegram.org/bot{_bot_token}/getUpdates"
            resp = requests.get(url, params={"offset": offset, "limit": 100}, timeout=30)
            data = resp.json()
            if not data.get("ok"):
                time.sleep(5)
                continue
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                _process_update(update)
        except Exception as e:
            log.error(f"Telegram polling error: {e}")
            time.sleep(5)


def _process_update(update):
    # Handle regular messages (incl. group messages where bot is mentioned or /command)
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

    # Handle chat_member updates (if allowed)
    chat_mem = update.get("chat_member")
    if chat_mem:
        chat_id = chat_mem["chat"]["id"]
        new_status = chat_mem.get("new_chat_member", {}).get("status", "")
        if new_status in ("member", "administrator"):
            _add_chat(chat_id)

    # Handle callback_query (buttons)
    cb = update.get("callback_query")
    if cb:
        chat_id = cb["message"]["chat"]["id"]
        _add_chat(chat_id)


def _handle_command(chat_id, text, msg):
    text = text.strip().lower()
    if text == "/start":
        _send_message(chat_id,
            "🤖 *ForexAI* бот активирован!\n\n"
            "Я буду присылать обновления сигналов и уведомления о сделках.\n\n"
            "*Команды:*\n"
            "`/status` — текущий статус\n"
            "`/signal` — текущий сигнал\n"
            "`/chats` — список активных чатов\n"
            "`/help` — справка",
            parse_mode="Markdown"
        )
    elif text == "/status":
        _send_message(chat_id, "📊 Статус запрошен... (скоро будет)")
    elif text == "/signal":
        _send_message(chat_id, "📡 Сигнал запрошен... (скоро будет)")
    elif text == "/chats":
        with _lock:
            count = len(_known_chats)
        _send_message(chat_id, f"📡 Активных чатов: {count}")
    elif text == "/help":
        _send_message(chat_id,
            "*ForexAI Telegram Bot*\n\n"
            "`/start` — активация\n"
            "`/status` — баланс и позиция\n"
            "`/signal` — текущий прогноз\n"
            "`/chats` — сколько чатов подключено\n"
            "`/help` — эта справка",
            parse_mode="Markdown"
        )


def _send_message(chat_id, text, parse_mode="Markdown"):
    if not _bot_token:
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
        if not resp.json().get("ok"):
            log.warning(f"Telegram send failed: {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        log.error(f"Telegram send error: {e}")
        return False


def _broadcast(text, parse_mode="Markdown"):
    with _lock:
        chats = list(_known_chats)
    if not chats:
        chats = TELEGRAM_CONFIG.get("default_chat_ids", [])
    if not chats:
        log.debug("No Telegram chats to broadcast to")
        return
    for chat_id in chats:
        try:
            _send_message(chat_id, text, parse_mode)
            time.sleep(0.05)
        except Exception as e:
            log.error(f"Broadcast to {chat_id} failed: {e}")


def _header():
    return (
        "```\n"
        "╔══════════════════════════════════════════╗\n"
        "║           🤖  F O R E X A I              ║\n"
        "╚══════════════════════════════════════════╝\n"
        "```"
    )


def _hash_sig(sig):
    """Create deterministic hash of signal core values for dedup."""
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
