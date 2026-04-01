"""Telegram message handlers.

handle_message  — plain text → agent → step messages + final answer
handle_document — file upload → read content → agent (same flow)
handle_start    — /start command
"""
from __future__ import annotations

import time
from collections import deque

import structlog
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes

import config
from agent.confirmation import confirmation_manager
from bot.confirmation import make_confirm_callback, resolve_pending
from bot.telegram_transport import send_error, send_final, send_step, send_typing

logger = structlog.get_logger()

# Agent instance injected by bot/main.py
_agent = None

# Track which user IDs are currently being processed to prevent overlapping requests
_currently_processing: set[int] = set()

# Max file size to read as text (500 KB)
_MAX_FILE_READ_BYTES = 500 * 1024

# Rate limiting: user_id → deque of request timestamps (unix seconds)
_request_timestamps: dict[int, deque] = {}


def _check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Return (allowed, seconds_until_next_slot).

    Evicts timestamps older than 1 hour, then checks against MAX_REQUESTS_PER_HOUR.
    Returns (True, 0) if unlimited (MAX_REQUESTS_PER_HOUR == 0).
    """
    if config.MAX_REQUESTS_PER_HOUR == 0:
        return True, 0

    now = time.time()
    window = 3600  # 1 hour in seconds

    timestamps = _request_timestamps.setdefault(user_id, deque())
    # Evict old entries
    while timestamps and now - timestamps[0] > window:
        timestamps.popleft()

    if len(timestamps) >= config.MAX_REQUESTS_PER_HOUR:
        # Next slot opens when the oldest timestamp expires
        wait = int(window - (now - timestamps[0])) + 1
        return False, wait

    timestamps.append(now)
    return True, 0


def set_agent(agent) -> None:
    """Inject the AdminAgent instance used by all handlers."""
    global _agent
    _agent = agent


def is_authorized(user_id: int) -> bool:
    """Return True if the user is allowed to interact with the bot."""
    if not config.ALLOWED_USER_IDS:
        return True  # empty list → allow everyone (dev mode)
    return user_id in config.ALLOWED_USER_IDS


# ── Handlers ────────────────────────────────────────────────────────────────────────────────

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    tools_count = len(_agent.registry.tools) if _agent else 0
    await update.message.reply_text(
        "👋 Привет! Я агент-администратор.\n\n"
        "Расскажи, чем я буду управлять:\n"
        "— скинь схему БД + credentials\n"
        "— или API-документацию + токен\n\n"
        f"Сейчас у меня {tools_count} инструментов.\n\n"
        "Команды:\n"
        "/reset — сбросить всё и начать с нового проекта"
    )


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset: wipe all project data and start fresh."""
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    if _agent is None:
        await update.message.reply_text("Агент не инициализирован.")
        return

    if user_id in _currently_processing:
        await update.message.reply_text("⏳ Дождитесь завершения текущего запроса перед сбросом.")
        return

    await update.message.reply_text("🗑 Сбрасываю проект... подождите.")
    await _agent.reset()
    await update.message.reply_text(
        "✅ Готово. Всё очищено:\n"
        "— история диалога\n"
        "— созданные инструменты\n"
        "— контекст проекта\n"
        "— MCP-серверы\n\n"
        "Можно начинать новый проект."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a plain text message through the agent ReAct loop."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    bot = context.bot

    if not is_authorized(user_id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    if _agent is None:
        await send_error(bot, chat_id, "Агент ещё не инициализирован.")
        return

    if user_id in _currently_processing:
        # Allow text-based confirmation as a fallback when agent is waiting for approval
        msg_lower = (update.message.text or "").strip().lower()
        if msg_lower in {"да", "yes", "+", "ок", "ok", "подтвердить", "confirm"}:
            if resolve_pending(chat_id, approved=True):
                return  # handled — agent will continue
        elif msg_lower in {"нет", "no", "-", "отмена", "cancel", "отменить", "стоп"}:
            if resolve_pending(chat_id, approved=False):
                return  # handled — agent will cancel
        await update.message.reply_text(
            "⏳ Обрабатываю предыдущий запрос, подождите..."
        )
        return

    allowed, wait_sec = _check_rate_limit(user_id)
    if not allowed:
        minutes = wait_sec // 60
        await update.message.reply_text(
            f"⛔ Превышен лимит: {config.MAX_REQUESTS_PER_HOUR} запросов в час.\n"
            f"Следующий запрос доступен через ~{minutes} мин."
        )
        return

    text = update.message.text.strip()
    if not text:
        return

    if len(text) > config.MAX_INPUT_CHARS:
        text = text[:config.MAX_INPUT_CHARS]
        await update.message.reply_text(
            f"⚠️ Сообщение обрезано до {config.MAX_INPUT_CHARS} символов."
        )

    _currently_processing.add(user_id)
    confirmation_manager.set_callback(make_confirm_callback(bot, chat_id))
    try:
        await send_typing(bot, chat_id)

        async def step_callback(step):
            await send_step(bot, chat_id, step)

        final = await _agent.handle_message(text, step_callback)
        await send_final(bot, chat_id, final)

    except Exception as exc:
        logger.error("handler_error", user_id=user_id, error=str(exc), exc_info=True)
        await send_error(bot, chat_id, f"Что-то пошло не так: {exc}")
    finally:
        confirmation_manager.set_callback(None)
        _currently_processing.discard(user_id)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle file uploads: download, read as text, pass to agent."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    bot = context.bot

    if not is_authorized(user_id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    if _agent is None:
        await send_error(bot, chat_id, "Агент ещё не инициализирован.")
        return

    if user_id in _currently_processing:
        await update.message.reply_text("⏳ Обрабатываю предыдущий запрос, подождите...")
        return

    doc = update.message.document
    filename = doc.file_name or "file"
    file_size = doc.file_size or 0

    # Hard limit: don't even download huge files
    if file_size > 20 * 1024 * 1024:
        await send_error(bot, chat_id, f"Файл слишком большой ({file_size // 1024 // 1024} МБ). Максимум — 20 МБ.")
        return

    await send_typing(bot, chat_id)

    # Download to agent_tmp/
    tmp_dir = Path(config.AGENT_TMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / filename

    tg_file = await bot.get_file(doc.file_id)
    await tg_file.download_to_drive(str(dest))
    logger.info("file_downloaded", filename=filename, size=file_size)

    # Try to read as text
    file_content = _read_file_as_text(dest, filename)

    # Compose message for the agent: caption (if any) + file content
    caption = (update.message.caption or "").strip()
    if caption:
        agent_text = f"{caption}\n\n[Файл: {filename}]\n{file_content}"
    else:
        agent_text = f"[Файл: {filename}]\n{file_content}"

    await update.message.reply_text(
        f"📎 Файл получен: {filename}\nПередаю агенту..."
    )

    _currently_processing.add(user_id)
    confirmation_manager.set_callback(make_confirm_callback(bot, chat_id))
    try:
        async def step_callback(step):
            await send_step(bot, chat_id, step)

        final = await _agent.handle_message(agent_text, step_callback)
        await send_final(bot, chat_id, final)

    except Exception as exc:
        logger.error("document_handler_error", user_id=user_id, error=str(exc), exc_info=True)
        await send_error(bot, chat_id, f"Что-то пошло не так: {exc}")
    finally:
        confirmation_manager.set_callback(None)
        _currently_processing.discard(user_id)


# ── Helpers ─────────────────────────────────────────────────────────────────────────────────

def _read_file_as_text(path: Path, filename: str) -> str:
    """Try to read the file as UTF-8 text. Returns truncated content or a notice."""
    try:
        raw = path.read_bytes()
        text = raw[:_MAX_FILE_READ_BYTES].decode("utf-8", errors="replace")
        truncated = len(raw) > _MAX_FILE_READ_BYTES
        if truncated:
            text += f"\n\n... (файл обрезан, показано {_MAX_FILE_READ_BYTES // 1024} КБ из {len(raw) // 1024} КБ)"
        return text
    except Exception as exc:
        return f"(не удалось прочитать файл как текст: {exc})"
