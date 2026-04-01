"""Telegram confirmation UI.

Provides:
  - make_confirm_callback(bot, chat_id) → async callback for ConfirmationManager
  - handle_confirmation_callback(update, context) → CallbackQueryHandler coroutine

When the agent needs confirmation, it calls confirmation_manager.ask(description).
This sends an InlineKeyboard message to the user and blocks (via asyncio.Event)
until the user presses Confirm or Cancel (or the timeout expires).
"""
from __future__ import annotations

import asyncio
import html

import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

logger = structlog.get_logger()

# Prefix for callback_data so we don't clash with other handlers
_CB_PREFIX = "sec_confirm"
_TIMEOUT_SECONDS = 120

# Pending confirmations: chat_id → {"event": asyncio.Event, "approved": bool}
_pending: dict[int, dict] = {}


def make_confirm_callback(bot: Bot, chat_id: int):
    """Return an async callable suitable for ConfirmationManager.set_callback()."""

    async def _confirm(description: str) -> bool:
        event = asyncio.Event()

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Подтвердить", callback_data=f"{_CB_PREFIX}:{chat_id}:yes"),
            InlineKeyboardButton("❌ Отмена",       callback_data=f"{_CB_PREFIX}:{chat_id}:no"),
        ]])

        # Set _pending AFTER send_message succeeds — avoids leaked entry on send failure.
        # Use HTML parse mode so raw SQL / special chars in description are safely escaped.
        await bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ <b>Требуется подтверждение</b>\n\n{html.escape(description)}",
            reply_markup=keyboard,
            parse_mode="HTML",
        )

        _pending[chat_id] = {"event": event, "approved": False}
        logger.info(
            "confirmation_waiting",
            chat_id=chat_id,
            pending_keys=list(_pending.keys()),
            timeout=_TIMEOUT_SECONDS,
        )
        logger.info("confirmation_message_sent", chat_id=chat_id)
        try:
            await asyncio.wait_for(event.wait(), timeout=_TIMEOUT_SECONDS)
            logger.info("confirmation_event_received", chat_id=chat_id)
        except asyncio.TimeoutError:
            _pending.pop(chat_id, None)
            await bot.send_message(
                chat_id=chat_id,
                text=f"⏱ Время ожидания истекло ({_TIMEOUT_SECONDS}с). Операция отменена.",
            )
            logger.warning("confirmation_timeout", chat_id=chat_id)
            return False

        result = _pending.pop(chat_id, {}).get("approved", False)
        logger.info("confirmation_result", chat_id=chat_id, approved=result)
        return result

    return _confirm


def resolve_pending(chat_id: int, approved: bool) -> bool:
    """Resolve a pending confirmation programmatically (e.g. from text reply).

    Returns True if there was a pending confirmation to resolve, False otherwise.
    """
    pending = _pending.get(chat_id)
    if not pending:
        return False
    pending["approved"] = approved
    pending["event"].set()
    logger.info("confirmation_resolved", chat_id=chat_id, approved=approved)
    return True


async def handle_confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button presses from confirmation messages."""
    logger.info(
        "confirmation_callback_invoked",
        data=getattr(update.callback_query, "data", None),
        pending_keys=list(_pending.keys()),
    )

    query = update.callback_query
    if query is None:
        logger.warning("confirmation_callback_no_query")
        return

    data = query.data or ""
    if not data.startswith(f"{_CB_PREFIX}:"):
        logger.warning("confirmation_callback_prefix_mismatch", data=data)
        return

    parts = data.split(":")
    if len(parts) != 3:
        logger.warning("confirmation_callback_bad_parts", parts=parts)
        return

    try:
        chat_id = int(parts[1])
    except ValueError:
        logger.warning("confirmation_callback_bad_chat_id", raw=parts[1])
        return

    approved = parts[2] == "yes"
    logger.info(
        "confirmation_callback_resolving",
        chat_id=chat_id,
        found=chat_id in _pending,
        pending_keys=list(_pending.keys()),
        approved=approved,
    )

    # Resolve BEFORE answering so the waiting coroutine is unblocked ASAP.
    resolved = resolve_pending(chat_id, approved)

    # Dismiss button spinner — ignore failures (callbacks expire after ~30s on Telegram's side)
    try:
        await query.answer()
    except Exception as exc:
        logger.warning("confirmation_answer_failed", error=str(exc))

    if resolved:
        label = "✅ Операция подтверждена." if approved else "❌ Операция отменена."
    else:
        # Timeout already fired — don't falsely show "confirmed"
        label = "⏱ Время ожидания истекло. Операция уже была отменена."
    try:
        await query.edit_message_text(label)
    except Exception as exc:
        logger.warning("confirmation_edit_failed", error=str(exc))
