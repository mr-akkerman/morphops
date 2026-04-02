"""Telegram transport layer.

Formats AgentStep objects into Telegram messages and handles sending them.

LiveMessage — the main abstraction for streaming responses:
  - Private chats: uses sendMessageDraft (Bot API 9.5 native streaming)
  - Group chats: sends one message then edits it on every update
All public functions are thin async wrappers — no business logic here.
"""
from __future__ import annotations

import time
import config
from telegram import Bot
from telegram.constants import ChatAction
from telegram.error import TelegramError
from kerb.agent.execution import AgentStep
import structlog

logger = structlog.get_logger()

# Telegram hard limit per message
_TG_MAX_LEN = 4096


def _sanitize(text: str) -> str:
    """Redact sensitive credential values from outgoing messages.

    Replaces exact values of ANTHROPIC_API_KEY and TELEGRAM_BOT_TOKEN
    with a [REDACTED] placeholder so they can never be leaked to chat.
    """
    sensitive = [
        v for v in (config.ANTHROPIC_API_KEY, config.TELEGRAM_BOT_TOKEN)
        if v and len(v) > 8
    ]
    for val in sensitive:
        if val in text:
            placeholder = f"[REDACTED:{val[:4]}...{val[-4:]}]"
            text = text.replace(val, placeholder)
            logger.warning("sensitive_value_redacted_from_output")
    return text


def _format_args(action_input) -> str:
    """Format tool arguments for display."""
    if not action_input:
        return ""
    if isinstance(action_input, dict):
        lines = []
        for k, v in action_input.items():
            v_str = repr(v)
            if len(v_str) > 200:
                v_str = v_str[:200] + "…"
            lines.append(f"  {k} = {v_str}")
        result = "\n".join(lines)
    else:
        result = str(action_input)
        if len(result) > 300:
            result = result[:300] + "…"
    return result


def _format_pre_step(step: AgentStep) -> str:
    """Format the pre-execution part of a step: thought + tool call announcement.

    Sent immediately when Claude decides to call a tool — before it actually runs.
    """
    parts: list[str] = []

    if step.thought:
        parts.append(f"🧠 {step.thought}")

    if step.action:
        args_str = _format_args(step.action_input)
        tool_line = f"🔧 Вызываю: {step.action}"
        if args_str:
            tool_line += f"\n{args_str}"
        parts.append(tool_line)

    return "\n\n".join(parts) if parts else "…"


def _format_observation(observation: str) -> str:
    """Format the post-execution part of a step: tool result.

    Sent after the tool has finished running.
    """
    obs = observation
    if len(obs) > 600:
        obs = obs[:600] + "\n… (обрезано)"
    return f"✅ Результат:\n{obs}"


def _split_text(text: str, limit: int = _TG_MAX_LEN) -> list[str]:
    """Split text into Telegram-sized chunks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


class LiveMessage:
    """A single Telegram message that is replaced on every agent step.

    Each call to append() overwrites the previous content — the user always sees
    only the current step, not an accumulation of all previous ones.
    At the end, finalize() replaces the step with the final answer.

    Private chats: uses sendMessageDraft (Bot API 9.5) — animated typing bubble
    that gets replaced each step. finalize() sends a permanent message and the
    draft disappears automatically.

    Group chats: sends one message then edits it (same message_id) on every step.
    finalize() edits in-place with the final answer.
    """

    def __init__(self, bot: Bot, chat_id: int, is_private: bool) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._is_private = is_private
        # draft_id must be non-zero; monotonic ms gives a unique value per session
        self._draft_id: int = (int(time.monotonic() * 1000) % 0x7FFFFFFF) or 1
        self._message_id: int | None = None

    async def append(self, text: str) -> None:
        """Replace the live message with new content (current step only)."""
        await self._push(text[:_TG_MAX_LEN] if len(text) > _TG_MAX_LEN else text)

    async def finalize(self, final_text: str) -> None:
        """Replace live content with the final answer.

        Private: sends a permanent message (draft bubble disappears automatically).
        Group: edits the existing live message in-place. If no intermediate steps
        were shown yet, falls back to sending a new message.
        """
        full = f"📋 {_sanitize(final_text)}"
        chunks = _split_text(full)

        if self._is_private:
            for chunk in chunks:
                try:
                    await self._bot.send_message(
                        chat_id=self._chat_id, text=chunk, parse_mode="Markdown"
                    )
                except TelegramError:
                    try:
                        await self._bot.send_message(chat_id=self._chat_id, text=chunk)
                    except TelegramError as exc:
                        logger.error("finalize_send_failed", error=str(exc))
        else:
            first, *rest = chunks
            if self._message_id is not None:
                try:
                    await self._bot.edit_message_text(
                        chat_id=self._chat_id,
                        message_id=self._message_id,
                        text=first,
                        parse_mode="Markdown",
                    )
                except TelegramError:
                    try:
                        await self._bot.edit_message_text(
                            chat_id=self._chat_id,
                            message_id=self._message_id,
                            text=first,
                        )
                    except TelegramError as exc:
                        logger.error("finalize_edit_failed", error=str(exc))
                for chunk in rest:
                    try:
                        await self._bot.send_message(chat_id=self._chat_id, text=chunk)
                    except TelegramError as exc:
                        logger.error("finalize_overflow_failed", error=str(exc))
            else:
                for chunk in chunks:
                    try:
                        await self._bot.send_message(
                            chat_id=self._chat_id, text=chunk, parse_mode="Markdown"
                        )
                    except TelegramError:
                        try:
                            await self._bot.send_message(chat_id=self._chat_id, text=chunk)
                        except TelegramError as exc:
                            logger.error("finalize_send_failed", error=str(exc))

    async def _push(self, text: str) -> None:
        """Push text to Telegram, replacing the previous content entirely."""
        if self._is_private:
            try:
                await self._bot.send_message_draft(
                    chat_id=self._chat_id,
                    draft_id=self._draft_id,
                    text=text,
                )
            except TelegramError as exc:
                logger.error("live_draft_failed", error=str(exc))
        else:
            if self._message_id is None:
                try:
                    msg = await self._bot.send_message(
                        chat_id=self._chat_id, text=text
                    )
                    self._message_id = msg.message_id
                except TelegramError as exc:
                    logger.error("live_send_failed", error=str(exc))
            else:
                try:
                    await self._bot.edit_message_text(
                        chat_id=self._chat_id,
                        message_id=self._message_id,
                        text=text,
                    )
                except TelegramError as exc:
                    logger.warning("live_edit_failed", error=str(exc))


async def send_step(live: LiveMessage, step: AgentStep) -> None:
    """Replace the live message with the current reasoning step.

    Called twice per tool use:
    - Before execution (step.observation is None): thought + tool announcement
    - After execution (step.observation is set): tool result
    """
    if not step.observation:
        text = _sanitize(_format_pre_step(step))
    else:
        text = _sanitize(_format_observation(step.observation))
    await live.append(text)


async def send_typing(bot: Bot, chat_id: int) -> None:
    """Show the 'typing…' indicator in the chat."""
    try:
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except TelegramError:
        pass  # not critical


async def send_error(bot: Bot, chat_id: int, message: str) -> None:
    """Send a user-visible error notice."""
    try:
        await bot.send_message(chat_id=chat_id, text=f"❌ {message}")
    except TelegramError as exc:
        logger.error("send_error_failed", chat_id=chat_id, error=str(exc))
