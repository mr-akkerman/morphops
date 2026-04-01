"""Telegram transport layer.

Formats AgentStep objects into Telegram messages and handles sending them.
All public functions are thin async wrappers — no business logic here.
"""
from __future__ import annotations

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


async def send_step(bot: Bot, chat_id: int, step: AgentStep) -> None:
    """Send one intermediate reasoning step.

    Called twice per tool use:
    - Before execution (step.observation is None): sends thought + tool announcement
    - After execution (step.observation is set): sends tool result
    """
    if not step.observation:  # None or "" — pre-execution phase
        text = _sanitize(_format_pre_step(step))
    else:
        text = _sanitize(_format_observation(step.observation))

    for chunk in _split_text(text):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk)
        except TelegramError as exc:
            logger.error("send_step_failed", chat_id=chat_id, error=str(exc))


async def send_final(bot: Bot, chat_id: int, text: str) -> None:
    """Send the agent's final answer.

    Tries Markdown formatting first; falls back to plain text if Telegram
    rejects the message (e.g. unbalanced markdown symbols).
    """
    full = f"📋 {_sanitize(text)}"
    for chunk in _split_text(full):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
        except TelegramError:
            # Markdown parse error — send without formatting
            try:
                await bot.send_message(chat_id=chat_id, text=chunk)
            except TelegramError as exc:
                logger.error("send_final_failed", chat_id=chat_id, error=str(exc))


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
