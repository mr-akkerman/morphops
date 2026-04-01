"""Bot entry point.

Builds the Telegram Application, wires up handlers, and runs polling.
Agent startup (loading tools, MCP, context) happens via post_init hook
so it runs inside the Application's async context before polling begins.
"""
from __future__ import annotations

import logging

import structlog
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

import config
from agent.core import AdminAgent
from bot.confirmation import handle_confirmation_callback
from bot.handlers import (
    handle_document,
    handle_message,
    handle_reset,
    handle_start,
    set_agent,
)

logger = structlog.get_logger()


class _StructlogBridge(logging.Handler):
    """Forward Python stdlib logging records to structlog so PTB internals are visible."""

    _LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "critical",
    }

    def emit(self, record: logging.LogRecord) -> None:
        level = self._LEVEL_MAP.get(record.levelno, "info")
        bound = structlog.get_logger(record.name)
        log_fn = getattr(bound, level)
        msg = record.getMessage()
        if record.exc_info:
            log_fn(msg, exc_info=record.exc_info)
        else:
            log_fn(msg)


def _configure_stdlib_logging() -> None:
    """Redirect all Python stdlib logging (incl. PTB internals) through structlog."""
    root = logging.getLogger()
    if not any(isinstance(h, _StructlogBridge) for h in root.handlers):
        root.addHandler(_StructlogBridge())
    root.setLevel(logging.DEBUG)


# Module-level agent instance — created once, shared across handlers
_agent: AdminAgent | None = None


async def _post_init(app: Application) -> None:
    """Called by PTB after Application.initialize() — before polling starts."""
    _configure_stdlib_logging()
    global _agent
    _agent = app.bot_data["agent"]
    logger.info("agent_startup_begin")
    await _agent.startup()
    logger.info("agent_startup_done", tools=len(_agent.registry.tools))


async def _error_handler(update: object, context) -> None:
    """Global PTB error handler — logs every exception that slips through handlers."""
    logger.error(
        "ptb_unhandled_error",
        update=str(update)[:200] if update else None,
        error=str(context.error),
        exc_info=context.error,
    )


async def _post_shutdown(app: Application) -> None:
    """Called by PTB during graceful shutdown."""
    agent = app.bot_data.get("agent")
    if agent:
        await agent.mcp_client.disconnect_all()
        logger.info("agent_shutdown_done")


def build_app() -> Application:
    """Build and configure the Telegram Application."""
    agent = AdminAgent()
    set_agent(agent)

    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)   # Required: allows CallbackQuery to be processed
        .post_init(_post_init)      # while handle_message is blocked on event.wait()
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.bot_data["agent"] = agent

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("reset", handle_reset))
    # block=False: run the callback handler in its own asyncio task so it is
    # not serialised behind the message handler that is blocked on event.wait().
    app.add_handler(
        CallbackQueryHandler(handle_confirmation_callback, pattern=r"^sec_confirm:", block=False)
    )
    # Documents must be matched BEFORE text, otherwise captions get double-handled
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(_error_handler)

    return app


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("missing_env", var="TELEGRAM_BOT_TOKEN")
        return

    if not config.ANTHROPIC_API_KEY:
        logger.error("missing_env", var="ANTHROPIC_API_KEY")
        return

    logger.info("bot_starting", model=config.LLM_MODEL)
    app = build_app()
    app.run_polling(
        drop_pending_updates=True,
        # Explicitly request all update types we need so Telegram doesn't
        # silently skip callback_query updates from a previous session config.
        allowed_updates=[
            "message",
            "callback_query",
            "edited_message",
        ],
    )


if __name__ == "__main__":
    main()
