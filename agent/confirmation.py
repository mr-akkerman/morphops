"""Confirmation manager — mediates between agent tools and the Telegram UI.

Agent tools (db_tools, api_tools) call confirmation_manager.ask() when a
dangerous operation is detected. The bot injects a Telegram-backed callback
via set_callback() before each message and clears it afterwards.

This module has no dependency on bot/ so it can be safely imported from tools/.
"""
from __future__ import annotations

from typing import Awaitable, Callable

ConfirmCallback = Callable[[str], Awaitable[bool]]


class ConfirmationManager:
    """Singleton that decouples "needs confirmation" from "has Telegram access"."""

    def __init__(self) -> None:
        self._callback: ConfirmCallback | None = None

    def set_callback(self, callback: ConfirmCallback | None) -> None:
        """Set (or clear) the active confirmation callback."""
        self._callback = callback

    async def ask(self, description: str) -> bool:
        """Ask for confirmation. Returns False (deny) if no callback is set."""
        if self._callback is None:
            return False
        return await self._callback(description)


# Module-level singleton — imported by db_tools and api_tools
confirmation_manager = ConfirmationManager()
