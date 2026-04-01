"""MCP (Model Context Protocol) client.

Manages connections to MCP servers over stdio transport.
Each server is a subprocess (spawned via uvx or npx) that exposes tools
through the MCP protocol.

Connection lifecycle:
  1. install_mcp_server() writes config to agent_mcp/servers.json (no process yet)
  2. connect() spawns the process, initialises the session, lists tools,
     and returns tool descriptors for registration in the ToolRegistry
  3. disconnect_all() closes all sessions and kills all subprocesses
     (called on graceful bot shutdown)

Connections are kept alive via contextlib.AsyncExitStack so the MCP
subprocess stays running while the bot is active.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool
from kerb.agent.tools import Tool, create_tool

logger = structlog.get_logger()


def _expand_env(value: str) -> str:
    """Expand ${VAR_NAME} references using os.environ."""
    return re.sub(
        r"\$\{([^}]+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _mcp_params_to_kerb(input_schema: dict) -> dict[str, dict[str, Any]]:
    """Convert a JSON Schema inputSchema to kerb Tool parameters format."""
    properties = input_schema.get("properties", {})
    required_list = input_schema.get("required", [])
    params: dict[str, dict[str, Any]] = {}
    for prop_name, prop_schema in properties.items():
        params[prop_name] = {
            "type": prop_schema.get("type", "string"),
            "description": prop_schema.get("description", ""),
            "required": prop_name in required_list,
        }
    return params


def _make_tool_name(server_name: str, mcp_tool_name: str) -> str:
    """Build a registry tool name: mcp_{server}_{tool}, sanitised to snake_case."""
    raw = f"mcp_{server_name}_{mcp_tool_name}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)


class MCPClient:
    """Manages connections to MCP servers and their tool wrappers."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        # name → {"session": ClientSession, "stack": AsyncExitStack, "tool_names": list[str]}
        self._connections: dict[str, dict] = {}
        # Runtime trust map — updated by set_trust() without requiring reconnect
        self._trusted: dict[str, bool] = {}

    # ── Config helpers ─────────────────────────────────────────────────────────

    def load_config(self) -> dict:
        """Read servers.json; return empty structure if the file doesn't exist."""
        if not self.config_path.exists():
            return {"servers": {}}
        with open(self.config_path, encoding="utf-8") as f:
            return json.load(f)

    def save_config(self, cfg: dict) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def set_server_status(self, name: str, status: str) -> None:
        cfg = self.load_config()
        if name in cfg.get("servers", {}):
            cfg["servers"][name]["status"] = status
            self.save_config(cfg)

    def set_trust(self, name: str, trusted: bool) -> None:
        """Update trust status at runtime and persist to servers.json.

        Takes effect immediately for already-connected servers — no reconnect needed.
        """
        self._trusted[name] = trusted
        cfg = self.load_config()
        if name in cfg.get("servers", {}):
            cfg["servers"][name]["trusted"] = trusted
            self.save_config(cfg)
        logger.info("mcp_server_trust_updated", name=name, trusted=trusted)

    def get_trust(self, name: str) -> bool:
        """Return current trust status for *name* (False if unknown)."""
        return self._trusted.get(name, False)

    # ── Server management ──────────────────────────────────────────────────────

    def save_server(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
    ) -> None:
        """Persist a server definition to servers.json (status=inactive, trusted=false)."""
        cfg = self.load_config()
        cfg.setdefault("servers", {})[name] = {
            "command": command,
            "args": args,
            "env": env,
            "status": "inactive",
            "trusted": False,
        }
        self.save_config(cfg)
        logger.info("mcp_server_saved", name=name, command=command)

    async def connect(self, name: str, server_cfg: dict) -> list[dict]:
        """Spawn MCP server process, initialise session, and return tool descriptors.

        Args:
            name: Server name (used as prefix for registered tool names).
            server_cfg: Dict with keys: command, args, env.

        Returns:
            List of dicts: {name, description, func, parameters} ready for
            registration in ToolRegistry via Tool(...) or create_tool().

        Raises:
            RuntimeError: If the server process fails to start or initialise.
        """
        if name in self._connections:
            logger.info("mcp_already_connected", name=name)
            return []

        # Expand ${VAR} in env values
        raw_env = server_cfg.get("env", {})
        expanded_env = {k: _expand_env(v) for k, v in raw_env.items()}
        process_env = {**os.environ, **expanded_env}

        params = StdioServerParameters(
            command=server_cfg["command"],
            args=server_cfg.get("args", []),
            env=process_env,
        )

        logger.info("mcp_connecting", name=name, command=params.command, args=params.args)

        stack = contextlib.AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session: ClientSession = await stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            result = await session.list_tools()
            mcp_tools: list[McpTool] = result.tools

        except Exception as exc:
            await stack.aclose()
            logger.error("mcp_connect_failed", name=name, error=str(exc))
            raise RuntimeError(f"Не удалось подключиться к MCP серверу '{name}': {exc}") from exc

        tool_descriptors: list[dict] = []
        tool_names: list[str] = []

        for mcp_tool in mcp_tools:
            tool_name = _make_tool_name(name, mcp_tool.name)
            tool_names.append(tool_name)

            # Capture loop variables for the closure
            _session = session
            _mcp_name = mcp_tool.name
            _tool_name = tool_name

            _server_name = name

            async def _wrapper(
                _session=_session,
                _mcp_name=_mcp_name,
                _tool_name=tool_name,
                _server=_server_name,
                _client=self,
                **kwargs,
            ):
                from agent.confirmation import confirmation_manager
                from agent.safety import is_dangerous_name

                trusted = _client._trusted.get(_server, False)

                if not trusted:
                    confirm_reason = "сервер не в списке доверенных"
                elif is_dangerous_name(_mcp_name):
                    confirm_reason = "имя инструмента содержит признаки деструктивной операции"
                else:
                    confirm_reason = None

                if confirm_reason:
                    args_preview = ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) or "—"
                    description = (
                        f"Вызов MCP-инструмента\n\n"
                        f"Сервер: {_server}\n"
                        f"Инструмент: {_tool_name}\n"
                        f"Аргументы: {args_preview}\n\n"
                        f"Причина проверки: {confirm_reason}."
                    )
                    approved = await confirmation_manager.ask(description)
                    if not approved:
                        return "Операция отменена пользователем."

                call_result = await _session.call_tool(_mcp_name, kwargs)
                if call_result.isError:
                    texts = [
                        item.text for item in call_result.content
                        if hasattr(item, "text")
                    ]
                    return f"MCP Error: {' '.join(texts)}"
                texts = [
                    item.text for item in call_result.content
                    if hasattr(item, "text")
                ]
                return "\n".join(texts) if texts else "(no output)"

            _wrapper.__name__ = tool_name
            _wrapper.__doc__ = (
                f"{mcp_tool.description or tool_name}\n"
                f"[MCP server: {name}, tool: {mcp_tool.name}]"
            )

            tool_descriptors.append(
                {
                    "name": tool_name,
                    "description": _wrapper.__doc__,
                    "func": _wrapper,
                    "parameters": _mcp_params_to_kerb(
                        mcp_tool.inputSchema.model_dump()
                        if hasattr(mcp_tool.inputSchema, "model_dump")
                        else (mcp_tool.inputSchema or {})
                    ),
                }
            )

        self._connections[name] = {
            "session": session,
            "stack": stack,
            "tool_names": tool_names,
        }
        self.set_server_status(name, "active")

        # Initialise trust from config (can be overridden later via set_trust)
        self._trusted.setdefault(name, server_cfg.get("trusted", False))

        logger.info("mcp_connected", name=name, tools=len(mcp_tools))
        return tool_descriptors

    async def connect_all_active(self) -> None:
        """Reconnect to all servers marked active in servers.json.

        Skips servers that fail to connect (logs error, continues).
        Called at startup; tool registration is handled by caller (AdminAgent).
        """
        cfg = self.load_config()
        for name, srv_cfg in cfg.get("servers", {}).items():
            if srv_cfg.get("status") == "active":
                try:
                    await self.connect(name, srv_cfg)
                except Exception as exc:
                    logger.error("mcp_startup_connect_failed", server=name, error=str(exc))

    async def disconnect_all(self) -> None:
        """Close all MCP server connections gracefully."""
        for name, conn in list(self._connections.items()):
            try:
                await conn["stack"].aclose()
                logger.info("mcp_disconnected", name=name)
            except Exception as exc:
                logger.warning("mcp_disconnect_error", name=name, error=str(exc))
        self._connections.clear()

    def list_servers(self) -> list[str]:
        return list(self._connections.keys())

    def connected_tool_names(self, server_name: str) -> list[str]:
        return self._connections.get(server_name, {}).get("tool_names", [])
