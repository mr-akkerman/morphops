"""Meta-tools: give the agent the ability to extend itself at runtime.

These are the foundational tools registered at startup. They allow the agent to:
- Write new Python tools and immediately use them
- Persist context (DB schemas, API docs, role descriptions) across restarts
- Store credentials in .env
- Install and connect MCP servers (Stage 7)

Module-level state is injected once via initialize() during agent startup.
All functions are synchronous (blocking) — the event loop runs them in a
thread-pool executor via AdminAgentLoop._execute_tool().
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

import config

if TYPE_CHECKING:
    from agent.tool_registry import ToolRegistry

logger = structlog.get_logger()

# ── Module-level state (set once via initialize()) ────────────────────────────

_registry: "ToolRegistry | None" = None
_tools_dir: Path = Path("./agent_tools")
_context_dir: Path = Path("./agent_context")
_tmp_dir: Path = Path("./agent_tmp")
_env_file: Path = Path(".env")
_mcp_client = None  # agent.mcp_client.MCPClient, set via initialize()

# Variables that cannot be overwritten by the agent
_PROTECTED_ENV_VARS: frozenset[str] = frozenset({
    "ANTHROPIC_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_USER_IDS",
})


def initialize(
    registry: "ToolRegistry",
    tools_dir: str | Path,
    context_dir: str | Path,
    tmp_dir: str | Path,
    env_file: str | Path = ".env",
    mcp_client=None,
) -> None:
    """Inject runtime state. Called once from AdminAgent.startup()."""
    global _registry, _tools_dir, _context_dir, _tmp_dir, _env_file, _mcp_client
    _registry = registry
    _tools_dir = Path(tools_dir)
    _context_dir = Path(context_dir)
    _tmp_dir = Path(tmp_dir)
    _env_file = Path(env_file)
    _mcp_client = mcp_client


def _require_init() -> None:
    if _registry is None:
        raise RuntimeError("meta_tools.initialize() has not been called yet.")


# ── AST security checker ──────────────────────────────────────────────────────

class _DangerChecker(ast.NodeVisitor):
    """AST visitor that rejects dangerous patterns in agent-written tools."""

    # Modules whose import is outright blocked.
    # Key principle: agent tools must go through built-in tools (execute_raw_sql,
    # http_get, etc.) so the confirmation gate and SSRF checks always apply.
    _BLOCKED_IMPORTS = frozenset({
        # Shell / process execution
        "subprocess", "ctypes", "multiprocessing", "importlib",
        # Direct DB access — bypasses execute_raw_sql confirmation gate
        "asyncpg", "psycopg2", "psycopg", "sqlite3", "pymysql", "aiomysql",
        "motor", "pymongo",
        # Direct HTTP — bypasses http_delete confirmation + SSRF check
        "httpx", "requests", "aiohttp", "urllib", "urllib2", "urllib3",
        # Raw network — bypasses SSRF check
        "socket", "ssl",
        # File system — bypasses open() block
        "pathlib", "shutil", "tempfile", "fileinput",
    })

    # Dangerous bare-name calls (builtins)
    _BLOCKED_BUILTINS = frozenset({"eval", "exec", "compile", "__import__"})

    # Dangerous os.* attribute calls
    _BLOCKED_OS_ATTRS = frozenset({
        "system", "popen", "remove", "unlink", "rmdir", "removedirs",
        "execv", "execve", "execvp", "execvpe", "spawnl", "spawnle",
        "fork", "kill", "killpg",
    })

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in self._BLOCKED_IMPORTS:
                self.errors.append(f"Запрещённый импорт: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in self._BLOCKED_IMPORTS:
                self.errors.append(f"Запрещённый импорт: from {node.module}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block os.environ access (env var exfiltration)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "environ"
        ):
            self.errors.append("Запрещён доступ к os.environ (утечка credentials)")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # eval / exec / compile / __import__
        if isinstance(node.func, ast.Name) and node.func.id in self._BLOCKED_BUILTINS:
            self.errors.append(f"Запрещённый вызов: {node.func.id}()")

        # os.system(...), os.remove(...), etc.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr in self._BLOCKED_OS_ATTRS
        ):
            self.errors.append(f"Запрещённый вызов: os.{node.func.attr}()")

        # open(...) — block entirely (both read and write) to prevent file exfiltration
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self.errors.append(
                "Запрещено: open() — для работы с файлами используй "
                "встроенные инструменты write_tmp / read_tmp"
            )

        self.generic_visit(node)


def _check_tool_code_safety(code: str) -> list[str]:
    """Parse code and return a list of security violation descriptions (empty = safe)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # syntax errors are caught separately before this is called
    checker = _DangerChecker()
    checker.visit(tree)
    return checker.errors


# ── Tool management ───────────────────────────────────────────────────────────

def write_tool(name: str, code: str) -> str:
    """Write Python source code to agent_tools/{name}.py and immediately register it.

    The code should define one or more public functions (no leading underscore).
    Each function MUST have:
    - A descriptive docstring (becomes the tool description for Claude)
    - Type-annotated parameters (used to build the tool's input schema)

    IMPORTANT — working with databases:
    - If a function needs to query the DB, make it `async def` and import
      execute_raw_sql at the top of the file:
          from tools.db_tools import execute_raw_sql
      Then call it with `await`:
          return await execute_raw_sql("SELECT ...", '[param1, param2]')
    - Never use asyncpg/psycopg2/os.environ directly — they are blocked by
      the security checker. Always go through execute_raw_sql.

    Example (simple sync tool):
        write_tool("greet", '''
        def greet(name: str) -> str:
            \"\"\"Greet a person by name.\"\"\"
            return f"Hello, {name}!"
        ''')

    Example (async DB tool):
        write_tool("count_users", '''
        from tools.db_tools import execute_raw_sql

        async def count_users() -> str:
            \"\"\"Return total number of users in the database.\"\"\"
            return await execute_raw_sql("SELECT COUNT(*) AS total FROM users")
        ''')

    Args:
        name: Module filename without .py extension. Use snake_case, e.g. "query_users".
        code: Valid Python source code defining one or more public functions.

    Returns:
        Confirmation message listing the names of functions that were registered.
    """
    _require_init()

    # Validate name: only identifiers allowed, no path traversal
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        return f"Ошибка: недопустимое имя '{name}'. Используй snake_case без расширения."

    # Syntax check before writing to disk
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"Ошибка синтаксиса в коде:\n{exc}"

    # Security check: block dangerous constructs
    violations = _check_tool_code_safety(code)
    if violations:
        return (
            "Ошибка безопасности — код содержит запрещённые конструкции:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )

    # Write file
    _tools_dir.mkdir(parents=True, exist_ok=True)
    dest = _tools_dir / f"{name}.py"
    dest.write_text(code, encoding="utf-8")
    logger.info("tool_file_written", path=str(dest))

    # Load into registry (hot-reload: registry handles sys.modules cleanup)
    loaded = _registry.load_from_file(dest)
    if not loaded:
        return (
            f"Файл {dest} записан, но публичные функции не найдены. "
            "Убедись, что функции не начинаются с '_' и имеют docstring."
        )

    names = ", ".join(t.name for t in loaded)
    return f"Инструменты зарегистрированы: {names}\nФайл: {dest}"


def list_tools() -> str:
    """Return the names and descriptions of all currently registered tools.

    Returns:
        Formatted list of tool names with their one-line descriptions.
    """
    _require_init()
    if not _registry.tools:
        return "Инструменты не зарегистрированы."
    lines = []
    for tool in sorted(_registry.tools, key=lambda t: t.name):
        desc_first_line = (tool.description or "").split("\n")[0].strip()
        lines.append(f"• {tool.name}: {desc_first_line}")
    return "\n".join(lines)


# ── Context management ────────────────────────────────────────────────────────

def write_context(name: str, content: str) -> str:
    """Save text to agent_context/{name}.md — persists across bot restarts.

    Use this to store:
    - Database schema descriptions and usage notes
    - API documentation summaries
    - Project role descriptions
    - Any notes the agent should remember

    Args:
        name: Filename without extension, e.g. "db_schema" or "project_role".
        content: Markdown text to write.

    Returns:
        Confirmation with the file path.
    """
    _require_init()
    if not re.match(r"^[a-zA-Z0-9_\-]+$", name):
        return f"Ошибка: недопустимое имя '{name}'. Используй только буквы, цифры, _ и -."

    if len(content) > config.MAX_CONTEXT_FILE_CHARS:
        return (
            f"Ошибка: содержимое слишком большое ({len(content)} символов). "
            f"Максимум — {config.MAX_CONTEXT_FILE_CHARS} символов на файл."
        )

    _context_dir.mkdir(parents=True, exist_ok=True)
    dest = _context_dir / f"{name}.md"
    dest.write_text(content, encoding="utf-8")
    logger.info("context_written", path=str(dest))
    return f"Контекст '{name}' сохранён: {dest}"


def read_context(name: str) -> str:
    """Read a file from agent_context/{name}.md and return its content.

    Args:
        name: Filename without extension.

    Returns:
        File content, or an error message if the file doesn't exist.
    """
    _require_init()
    path = _context_dir / f"{name}.md"
    if not path.exists():
        available = [p.stem for p in _context_dir.glob("*.md")] if _context_dir.exists() else []
        hint = f" Доступные файлы: {', '.join(available)}" if available else ""
        return f"Файл '{name}.md' не найден.{hint}"
    return path.read_text(encoding="utf-8")


def list_context() -> str:
    """Return the names of all files in agent_context/.

    Returns:
        Newline-separated list of context file names (without .md extension).
    """
    _require_init()
    if not _context_dir.exists():
        return "Директория agent_context/ пуста."
    files = sorted(p.stem for p in _context_dir.glob("*.md"))
    if not files:
        return "Контекстных файлов нет."
    return "\n".join(f"• {f}" for f in files)


# ── Temporary files ───────────────────────────────────────────────────────────

def write_tmp(name: str, content: str) -> str:
    """Write a temporary file to agent_tmp/{name}.

    Args:
        name: Filename including extension, e.g. "schema.sql" or "response.json".
        content: File content.

    Returns:
        Absolute path to the written file.
    """
    _require_init()
    # Prevent path traversal
    safe_name = Path(name).name
    _tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = _tmp_dir / safe_name
    dest.write_text(content, encoding="utf-8")
    return str(dest.resolve())


def read_tmp(name: str) -> str:
    """Read a file from agent_tmp/{name} and return its content.

    Args:
        name: Filename including extension.

    Returns:
        File content, or an error message if the file doesn't exist.
    """
    _require_init()
    safe_name = Path(name).name
    path = _tmp_dir / safe_name
    if not path.exists():
        return f"Временный файл '{safe_name}' не найден."
    return path.read_text(encoding="utf-8")


# ── Environment / credentials ─────────────────────────────────────────────────

def save_env_var(key: str, value: str) -> str:
    """Add or update an environment variable in the .env file.

    Also updates os.environ so the change takes effect immediately in the
    current process (no restart needed).

    Use this to store credentials: DB_DSN, API tokens, base URLs, etc.
    Variable names must be UPPER_SNAKE_CASE.

    Args:
        key: Variable name, e.g. "DB_DSN" or "VPN_API_TOKEN".
        value: Variable value.

    Returns:
        Confirmation message.
    """
    _require_init()
    if key in _PROTECTED_ENV_VARS:
        return (
            f"Ошибка: переменная '{key}' защищена и не может быть изменена через агента. "
            f"Защищённые переменные: {', '.join(sorted(_PROTECTED_ENV_VARS))}"
        )
    if not re.match(r"^[A-Z][A-Z0-9_]*$", key):
        return f"Ошибка: имя переменной '{key}' должно быть в формате UPPER_SNAKE_CASE."

    # Read existing .env lines
    if _env_file.exists():
        lines = _env_file.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    new_line = f'{key}="{value}"' if " " in value or not value else f"{key}={value}"

    # Update existing line or append
    updated = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = new_line
            updated = True
            break

    if not updated:
        lines.append(new_line)

    _env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update current process immediately
    os.environ[key] = value
    logger.info("env_var_saved", key=key)
    return f"Переменная {key} сохранена в {_env_file} и применена в текущем процессе."


# ── MCP tools ─────────────────────────────────────────────────────────────────

async def install_mcp_server(package: str, manager: str = "uvx", env_vars_json: str = "") -> str:
    """Save an MCP server configuration to agent_mcp/servers.json.

    Does NOT install anything yet — the server process is started on the first
    call to connect_mcp_server(). uvx/npx download packages on demand.

    Args:
        package: Package name, e.g. "mcp-server-postgres",
                 "@modelcontextprotocol/server-github", or a local path.
        manager: "uvx" for Python packages, "npx" for Node.js packages.
        env_vars_json: JSON object with environment variables to pass to the
                       server process. Use ${VAR} to reference existing env vars.
                       e.g. {"CONNECTION_STRING": "${DB_DSN}", "API_KEY": "${GITHUB_TOKEN}"}

    Returns:
        Confirmation with the server name and next steps.
    """
    _require_init()
    if _mcp_client is None:
        return "Ошибка: MCP client не инициализирован."

    if manager not in ("uvx", "npx"):
        return f"Ошибка: manager должен быть 'uvx' или 'npx', получено: '{manager}'"

    # Parse env_vars
    env: dict = {}
    if env_vars_json and env_vars_json.strip():
        import json
        try:
            parsed = json.loads(env_vars_json)
            if not isinstance(parsed, dict):
                return "Ошибка: env_vars_json должен быть JSON-объектом."
            env = parsed
        except json.JSONDecodeError as exc:
            return f"Ошибка парсинга env_vars_json: {exc}"

    # Derive a short name from the package (last dash-separated segment)
    name_raw = package.split("/")[-1]          # strip npm scope
    name_raw = re.sub(r"^mcp[-_]server[-_]", "", name_raw)  # strip common prefix
    name_raw = re.sub(r"[-_]mcp$", "", name_raw)            # strip common suffix
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name_raw).strip("_") or "mcp_server"

    from agent.confirmation import confirmation_manager

    description = (
        f"Установка MCP-сервера\n\n"
        f"Менеджер: {manager}\n"
        f"Пакет: {package}\n\n"
        f"Сервер будет запущен как подпроцесс с доступом к переменным окружения."
    )
    approved = await confirmation_manager.ask(description)
    if not approved:
        return "Установка MCP-сервера отменена пользователем."

    _mcp_client.save_server(
        name=name,
        command=manager,
        args=[package],
        env=env,
    )

    return (
        f"Сервер '{name}' сохранён в конфиге.\n"
        f"Команда: {manager} {package}\n"
        f"Для подключения и загрузки инструментов вызови: connect_mcp_server('{name}')"
    )


def list_mcp_servers() -> str:
    """Return the names, statuses, trust flags, and tool counts of all configured MCP servers.

    Returns:
        Formatted list of server names with status (active/inactive), trust level,
        and tool count. Untrusted servers show a lock icon — all their tool calls
        require user confirmation regardless of tool name.
    """
    _require_init()
    if _mcp_client is None:
        return "Ошибка: MCP client не инициализирован."

    cfg = _mcp_client.load_config()
    servers = cfg.get("servers", {})
    if not servers:
        return "MCP серверов не настроено."

    lines = []
    for name, srv in servers.items():
        status = srv.get("status", "unknown")
        trusted = _mcp_client.get_trust(name)
        status_icon = "🟢" if status == "active" else "⚪"
        trust_badge = "✅ trusted" if trusted else "🔒 untrusted"
        tool_count = len(_mcp_client.connected_tool_names(name))
        tool_info = f" ({tool_count} инструментов)" if status == "active" else ""
        lines.append(
            f"{status_icon} {name}: {status}{tool_info}  {trust_badge}"
            f"  [{srv['command']} {' '.join(srv['args'])}]"
        )

    return "\n".join(lines)


def trust_mcp_server(name: str) -> str:
    """Mark an MCP server as trusted so its tools no longer require confirmation by default.

    After trusting, only tools whose names contain destructive keywords
    (delete, purge, drop, wipe, etc.) will still require explicit confirmation.
    All other tools from this server will be called without interruption.

    Args:
        name: Server name as shown by list_mcp_servers().

    Returns:
        Confirmation message, or an error if the server is not found.
    """
    _require_init()
    if _mcp_client is None:
        return "Ошибка: MCP client не инициализирован."

    cfg = _mcp_client.load_config()
    if name not in cfg.get("servers", {}):
        available = list(cfg.get("servers", {}).keys())
        hint = f" Доступные серверы: {', '.join(available)}" if available else ""
        return f"Сервер '{name}' не найден в конфиге.{hint}"

    _mcp_client.set_trust(name, True)
    tool_names = _mcp_client.connected_tool_names(name)
    tools_info = f"\nЗагружено инструментов: {len(tool_names)}" if tool_names else ""
    return (
        f"Сервер '{name}' помечен как доверенный (trusted).\n"
        f"Теперь его инструменты вызываются без подтверждения,\n"
        f"кроме тех, чьё имя содержит деструктивные ключевые слова."
        f"{tools_info}"
    )


def untrust_mcp_server(name: str) -> str:
    """Put an MCP server back into quarantine mode — all its tool calls require confirmation.

    Args:
        name: Server name as shown by list_mcp_servers().

    Returns:
        Confirmation message, or an error if the server is not found.
    """
    _require_init()
    if _mcp_client is None:
        return "Ошибка: MCP client не инициализирован."

    cfg = _mcp_client.load_config()
    if name not in cfg.get("servers", {}):
        available = list(cfg.get("servers", {}).keys())
        hint = f" Доступные серверы: {', '.join(available)}" if available else ""
        return f"Сервер '{name}' не найден в конфиге.{hint}"

    _mcp_client.set_trust(name, False)
    return (
        f"Сервер '{name}' переведён в режим карантина (untrusted).\n"
        f"Все вызовы его инструментов теперь требуют подтверждения."
    )


async def connect_mcp_server(name: str) -> str:
    """Connect to a configured MCP server and load its tools into the registry.

    Spawns the server process (uvx/npx), initialises the MCP session, lists
    available tools, and registers them in the tool registry.

    Args:
        name: Server name as defined in agent_mcp/servers.json (returned by
              install_mcp_server).

    Returns:
        Confirmation with the list of tool names loaded from the server.
    """
    _require_init()
    if _mcp_client is None:
        return "Ошибка: MCP client не инициализирован."

    cfg = _mcp_client.load_config()
    server_cfg = cfg.get("servers", {}).get(name)
    if server_cfg is None:
        available = list(cfg.get("servers", {}).keys())
        hint = f" Доступные серверы: {', '.join(available)}" if available else ""
        return f"Сервер '{name}' не найден в конфиге.{hint}"

    try:
        from kerb.agent.tools import Tool
        tool_descriptors = await _mcp_client.connect(name, server_cfg)
    except Exception as exc:
        return f"Ошибка подключения к серверу '{name}': {exc}"

    if not tool_descriptors:
        return f"Сервер '{name}' подключён, но инструменты не найдены (или уже загружены)."

    # Register each MCP tool in the agent's tool registry
    for desc in tool_descriptors:
        tool = Tool(
            name=desc["name"],
            description=desc["description"],
            func=desc["func"],
            parameters=desc["parameters"],
        )
        _registry.register(tool)

    names_str = "\n".join(f"  • {d['name']}" for d in tool_descriptors)
    return (
        f"Сервер '{name}' подключён. Загружено {len(tool_descriptors)} инструментов:\n"
        f"{names_str}"
    )
