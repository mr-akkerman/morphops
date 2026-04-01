from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
_raw_ids = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: list[int] = [
    int(x.strip()) for x in _raw_ids.split(",") if x.strip().isdigit()
]

# ─── Agent directories ────────────────────────────────────────────────────────
AGENT_TOOLS_DIR: str = os.getenv("AGENT_TOOLS_DIR", "./agent_tools")
AGENT_MCP_DIR: str = os.getenv("AGENT_MCP_DIR", "./agent_mcp")
AGENT_CONTEXT_DIR: str = os.getenv("AGENT_CONTEXT_DIR", "./agent_context")
AGENT_TMP_DIR: str = os.getenv("AGENT_TMP_DIR", "./agent_tmp")

# ─── Limits ───────────────────────────────────────────────────────────────────
MAX_TOOL_EXECUTION_TIME: int = int(os.getenv("MAX_TOOL_EXECUTION_TIME", "30"))
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "15"))

# ─── Security ─────────────────────────────────────────────────────────────────
# Max user-initiated conversation turns kept in history (older turns are dropped)
MAX_HISTORY_PAIRS: int = int(os.getenv("MAX_HISTORY_PAIRS", "30"))
# Max input message length in characters (longer messages are truncated)
MAX_INPUT_CHARS: int = int(os.getenv("MAX_INPUT_CHARS", "8000"))
# Max requests per user per hour (0 = unlimited)
MAX_REQUESTS_PER_HOUR: int = int(os.getenv("MAX_REQUESTS_PER_HOUR", "40"))
# Max size of a single context file written via write_context (chars)
MAX_CONTEXT_FILE_CHARS: int = int(os.getenv("MAX_CONTEXT_FILE_CHARS", "20000"))
# Max total size of all context files injected into the system prompt (chars)
MAX_CONTEXT_TOTAL_CHARS: int = int(os.getenv("MAX_CONTEXT_TOTAL_CHARS", "50000"))
# If true, only SELECT/EXPLAIN/SHOW queries are allowed via execute_raw_sql
DB_READONLY: bool = os.getenv("DB_READONLY", "false").lower() in ("1", "true", "yes")
