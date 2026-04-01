"""Built-in database tools.

Provides three built-in tools that are always available:
  - test_db_connection(dsn)    — verify a DSN is reachable
  - execute_raw_sql(query, params_json) — run any SQL against the active DB
  - get_db_schema_from_db()   — introspect tables/columns via INFORMATION_SCHEMA

These are *async* functions — AdminAgentLoop._execute_tool() awaits them
directly without going through a thread executor.

When the agent creates custom query tools via write_tool(), those tools
import and call execute_raw_sql() from this module.

Supported databases (MVP): PostgreSQL via asyncpg.
DSN format: postgresql://user:pass@host:port/dbname
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, datetime
from decimal import Decimal

import asyncpg
import structlog

import config
from agent.confirmation import confirmation_manager

logger = structlog.get_logger()


def _dangerous_sql_description(query: str) -> str | None:
    """Return a human-readable danger description, or None if query is safe."""
    q = query.strip().upper()

    if re.search(r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX|SEQUENCE|FUNCTION|PROCEDURE|TRIGGER|VIEW)\b", q):
        return "DROP — необратимое удаление объекта базы данных"

    if re.search(r"\bTRUNCATE\b", q):
        return "TRUNCATE — мгновенная очистка таблицы (необратимо)"

    # Any DELETE is potentially destructive — always require confirmation
    if re.search(r"\bDELETE\s+FROM\b", q):
        return "DELETE — удаление строк из таблицы (необратимо)"

    # UPDATE without WHERE — modifies every row in the table
    if re.search(r"\bUPDATE\b", q) and re.search(r"\bSET\b", q) and not re.search(r"\bWHERE\b", q):
        return "UPDATE без WHERE — изменение ВСЕХ строк таблицы"

    # ALTER TABLE ... DROP
    if re.search(r"\bALTER\s+TABLE\b", q) and re.search(r"\bDROP\b", q):
        return "ALTER TABLE ... DROP — необратимое изменение структуры"

    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_dsn() -> str:
    """Read the active DB DSN from the environment."""
    dsn = os.environ.get("DB_DSN", "").strip()
    if not dsn:
        raise RuntimeError(
            "DB_DSN переменная окружения не задана. "
            "Используй save_env_var('DB_DSN', 'postgresql://...') сначала."
        )
    return dsn


def _serialize_value(v):
    """Convert a PostgreSQL value to a JSON-serialisable Python type."""
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, (list, tuple)):
        return [_serialize_value(i) for i in v]
    return v


def _serialize_rows(rows) -> list[dict]:
    return [{k: _serialize_value(v) for k, v in dict(row).items()} for row in rows]


def _format_results(rows: list[dict], limit_display: int = 50) -> str:
    """Format query results as a readable string for the agent."""
    if not rows:
        return "Запрос выполнен, результатов нет."
    total = len(rows)
    display = rows[:limit_display]
    try:
        text = json.dumps(display, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(display)
    suffix = f"\n\n... показано {limit_display} из {total}" if total > limit_display else ""
    return text + suffix


def _is_select_query(query: str) -> bool:
    kw = query.strip().upper().lstrip("(")
    return any(kw.startswith(k) for k in ("SELECT", "WITH", "EXPLAIN", "SHOW", "TABLE", "VALUES"))


# ── Built-in tools ─────────────────────────────────────────────────────────────

async def test_db_connection(dsn: str) -> str:
    """Test a database connection using the provided DSN.

    Connects, runs SELECT 1, then disconnects. Does NOT modify DB_DSN env var —
    use save_env_var('DB_DSN', dsn) separately to persist it.

    Args:
        dsn: PostgreSQL connection string, e.g. postgresql://user:pass@host:5432/dbname

    Returns:
        Success message with server version, or an error description.
    """
    try:
        conn = await asyncpg.connect(dsn, timeout=10)
        try:
            version = await conn.fetchval("SELECT version()")
        finally:
            await conn.close()
        logger.info("db_connection_ok", dsn=dsn[:40] + "...")
        return f"Подключение успешно.\nСервер: {version}"
    except Exception as exc:
        logger.error("db_connection_failed", error=str(exc))
        return f"Ошибка подключения: {exc}"


async def execute_raw_sql(query: str, params_json: str = "") -> str:
    """Execute a SQL query against the active database (DB_DSN env var).

    Supports SELECT, INSERT, UPDATE, DELETE, CREATE, and any other statement.
    For SELECT-like queries returns the result rows as JSON.
    For modification queries returns the status string (e.g. "UPDATE 3").

    Args:
        query: SQL query. Use $1, $2, ... for positional parameters.
        params_json: Optional JSON array of positional parameter values,
                     e.g. '[5000, 100]'. Leave empty for no parameters.

    Returns:
        Query results as formatted JSON, or execution status for non-SELECT queries.
    """
    dsn = _get_dsn()

    # Read-only mode: block any non-SELECT statement
    if config.DB_READONLY and not _is_select_query(query):
        return (
            "Ошибка: база данных подключена в режиме read-only (DB_READONLY=true). "
            "Разрешены только SELECT/EXPLAIN/SHOW/WITH запросы."
        )

    # Parse params
    params: list = []
    if params_json and params_json.strip():
        try:
            parsed = json.loads(params_json)
            if isinstance(parsed, list):
                params = parsed
            else:
                return f"Ошибка: params_json должен быть JSON-массивом, получено: {params_json!r}"
        except json.JSONDecodeError as exc:
            return f"Ошибка парсинга params_json: {exc}"

    # Check for dangerous operations and ask for confirmation
    danger = _dangerous_sql_description(query)
    if danger:
        description = (
            f"Опасная операция с базой данных\n\n"
            f"Причина: {danger}\n\n"
            f"Запрос:\n{query}"
        )
        approved = await confirmation_manager.ask(description)
        if not approved:
            return "Операция отменена пользователем."

    try:
        conn = await asyncpg.connect(dsn, timeout=10)
        try:
            if _is_select_query(query):
                rows = await conn.fetch(query, *params)
                data = _serialize_rows(rows)
                return _format_results(data)
            else:
                status = await conn.execute(query, *params)
                return f"Запрос выполнен: {status}"
        finally:
            await conn.close()

    except asyncpg.PostgresError as exc:
        logger.error("sql_error", query=query[:100], error=str(exc))
        return f"Ошибка PostgreSQL: {exc}"
    except Exception as exc:
        logger.error("db_execute_error", error=str(exc))
        return f"Ошибка выполнения запроса: {exc}"


async def get_db_schema_from_db() -> str:
    """Introspect the active database and return its schema as structured text.

    Queries INFORMATION_SCHEMA for all tables and columns in the public schema.
    Use this when the user connects a DB without providing a schema file.

    Returns:
        Human-readable schema: table names with columns, types, and nullability.
    """
    dsn = _get_dsn()

    query = """
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            c.character_maximum_length,
            c.is_nullable,
            c.column_default,
            COALESCE(
                (SELECT 'PK'
                 FROM information_schema.table_constraints tc
                 JOIN information_schema.key_column_usage kcu
                   ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_name      = kcu.table_name
                 WHERE tc.constraint_type = 'PRIMARY KEY'
                   AND kcu.table_name     = c.table_name
                   AND kcu.column_name    = c.column_name
                 LIMIT 1),
                ''
            ) AS key_type
        FROM information_schema.columns c
        WHERE c.table_schema = 'public'
        ORDER BY c.table_name, c.ordinal_position
    """

    try:
        conn = await asyncpg.connect(dsn, timeout=10)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
    except Exception as exc:
        return f"Ошибка получения схемы: {exc}"

    if not rows:
        return "Таблицы в схеме public не найдены."

    # Group by table
    tables: dict[str, list] = {}
    for row in rows:
        tbl = row["table_name"]
        tables.setdefault(tbl, []).append(row)

    lines = [f"Схема БД ({len(tables)} таблиц):\n"]
    for tbl, cols in tables.items():
        lines.append(f"TABLE {tbl} (")
        col_lines = []
        for col in cols:
            dtype = col["data_type"]
            if col["character_maximum_length"]:
                dtype += f"({col['character_maximum_length']})"
            nullable = "" if col["is_nullable"] == "YES" else " NOT NULL"
            default = f" DEFAULT {col['column_default']}" if col["column_default"] else ""
            pk = " [PK]" if col["key_type"] == "PK" else ""
            col_lines.append(f"    {col['column_name']} {dtype}{nullable}{default}{pk}")
        lines.append(",\n".join(col_lines))
        lines.append(");\n")

    return "\n".join(lines)
