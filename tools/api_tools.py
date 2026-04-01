"""Built-in HTTP tools for working with external APIs.

Provides four async tools that are always available:
  - http_get(url, headers_json, params_json)
  - http_post(url, headers_json, json_body)
  - http_patch(url, headers_json, json_body)
  - http_delete(url, headers_json)

Headers, query params, and request bodies are passed as JSON strings so
Claude can construct them naturally without nested object schemas.

All functions are async — AdminAgentLoop._execute_tool() awaits them directly.

When the agent creates custom API wrapper tools via write_tool(), those tools
typically call these functions (or httpx directly) with pre-filled base URLs
and auth headers read from environment variables.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import textwrap
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from agent.confirmation import confirmation_manager

logger = structlog.get_logger()

_DEFAULT_TIMEOUT = 30.0
_MAX_RESPONSE_CHARS = 4000  # truncate large responses before returning to agent

# URL path segments that suggest a destructive POST/PATCH operation
_DESTRUCTIVE_URL_PATTERNS = re.compile(
    r"/(delete|remove|purge|wipe|clear|destroy|bulk.?delete|mass.?delete|drop|reset.?all|erase)",
    re.IGNORECASE,
)


def _check_ssrf(url: str) -> str | None:
    """Return an error string if the URL targets a private/reserved address, else None."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return "Некорректный URL: не удалось определить hostname."

        # Resolve hostname to IP addresses
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, ValueError) as exc:
        # Can't resolve — let httpx handle the error naturally
        logger.debug("ssrf_check_resolve_failed", hostname=url, error=str(exc))
        return None

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return (
                f"Запрос заблокирован: адрес {addr} ({hostname}) является "
                f"внутренним/зарезервированным. HTTP-инструменты работают только с публичными адресами."
            )
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_arg(value: str, arg_name: str) -> tuple[Any, str | None]:
    """Parse a JSON string argument.

    Returns (parsed_value, None) on success or (None, error_message) on failure.
    Empty / blank strings return ({} or None) as appropriate.
    """
    if not value or not value.strip():
        return {}, None
    try:
        return json.loads(value), None
    except json.JSONDecodeError as exc:
        return None, f"Ошибка парсинга {arg_name}: {exc}\nПолучено: {value!r}"


def _format_response(response: httpx.Response) -> str:
    """Format an httpx Response into a readable string for the agent."""
    status_line = f"HTTP {response.status_code} {response.reason_phrase}"

    # Try JSON body first
    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        try:
            data = response.json()
            body = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            body = response.text
    else:
        body = response.text

    # Truncate large bodies
    if len(body) > _MAX_RESPONSE_CHARS:
        body = body[:_MAX_RESPONSE_CHARS] + f"\n\n... (обрезано, показано {_MAX_RESPONSE_CHARS} из {len(body)} символов)"

    return f"{status_line}\n\n{body}"


def _handle_error(exc: Exception, url: str) -> str:
    """Convert an httpx exception into a readable error string."""
    if isinstance(exc, httpx.TimeoutException):
        return f"Таймаут запроса к {url} (>{_DEFAULT_TIMEOUT}с)"
    if isinstance(exc, httpx.ConnectError):
        return f"Ошибка подключения к {url}: {exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP ошибка {exc.response.status_code}: {_format_response(exc.response)}"
    return f"Ошибка запроса к {url}: {exc}"


async def _request(
    method: str,
    url: str,
    headers_json: str = "",
    params_json: str = "",
    json_body: str = "",
) -> str:
    """Core HTTP request handler used by all public tools."""
    # Parse headers
    headers, err = _parse_json_arg(headers_json, "headers_json")
    if err:
        return err

    # Parse query params
    params, err = _parse_json_arg(params_json, "params_json")
    if err:
        return err

    # Parse JSON body (for POST/PATCH/PUT)
    body = None
    if json_body and json_body.strip():
        body, err = _parse_json_arg(json_body, "json_body")
        if err:
            return err

    # SSRF protection: block requests to private/internal addresses
    ssrf_err = _check_ssrf(url)
    if ssrf_err:
        logger.warning("ssrf_blocked", url=url)
        return ssrf_err

    logger.info("http_request", method=method, url=url)

    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers or {},
                params=params or {},
                json=body,
            )
        logger.info(
            "http_response",
            method=method,
            url=url,
            status=response.status_code,
        )
        return _format_response(response)

    except Exception as exc:
        logger.error("http_error", method=method, url=url, error=str(exc))
        return _handle_error(exc, url)


# ── Public tools ──────────────────────────────────────────────────────────────

async def http_get(url: str, headers_json: str = "", params_json: str = "") -> str:
    """Perform an HTTP GET request.

    Args:
        url: Full URL including scheme, e.g. https://api.example.com/users
        headers_json: Optional JSON object with request headers,
                      e.g. {"Authorization": "Bearer TOKEN", "Accept": "application/json"}
        params_json: Optional JSON object with query string parameters,
                     e.g. {"page": 1, "limit": 50, "status": "active"}

    Returns:
        HTTP status line followed by the response body (JSON formatted if applicable).
        Large responses are truncated to 4000 characters.
    """
    return await _request("GET", url, headers_json=headers_json, params_json=params_json)


async def http_post(url: str, headers_json: str = "", json_body: str = "") -> str:
    """Perform an HTTP POST request with a JSON body.

    Requires confirmation if the URL path contains destructive keywords
    (delete, purge, wipe, clear, destroy, bulk-delete, etc.).

    Args:
        url: Full URL including scheme.
        headers_json: Optional JSON object with request headers.
                      Content-Type: application/json is set automatically.
        json_body: JSON object to send as the request body,
                   e.g. {"username": "john", "email": "john@example.com"}

    Returns:
        HTTP status line followed by the response body.
    """
    if _DESTRUCTIVE_URL_PATTERNS.search(url):
        description = f"HTTP POST к потенциально деструктивному endpoint\n\nURL: {url}\n\nЭто действие может быть необратимым."
        approved = await confirmation_manager.ask(description)
        if not approved:
            return "Операция отменена пользователем."
    return await _request("POST", url, headers_json=headers_json, json_body=json_body)


async def http_patch(url: str, headers_json: str = "", json_body: str = "") -> str:
    """Perform an HTTP PATCH request with a JSON body (partial update).

    Requires confirmation if the URL path contains destructive keywords.

    Args:
        url: Full URL including scheme, typically with a resource ID,
             e.g. https://api.example.com/users/42
        headers_json: Optional JSON object with request headers.
        json_body: JSON object with the fields to update,
                   e.g. {"status": "banned", "balance": 0}

    Returns:
        HTTP status line followed by the response body.
    """
    if _DESTRUCTIVE_URL_PATTERNS.search(url):
        description = f"HTTP PATCH к потенциально деструктивному endpoint\n\nURL: {url}\n\nЭто действие может быть необратимым."
        approved = await confirmation_manager.ask(description)
        if not approved:
            return "Операция отменена пользователем."
    return await _request("PATCH", url, headers_json=headers_json, json_body=json_body)


async def http_delete(url: str, headers_json: str = "") -> str:
    """Perform an HTTP DELETE request.

    Always requires explicit user confirmation before sending, since DELETE
    operations on external APIs are typically irreversible.

    Args:
        url: Full URL including scheme with the resource ID to delete,
             e.g. https://api.example.com/users/42
        headers_json: Optional JSON object with request headers.

    Returns:
        HTTP status line followed by the response body (often empty for 204).
    """
    description = f"HTTP DELETE запрос\n\nURL: {url}\n\nЭто действие может быть необратимым."
    approved = await confirmation_manager.ask(description)
    if not approved:
        return "Операция отменена пользователем."
    return await _request("DELETE", url, headers_json=headers_json)
