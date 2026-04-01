"""Shared safety utilities — dangerous tool name detection.

Used by mcp_client (MCP tool wrappers) and AdminAgentLoop (custom file tools)
to gate destructive-sounding calls behind the confirmation system — without any
LLM involvement.

The check is intentionally conservative: it fires on keyword matches within
snake_case tool names (e.g. ``delete_user``, ``bulk_delete``, ``reset_all``).
False positives (a harmless tool named ``clear_cache``) produce a confirmation
prompt — which is a much better outcome than silently executing a destructive op.
"""
from __future__ import annotations

import re

# Matches a keyword as a complete snake_case segment (not a substring of another word).
# Anchors: start-of-string or preceding underscore; end-of-string or following underscore.
_DANGEROUS_NAME_RE = re.compile(
    r"(?:^|(?<=_))"
    r"(delete|remove|purge|wipe|drop|destroy|erase|truncate|nuke|scrub|"
    r"kill|terminate|ban|disable|revoke|deactivate|flush|reset|clear|"
    r"uninstall|unregister|dismiss|reject|expire|invalidate)"
    r"(?=_|$)",
    re.IGNORECASE,
)


def is_dangerous_name(name: str) -> bool:
    """Return True if *name* contains a destructive keyword as a snake_case segment."""
    return bool(_DANGEROUS_NAME_RE.search(name))
