from __future__ import annotations

import importlib.util
import inspect
import sys
import structlog
from pathlib import Path
from kerb.agent.tools import Tool, create_tool

logger = structlog.get_logger()


class ToolRegistry:
    """Registry of all tools available to the agent.

    Holds built-in tools (meta, db, api) and dynamically loaded tools
    from agent_tools/.
    """

    def __init__(self) -> None:
        self.tools: list[Tool] = []
        self._names: set[str] = set()
        # Names of tools loaded from agent_tools/ files (used for safety checks)
        self._file_loaded: set[str] = set()

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry (skips duplicates by name)."""
        if tool.name in self._names:
            # Replace existing tool with updated version
            self.tools = [t for t in self.tools if t.name != tool.name]
            self._names.discard(tool.name)
        self.tools.append(tool)
        self._names.add(tool.name)
        logger.debug("tool_registered", name=tool.name)

    def register_function(self, func) -> Tool:
        """Wrap a plain function in a Tool and register it."""
        tool = create_tool(name=func.__name__, func=func)
        self.register(tool)
        return tool

    def load_from_file(self, path: str | Path) -> list[Tool]:
        """Dynamically import a Python file and register all public callables.

        Removes any prior cached version of the module from sys.modules so
        that write_tool() hot-reloads always pick up the latest code on disk.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("tool_file_not_found", path=str(path))
            return []

        # Evict stale module cache so reimport reads fresh code from disk
        sys.modules.pop(path.stem, None)

        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        loaded: list[Tool] = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_"):
                continue
            if obj.__module__ != module.__name__:
                continue  # skip re-exported functions
            tool = create_tool(name=name, func=obj)
            self.register(tool)
            self._file_loaded.add(name)
            loaded.append(tool)

        logger.info("tools_loaded_from_file", path=str(path), count=len(loaded))
        return loaded

    def reload_all(self, tools_dir: str | Path) -> None:
        """Load (or reload) all .py files from agent_tools/ directory."""
        tools_dir = Path(tools_dir)
        if not tools_dir.exists():
            return
        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self.load_from_file(py_file)

    def get_anthropic_tools(self) -> list[dict]:
        """Return all tools in Anthropic API tool format."""
        return [t.to_anthropic_tool() for t in self.tools]

    def get_by_name(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def is_file_loaded(self, name: str) -> bool:
        """Return True if *name* was loaded from an agent_tools/ file (not built-in)."""
        return name in self._file_loaded

    def list_names(self) -> list[str]:
        return sorted(self._names)
