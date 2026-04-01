"""Core ReAct reasoning loop.

Drives Claude through a native tool_use cycle:
  Thought → Tool call → Observation → Thought → ... → Final answer

Uses anthropic.AsyncAnthropic directly for full control over tool_use blocks.
Kerb is used only for Tool/AgentStep/AgentResult data structures.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable

import anthropic
import structlog
from kerb.agent.core import AgentStatus
from kerb.agent.execution import AgentResult, AgentStep

import config
from agent.tool_registry import ToolRegistry

logger = structlog.get_logger()

# Async function called with each AgentStep as it is produced
StepCallback = Callable[[AgentStep], Awaitable[None]]


def _trim_history(history: list[dict], max_pairs: int) -> list[dict]:
    """Return history trimmed to the last max_pairs user-initiated turns.

    Cuts only at plain-text user message boundaries so we never split a
    tool_use / tool_result pair (which would cause an Anthropic API 400).
    """
    # Identify indices of plain user text messages (conversation turn starts)
    turn_indices = [
        i for i, msg in enumerate(history)
        if msg["role"] == "user" and isinstance(msg.get("content"), str)
    ]
    if len(turn_indices) <= max_pairs:
        return history
    keep_from = turn_indices[-max_pairs]
    trimmed = len(turn_indices) - max_pairs
    logger.debug("history_trimmed", dropped_turns=trimmed, kept=max_pairs)
    return history[keep_from:]


class AdminAgentLoop:
    """Native Claude tool_use ReAct loop.

    Each call to run() handles one full user turn:
    - Appends user message to history
    - Iterates: call Claude → emit step → execute tool → send result back
    - Stops when Claude returns a final text response (no tool_use blocks)
    - Appends final assistant message to history
    """

    def __init__(
        self,
        registry: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 15,
    ) -> None:
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    async def run(
        self,
        user_message: str,
        history: list[dict],
        step_callback: StepCallback,
    ) -> AgentResult:
        """Execute one full user turn and return the result.

        Mutates `history` in place so AdminAgent can persist dialog across turns.

        Args:
            user_message: Latest message from the user.
            history: Conversation history in Anthropic message format (mutated).
            step_callback: Async callback invoked with each intermediate AgentStep.

        Returns:
            AgentResult with final output text and all steps.
        """
        start_time = time.monotonic()
        steps: list[AgentStep] = []

        history.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            logger.debug("loop_iteration", iteration=iteration, history_len=len(history))

            response = await self._call_claude(history)

            # ── Parse response content ────────────────────────────────────────
            text_parts: list[str] = []
            tool_uses: list = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    text_parts.append(block.text.strip())
                elif block.type == "tool_use":
                    tool_uses.append(block)

            thought = "\n\n".join(text_parts)

            # ── No tool calls → final answer ──────────────────────────────────
            if not tool_uses:
                history.append({"role": "assistant", "content": thought})
                steps.append(AgentStep(step_number=len(steps) + 1, thought=thought))
                logger.info("loop_finished", steps=len(steps), iterations=iteration + 1)
                break

            # ── Execute each tool call ────────────────────────────────────────
            tool_results: list[dict] = []
            first_in_round = True

            for tool_use in tool_uses:
                step = AgentStep(
                    step_number=len(steps) + 1,
                    # Show thought only on the first tool of this Claude response
                    thought=thought if first_in_round else "",
                    action=tool_use.name,
                    action_input=tool_use.input,
                )
                first_in_round = False

                # 1. Notify user BEFORE execution: thought + "🔧 Вызываю: ..."
                await step_callback(step)

                # 2. Execute tool
                observation = await self._execute_tool(tool_use.name, tool_use.input)
                step.observation = observation
                steps.append(step)

                logger.debug(
                    "tool_executed",
                    tool=tool_use.name,
                    obs_len=len(observation),
                )

                # 3. Notify user AFTER execution: "✅ Результат: ..."
                await step_callback(step)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": observation,
                    }
                )

            # Append full assistant turn + tool results back into history
            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})

        else:
            # Max iterations reached without final answer
            logger.warning("loop_max_iterations_reached", max=self.max_iterations)
            fallback = "Достигнут лимит итераций. Подведу итог на основе полученных данных."
            history.append({"role": "assistant", "content": fallback})
            steps.append(AgentStep(step_number=len(steps) + 1, thought=fallback))

        final_output = steps[-1].thought if steps else ""
        return AgentResult(
            output=final_output,
            steps=steps,
            status=AgentStatus.COMPLETED,
            total_time=time.monotonic() - start_time,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _call_claude(self, messages: list[dict]):
        """Call Claude API with current messages and registered tools."""
        kwargs: dict = {
            "model": config.LLM_MODEL,
            "system": self.system_prompt,
            "messages": _trim_history(messages, config.MAX_HISTORY_PAIRS),
            "max_tokens": 4096,
        }
        tools = self.registry.get_anthropic_tools()
        if tools:
            kwargs["tools"] = tools

        return await self._client.messages.create(**kwargs)

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Find and execute a tool by name. Returns string observation.

        Calls tool.func directly (bypasses kerb's execute_async which has a
        kwargs bug with run_in_executor). Async tools are awaited; sync tools
        are run in the default thread-pool executor to avoid blocking the loop.

        Safety gate for custom file-loaded tools: if the function name contains
        a destructive keyword (delete, purge, drop, etc.) the user is asked to
        confirm before execution — same flow as for SQL and HTTP DELETE.
        MCP tools handle their own safety gate inside _wrapper.
        """
        import asyncio
        import functools

        from agent.confirmation import confirmation_manager
        from agent.safety import is_dangerous_name

        tool = self.registry.get_by_name(name)
        if tool is None:
            return f"Ошибка: инструмент '{name}' не найден в реестре."

        # Safety check for custom tools loaded from agent_tools/ files
        if self.registry.is_file_loaded(name) and is_dangerous_name(name):
            args_preview = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "—"
            description = (
                f"Вызов кастомного инструмента\n\n"
                f"Функция: {name}\n"
                f"Аргументы: {args_preview}\n\n"
                f"Причина проверки: имя содержит признаки деструктивной операции."
            )
            approved = await confirmation_manager.ask(description)
            if not approved:
                return "Операция отменена пользователем."

        try:
            if asyncio.iscoroutinefunction(tool.func):
                output = await tool.func(**args)
            else:
                fn = functools.partial(tool.func, **args)
                output = await asyncio.get_running_loop().run_in_executor(None, fn)
            return str(output) if output is not None else "(нет результата)"
        except Exception as exc:
            logger.error("tool_execution_error", tool=name, error=str(exc))
            return f"Ошибка выполнения '{name}': {exc}"
