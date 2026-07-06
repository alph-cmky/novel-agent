"""BaseAgent — model calling, tool execution loop, trace recording."""

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI

from novel_agent.tools.base import BaseTool, ToolResult


class AgentConfig:
    """Configuration for an agent instance."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.8,
    ):
        self.model = model or os.getenv("BUDGET_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_tokens = max_tokens
        self.temperature = temperature


class TraceStep:
    """Record of a single agent action."""

    def __init__(self, agent: str, action: str):
        self.step_id = str(uuid4())
        self.agent = agent
        self.action = action
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls: list[dict] = []
        self.duration_ms = 0
        self.model = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "agent": self.agent,
            "action": self.action,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "duration_ms": self.duration_ms,
            "model": self.model,
        }


class BaseAgent:
    """Base agent with model calling, tool execution, and trace recording."""

    name: str = "base"

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self._tools: dict[str, BaseTool] = {}
        self._latest_trace: TraceStep | None = None

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool

    async def call_model(
        self,
        messages: list[dict],
        tools: list[BaseTool] | None = None,
        action: str = "model_call",
    ) -> Any:
        """Call the LLM with optional tool definitions. Returns the message."""
        tool_list = tools or self.tools
        tool_schemas = [t.get_schema() for t in tool_list] if tool_list else None

        t0 = time.monotonic()
        trace = TraceStep(agent=self.name, action=action)
        trace.model = self.config.model

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = await self._client.chat.completions.create(**kwargs)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        msg = response.choices[0].message
        trace.input_tokens = response.usage.prompt_tokens if response.usage else 0
        trace.output_tokens = response.usage.completion_tokens if response.usage else 0
        trace.duration_ms = elapsed_ms
        self._latest_trace = trace

        return msg

    async def call_model_stream(
        self,
        messages: list[dict],
        action: str = "model_call",
    ) -> AsyncIterator[str]:
        """Call the LLM with streaming enabled. Yields content chunks.

        Handles reasoning models (e.g., step-3.7-flash) that output
        content in delta.content only after reasoning phase completes.
        Also captures reasoning_content for token tracking but does NOT
        yield it to keep the output clean for end users.

        Note: does not support tool calling. Use call_model() for tool-enabled calls.
        """
        t0 = time.monotonic()
        trace = TraceStep(agent=self.name, action=action)
        trace.model = self.config.model

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        stream = await self._client.chat.completions.create(**kwargs)
        total_input = 0
        total_output = 0
        reasoning_chars = 0
        async for chunk in stream:
            if chunk.usage:
                total_input = chunk.usage.prompt_tokens or 0
                total_output = chunk.usage.completion_tokens or 0
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                rc = getattr(delta, "reasoning_content", "") or ""
                if rc:
                    reasoning_chars += len(rc)
                if delta.content:
                    yield delta.content

        if reasoning_chars > 0 and total_output == 0:
            # Reasoning model used all tokens for thinking — estimate output
            total_output = reasoning_chars

        trace.input_tokens = total_input
        trace.output_tokens = total_output
        trace.duration_ms = int((time.monotonic() - t0) * 1000)
        self._latest_trace = trace

    async def execute_tool_calls(
        self,
        tool_calls: list[Any],
        trace: TraceStep | None = None,
    ) -> list[dict]:
        """Execute tool calls from a model response and return results."""
        results = []
        for tc in tool_calls:
            tool_name = tc.function.name
            tool = self._tools.get(tool_name)
            if tool is None:
                results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                })
                continue

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result: ToolResult = await tool.execute(**args)
            results.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "content": json.dumps(result.data, ensure_ascii=False),
            })

            if trace:
                trace.tool_calls.append({
                    "tool": tool_name,
                    "args": args,
                    "success": result.success,
                })

        return results

    async def run_with_tools(
        self,
        messages: list[dict],
        max_rounds: int = 5,
        action: str = "agent_run",
    ) -> tuple[str, TraceStep]:
        """Run the agent with tool-calling loop. Returns (final_text, trace)."""
        t0 = time.monotonic()
        trace = TraceStep(agent=self.name, action=action)
        trace.model = self.config.model

        for _ in range(max_rounds):
            msg = await self.call_model(messages, action=action)
            if self._latest_trace:
                trace.input_tokens += self._latest_trace.input_tokens
                trace.output_tokens += self._latest_trace.output_tokens

            # Execute tool_calls first — model may return both
            # content and tool_calls; tools must not be dropped.
            if msg.tool_calls:
                tool_results = await self.execute_tool_calls(msg.tool_calls, trace)
                messages.append({"role": "assistant", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }} for tc in msg.tool_calls
                ]})
                messages.extend(tool_results)
                if msg.content:
                    # Accumulate content and continue — model may produce
                    # more after tool results are fed back.
                    continue
                continue

            if msg.content:
                trace.duration_ms = int((time.monotonic() - t0) * 1000)
                self._latest_trace = trace
                return msg.content, trace

            # No content and no tool calls — shouldn't happen
            break

        self._latest_trace = trace
        return "", trace
