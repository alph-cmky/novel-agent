"""BaseAgent — model calling, tool execution loop, trace recording."""

import json
import os
import time
import warnings
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from novel_agent.config import REASONING_EFFORT
from novel_agent.observability.langfuse import get_handler as _get_lf_handler
from novel_agent.tools.base import BaseTool, ToolResult


class AgentConfig:
    """Configuration for an agent instance.

    Validates inputs early so misconfiguration is caught at startup,
    not deep inside an LLM call.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.8,
        is_reasoning: bool = False,
    ):
        self.model = model or os.getenv("BUDGET_MODEL", "deepseek-chat")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Validate types
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"model must be a non-empty string, got: {self.model!r}")
        if not isinstance(self.api_key, str):
            raise ValueError(f"api_key must be a string, got: {type(self.api_key).__name__}")
        if not isinstance(self.base_url, str):
            raise ValueError(f"base_url must be a string, got: {type(self.base_url).__name__}")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive int, got: {max_tokens}")
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ValueError(f"temperature must be 0-2, got: {temperature}")
        if not isinstance(is_reasoning, bool):
            raise ValueError(f"is_reasoning must be a bool, got: {type(is_reasoning).__name__}")

        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.is_reasoning = is_reasoning

        # Warn if API key is empty (but don't crash — user may fix it later)
        if not self.api_key:
            import warnings

            warnings.warn(
                "OPENAI_API_KEY is not set. LLM calls will fail until configured.",
                RuntimeWarning,
                stacklevel=2,
            )


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


def _to_langchain_messages(messages: list[dict]) -> list[BaseMessage]:
    """Convert OpenAI-style message dicts to LangChain BaseMessage list."""
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            aim = AIMessage(content=content)
            tcs = m.get("tool_calls", [])
            if tcs:
                aim.tool_calls = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": (
                            json.loads(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], str)
                            else tc["function"]["arguments"]
                        ),
                    }
                    for tc in tcs
                ]
            result.append(aim)
        elif role == "tool":
            result.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
        else:
            result.append(HumanMessage(content=content))
    return result


def build_chat_model(config: AgentConfig) -> ChatOpenAI:
    """构造 ChatOpenAI；对 reasoning 模型注入 reasoning_effort 压低推理预算。

    reasoning 模型（如 StepFun）在长 thinking 阶段会 >120s 不吐任何 chunk，
    langchain-openai 默认 stream_chunk_timeout=120 会误判为连接卡死并抛
    StreamChunkTimeoutError，故对 reasoning 模型禁用流式超时。

    是否 reasoning 由 AgentConfig.is_reasoning 声明（env: QUALITY_IS_REASONING
    / BUDGET_IS_REASONING），不在代码里写死具体模型名。
    """
    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    if config.is_reasoning:
        kwargs["reasoning_effort"] = REASONING_EFFORT
        kwargs["stream_chunk_timeout"] = None
    return ChatOpenAI(**kwargs)


def _warn_undeclared_reasoning(config: AgentConfig, reasoning_content: str) -> None:
    """reasoning_content 已返回但 is_reasoning 未声明时告警，提示补 env 声明。"""
    if reasoning_content and not config.is_reasoning:
        warnings.warn(
            "检测到 reasoning_content 输出但 is_reasoning=False。请在 env 声明 "
            "QUALITY_IS_REASONING / BUDGET_IS_REASONING=true 以正确注入 "
            "reasoning_effort 并禁用 stream_chunk_timeout。",
            RuntimeWarning,
            stacklevel=2,
        )


class BaseAgent:
    """Base agent with model calling, tool execution, and trace recording.

    Instance-level cost counters (model_calls / tool_call_counts) accumulate
    across every LLM request and tool execution on this agent instance —
    callers read them after a run to observe per-node LLM cost.
    """

    name: str = "base"

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._tools: dict[str, BaseTool] = {}
        self._latest_trace: TraceStep | None = None
        self.model_calls = 0
        self.tool_call_counts: dict[str, int] = {}

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    @property
    def latest_trace(self) -> TraceStep | None:
        """Most recent trace step from the last model call."""
        return self._latest_trace

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool

    async def call_model(
        self,
        messages: list[dict],
        tools: list[BaseTool] | None = None,
        action: str = "model_call",
    ) -> AIMessage:
        """Call the LLM with optional tool definitions. Returns AIMessage."""
        tool_list = tools or self.tools
        lc_messages = _to_langchain_messages(messages)

        t0 = time.monotonic()
        trace = TraceStep(agent=self.name, action=action)
        trace.model = self.config.model

        model = build_chat_model(self.config)
        bound = (
            model.bind_tools([t.get_schema()["function"] for t in tool_list])
            if tool_list
            else model
        )

        config: dict[str, Any] = {}
        lf_handler = _get_lf_handler()
        if lf_handler:
            config["callbacks"] = [lf_handler]

        # step-3.7-flash 等模型偶发返回空 content（无 tool_calls），重试最多 3 次
        response: AIMessage | None = None
        for _ in range(3):
            self.model_calls += 1
            response = await bound.ainvoke(lc_messages, config=config)
            if response.content or getattr(response, "tool_calls", None):
                break
        assert response is not None  # 循环至少执行一次
        _warn_undeclared_reasoning(self.config, getattr(response, "reasoning_content", "") or "")
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        usage = getattr(response, "usage_metadata", {}) or {}
        trace.input_tokens = usage.get("input_tokens", 0)
        trace.output_tokens = usage.get("output_tokens", 0)
        trace.duration_ms = elapsed_ms
        self._latest_trace = trace

        return response

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

        lc_messages = _to_langchain_messages(messages)
        total_input = 0
        total_output = 0
        reasoning_chars = 0
        warned = False

        model = build_chat_model(self.config)
        config: dict[str, Any] = {}
        lf_handler = _get_lf_handler()
        if lf_handler:
            config["callbacks"] = [lf_handler]

        self.model_calls += 1
        async for chunk in model.astream(lc_messages, config=config):
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                total_input = usage.get("input_tokens", total_input)
                total_output = usage.get("output_tokens", total_output)

            content = getattr(chunk, "content", "") or ""
            rc = getattr(chunk, "reasoning_content", "") or ""
            if rc:
                reasoning_chars += len(rc)
                if not warned:
                    _warn_undeclared_reasoning(self.config, rc)
                    warned = True
            if content:
                yield content

        if reasoning_chars > 0 and total_output == 0:
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
            if isinstance(tc, dict):
                tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                tool_args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                tool_id = tc.get("id", "")
            else:
                tool_name = getattr(tc, "name", "")
                tool_args = getattr(tc, "args", {})
                tool_id = getattr(tc, "id", "")

            tool = self._tools.get(tool_name)
            if tool is None:
                results.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    }
                )
                continue

            self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            try:
                result: ToolResult = await tool.execute(**tool_args)
            except Exception as exc:
                result = ToolResult(success=False, error=str(exc))

            results.append(
                {
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "content": (
                        json.dumps(result.data, ensure_ascii=False)
                        if result.success
                        else json.dumps({"error": result.error})
                    ),
                }
            )

            if trace:
                trace.tool_calls.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "success": result.success,
                    }
                )

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
            response = await self.call_model(messages, action=action)
            if self._latest_trace:
                trace.input_tokens += self._latest_trace.input_tokens
                trace.output_tokens += self._latest_trace.output_tokens

            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                tool_results = await self.execute_tool_calls(tool_calls, trace)
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": (
                                        json.dumps(tc.get("args", {}))
                                        if isinstance(tc.get("args"), dict)
                                        else tc.get("args", "")
                                    ),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                messages.extend(tool_results)
                continue

            content = response.content or ""
            if content:
                trace.duration_ms = int((time.monotonic() - t0) * 1000)
                self._latest_trace = trace
                return content, trace

            break

        self._latest_trace = trace
        return "", trace
