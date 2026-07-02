"""Trace collection and Rich CLI viewer."""

from novel_agent.trace.collector import TraceCollector
from novel_agent.trace.viewer import list_traces, show_trace

__all__ = ["TraceCollector", "list_traces", "show_trace"]
