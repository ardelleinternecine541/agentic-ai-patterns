"""
Agentic AI Patterns - Design patterns for building autonomous AI agent systems.

This package provides production-ready implementations of design patterns
for AI agent orchestration, reasoning, memory management, and tool-use.
"""

__version__ = "0.1.0"
__author__ = "Camilo Girardelli"
__license__ = "MIT"

from agentic_patterns.react import ReActAgent
from agentic_patterns.tool_gateway import ToolGateway
from agentic_patterns.memory import ShortTermMemory, LongTermMemory
from agentic_patterns.fallback import FallbackChain
from agentic_patterns.orchestrator import AgentOrchestrator

__all__ = [
    "ReActAgent",
    "ToolGateway",
    "ShortTermMemory",
    "LongTermMemory",
    "FallbackChain",
    "AgentOrchestrator",
]
