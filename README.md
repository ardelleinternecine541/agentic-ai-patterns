# Agentic AI Patterns

Design patterns for building autonomous AI agent systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE](https://img.shields.io/badge/IEEE-Member-blue.svg)](https://www.ieee.org/)

## Overview

A comprehensive reference implementation of design patterns and best practices for systems built on autonomous AI agents. This project provides production-ready patterns for orchestration, fallback mechanisms, memory management, tool-use integration, and multi-agent coordination.

Whether you're building a single-agent reasoning system or complex multi-agent pipelines, these patterns provide a solid foundation for reliable, maintainable AI systems.

## Table of Contents

- [Patterns Covered](#patterns-covered)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Author](#author)
- [License](#license)

## Patterns Covered

### ReAct Pattern
Implements the Reasoning + Acting loop where agents alternate between thinking (using language models for reasoning) and acting (executing tools/functions). Provides a structured approach to agent decision-making with clear observation-action-reflection cycles.

### Chain-of-Thought Orchestration
Orchestrates complex reasoning chains where intermediate reasoning steps are preserved and used to guide subsequent actions. Enables transparent, interpretable multi-step agent reasoning with full step history.

### Tool-Use Gateway
A centralized pattern for managing tool registration, input validation, execution routing, and error handling. Provides a secure, extensible interface for agents to interact with external systems and APIs.

### Memory Management
Dual-layer memory system with short-term memory for conversation context and long-term memory for persistent knowledge. Supports vector-based retrieval, conversation buffering, and semantic search capabilities.

### Fallback & Recovery
Graceful degradation and error recovery with configurable retry policies, exponential backoff, and fallback chains. Ensures agent systems remain operational even when primary execution paths fail.

### Multi-Agent Coordination
Orchestrates multiple specialized agents working towards common goals. Handles task distribution, result aggregation, and inter-agent communication with dependency management.

### Human-in-the-Loop
Integration points for human oversight and intervention in autonomous agent workflows. Supports approval gates, feedback loops, and human-assisted decision making.

## Quick Start

### Installation

```bash
pip install agentic-patterns
```

### Basic Usage

```python
from agentic_patterns.react import ReActAgent
from agentic_patterns.tool_gateway import ToolGateway

# Initialize the tool gateway
gateway = ToolGateway()

@gateway.register_tool(description="Calculate the sum of two numbers")
def add(a: int, b: int) -> int:
    return a + b

# Create a ReAct agent
agent = ReActAgent(
    model="gpt-4",
    tool_gateway=gateway,
    max_iterations=10
)

# Run the agent
result = agent.run("What is 5 plus 3?")
print(result)
```

## Architecture

```
User Input
    |
    v
+---------------------------------------+
|     ReAct Agent Loop                  |
|  [Think -> Act -> Observe -> Loop]    |
+---------------------------------------+
    |
    +---> Tool Gateway (validation, routing, error handling)
    |           |
    |           v
    |     Tool Execution
    |           |
    |           v
    |     Memory System (short/long-term)
    |
    v
Agent Output

Multi-Agent Coordination Layer
    |
    +---> Agent 1 (Planner)
    |
    +---> Agent 2 (Executor)
    |
    +---> Agent 3 (Validator)
    |
    v
Orchestrator (Aggregates Results)
    |
    v
Final Output
```

## Usage Examples

### ReAct Agent

```python
from agentic_patterns.react import ReActAgent

agent = ReActAgent(model="gpt-4", max_iterations=5)
response = agent.run("Solve: 2 + 2 * 3")
```

### Multi-Agent Pipeline

```python
from agentic_patterns.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(agents=[planner, executor, validator])
result = orchestrator.execute_task("Build a web scraper for product prices")
```

### Memory Management

```python
from agentic_patterns.memory import ShortTermMemory, LongTermMemory

short_term = ShortTermMemory(max_turns=10)
long_term = LongTermMemory(embedding_model="text-embedding-ada-002")

short_term.add("user", "What's the capital of France?")
short_term.add("assistant", "The capital of France is Paris.")

embeddings = long_term.store("Paris is the capital and largest city of France.")
```

### Fallback Chain

```python
from agentic_patterns.fallback import FallbackChain

chain = FallbackChain()
chain.add_step(primary_api_call, max_retries=3)
chain.add_step(fallback_api_call, max_retries=2)
chain.add_step(cached_response, max_retries=1)

result = chain.execute()
```

## API Reference

### ReActAgent

```python
class ReActAgent:
    def __init__(self, model: str, tool_gateway: ToolGateway, max_iterations: int = 10)
    def think() -> str
    def act(action: str) -> Any
    def observe(result: Any) -> None
    def run(task: str) -> str
```

### ToolGateway

```python
class ToolGateway:
    def register_tool(func: Callable, description: str) -> None
    def validate_input(tool_name: str, inputs: Dict) -> bool
    def execute_tool(tool_name: str, inputs: Dict) -> Any
```

### Memory Classes

```python
class ShortTermMemory:
    def add(role: str, content: str) -> None
    def get_context(num_turns: int) -> List[Dict]
    def clear() -> None

class LongTermMemory:
    def store(content: str) -> List[float]
    def retrieve(query: str, top_k: int) -> List[str]
```

## Author

**Camilo Girardelli**
- IEEE Senior Member
- Senior Software Architect
- CTO, Girardelli Tecnologia
- [Visit Website](https://girardelli.tech)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright 2026 Camilo Girardelli / Girardelli Tecnologia
