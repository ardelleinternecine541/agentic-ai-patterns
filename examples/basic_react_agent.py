"""
Basic ReAct Agent Example.

Demonstrates the usage of the ReAct agent pattern with a simple
question-answering task and tool integration.
"""

from agentic_patterns.react import ReActAgent
from agentic_patterns.tool_gateway import ToolGateway


def create_calculator_tools(gateway: ToolGateway) -> None:
    """Register mathematical tools to the gateway."""

    @gateway.register_tool(description="Add two numbers")
    def add(a: float, b: float) -> float:
        return a + b

    @gateway.register_tool(description="Subtract two numbers")
    def subtract(a: float, b: float) -> float:
        return a - b

    @gateway.register_tool(description="Multiply two numbers")
    def multiply(a: float, b: float) -> float:
        return a * b

    @gateway.register_tool(description="Divide two numbers")
    def divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


def main():
    """Run basic ReAct agent example."""
    print("Agentic AI Patterns - Basic ReAct Agent Example\n")

    # Initialize tool gateway
    gateway = ToolGateway()
    create_calculator_tools(gateway)

    # Print available tools
    print("Available tools:")
    for tool in gateway.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")
    print()

    # Create ReAct agent
    agent = ReActAgent(
        model="gpt-4",
        tool_gateway=gateway,
        max_iterations=5,
        temperature=0.7,
    )

    # Run agent on a task
    task = "What is 10 + 5 multiplied by 2?"
    print(f"Task: {task}")
    print("-" * 60)

    try:
        result = agent.run(task)
        print(f"\nFinal Answer: {result}\n")

        # Print reasoning history
        history = agent.get_history()
        print("Agent Reasoning History:")
        print(f"  Iterations: {history['iterations']}")
        print(f"  Thoughts generated: {len(history['thoughts'])}")
        print(f"  Actions taken: {len(history['actions'])}")
        print(f"  Observations: {len(history['observations'])}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
