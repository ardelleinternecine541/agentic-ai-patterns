"""
Unit tests for ReAct agent pattern.

Tests the core functionality of the ReAct agent including
thinking, acting, observing, and the main run loop.
"""

import pytest
from agentic_patterns.react import ReActAgent, AgentState
from agentic_patterns.tool_gateway import ToolGateway


class TestReActAgent:
    """Test suite for ReActAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a ReAct agent for testing."""
        return ReActAgent(
            model="gpt-4",
            max_iterations=5,
            temperature=0.7,
        )

    @pytest.fixture
    def agent_with_tools(self):
        """Create a ReAct agent with tools."""
        gateway = ToolGateway()

        @gateway.register_tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        agent = ReActAgent(
            model="gpt-4",
            tool_gateway=gateway,
            max_iterations=5,
        )
        return agent, gateway

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.model == "gpt-4"
        assert agent.max_iterations == 5
        assert agent.temperature == 0.7
        assert agent.state == AgentState.THINKING

    def test_think(self, agent):
        """Test the think method."""
        agent.task = "Test task"
        thought = agent.think()

        assert isinstance(thought, str)
        assert len(agent.thoughts) == 1
        assert agent.thoughts[0].content == thought

    def test_observe(self, agent):
        """Test the observe method."""
        observation_content = "Test observation"
        agent.observe(observation_content)

        assert len(agent.observations) == 1
        assert agent.observations[0].content == observation_content

    def test_run_basic(self, agent):
        """Test basic run execution."""
        task = "Simple task"
        result = agent.run(task)

        assert isinstance(result, str)
        assert agent.state == AgentState.DONE
        assert agent.task == task

    def test_run_with_tools(self, agent_with_tools):
        """Test run with tool gateway."""
        agent, gateway = agent_with_tools
        task = "Calculate 5 + 3"
        result = agent.run(task)

        assert isinstance(result, str)
        assert agent.state == AgentState.DONE

    def test_max_iterations(self, agent):
        """Test that agent respects max iterations."""
        agent.max_iterations = 2
        result = agent.run("Test task")

        assert agent.iteration <= agent.max_iterations

    def test_get_history(self, agent):
        """Test history retrieval."""
        agent.task = "Test task"
        agent.think()
        agent.observe("Test observation")

        history = agent.get_history()

        assert "task" in history
        assert "iterations" in history
        assert "thoughts" in history
        assert "observations" in history
        assert len(history["thoughts"]) == 1
        assert len(history["observations"]) == 1

    def test_should_finish_detection(self, agent):
        """Test finish condition detection."""
        agent.task = "Test"
        assert agent._should_finish("Therefore the answer is 42") == True
        assert agent._should_finish("Let me think about this") == False
        assert agent._should_finish("The conclusion is obvious") == True

    def test_parse_action(self, agent):
        """Test action parsing."""
        action_desc = "calculate: 2 + 2"
        action = agent._parse_action(action_desc)

        assert action.tool == "calculate"
        assert "description" in action.input


class TestReActAgentWithToolGateway:
    """Test ReActAgent integration with ToolGateway."""

    def test_agent_tool_execution(self):
        """Test agent executing registered tools."""
        gateway = ToolGateway()

        @gateway.register_tool(description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        agent = ReActAgent(
            model="gpt-4",
            tool_gateway=gateway,
            max_iterations=3,
        )

        # Test tool execution
        result = agent.act("multiply: 3 * 4")
        assert result is not None

    def test_agent_without_tools(self):
        """Test agent without tool gateway."""
        agent = ReActAgent(
            model="gpt-4",
            tool_gateway=None,
            max_iterations=2,
        )

        agent.task = "Simple task"
        result = agent.act("calculate: 5 + 3")

        # Should return placeholder message
        assert isinstance(result, str)
        assert "Would execute" in result or "calculate" in result


class TestReActAgentStates:
    """Test state transitions in ReActAgent."""

    def test_state_transitions(self):
        """Test proper state transitions."""
        agent = ReActAgent(max_iterations=1)

        assert agent.state == AgentState.THINKING

        agent.state = AgentState.ACTING
        assert agent.state == AgentState.ACTING

        agent.state = AgentState.OBSERVING
        assert agent.state == AgentState.OBSERVING

        agent.state = AgentState.DONE
        assert agent.state == AgentState.DONE

    def test_state_after_run(self):
        """Test state after complete run."""
        agent = ReActAgent(max_iterations=2)
        agent.run("Test task")

        assert agent.state == AgentState.DONE
