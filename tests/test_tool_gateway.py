"""
Unit tests for Tool Gateway pattern.

Tests tool registration, validation, execution, and error handling.
"""

import pytest
from agentic_patterns.tool_gateway import (
    ToolGateway,
    ToolNotFoundError,
    ToolValidationError,
    ToolExecutionError,
)


class TestToolGateway:
    """Test suite for ToolGateway class."""

    @pytest.fixture
    def gateway(self):
        """Create a tool gateway for testing."""
        return ToolGateway()

    @pytest.fixture
    def gateway_with_tools(self):
        """Create a gateway with registered tools."""
        gateway = ToolGateway()

        @gateway.register_tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        @gateway.register_tool(description="Subtract two numbers")
        def subtract(a: int, b: int) -> int:
            return a - b

        return gateway

    def test_gateway_initialization(self, gateway):
        """Test gateway initialization."""
        assert len(gateway.tools) == 0
        assert len(gateway.execution_history) == 0

    def test_tool_registration(self, gateway):
        """Test tool registration."""

        @gateway.register_tool(description="Test tool")
        def test_tool(x: int) -> int:
            return x * 2

        assert "test_tool" in gateway.tools
        assert gateway.tools["test_tool"].description == "Test tool"

    def test_list_tools(self, gateway_with_tools):
        """Test listing registered tools."""
        tools = gateway_with_tools.list_tools()

        assert len(tools) == 2
        assert any(t["name"] == "add" for t in tools)
        assert any(t["name"] == "subtract" for t in tools)

    def test_get_tool_description(self, gateway_with_tools):
        """Test getting tool description."""
        desc = gateway_with_tools.get_tool_description("add")
        assert desc == "Add two numbers"

    def test_get_nonexistent_tool_description(self, gateway):
        """Test getting description of non-existent tool."""
        with pytest.raises(ToolNotFoundError):
            gateway.get_tool_description("nonexistent")

    def test_execute_tool_success(self, gateway_with_tools):
        """Test successful tool execution."""
        result = gateway_with_tools.execute_tool("add", {"a": 5, "b": 3})
        assert result == 8

    def test_execute_tool_nonexistent(self, gateway):
        """Test executing non-existent tool."""
        with pytest.raises(ToolNotFoundError):
            gateway.execute_tool("nonexistent", {})

    def test_tool_input_validation(self, gateway):
        """Test tool input validation."""

        @gateway.register_tool(
            description="Test validation",
            input_schema={"required": ["required_field"]},
        )
        def validated_tool(required_field: str) -> str:
            return required_field

        # Should pass validation
        assert gateway.validate_input("validated_tool", {"required_field": "value"})

        # Should fail validation
        with pytest.raises(ToolValidationError):
            gateway.validate_tool("validated_tool", {})

    def test_tool_execution_with_wrong_args(self, gateway_with_tools):
        """Test tool execution with wrong arguments."""
        with pytest.raises(ToolExecutionError):
            gateway_with_tools.execute_tool("add", {"a": 5})  # Missing 'b'

    def test_execution_history(self, gateway_with_tools):
        """Test execution history tracking."""
        gateway_with_tools.execute_tool("add", {"a": 10, "b": 20})
        gateway_with_tools.execute_tool("subtract", {"a": 10, "b": 5})

        history = gateway_with_tools.get_execution_history()
        assert len(history) == 2
        assert history[0]["tool_name"] == "add"
        assert history[0]["result"] == 30
        assert history[1]["tool_name"] == "subtract"
        assert history[1]["result"] == 5

    def test_clear_execution_history(self, gateway_with_tools):
        """Test clearing execution history."""
        gateway_with_tools.execute_tool("add", {"a": 1, "b": 1})
        assert len(gateway_with_tools.get_execution_history()) == 1

        gateway_with_tools.clear_execution_history()
        assert len(gateway_with_tools.get_execution_history()) == 0

    def test_tool_with_multiple_executions(self, gateway):
        """Test tool executed multiple times."""

        @gateway.register_tool(description="Counter tool")
        def counter(n: int) -> int:
            return n + 1

        result1 = gateway.execute_tool("counter", {"n": 5})
        result2 = gateway.execute_tool("counter", {"n": 10})

        assert result1 == 6
        assert result2 == 11
        assert len(gateway.get_execution_history()) == 2

    def test_tool_raising_exception(self, gateway):
        """Test tool that raises an exception."""

        @gateway.register_tool(description="Failing tool")
        def failing_tool(x: int) -> int:
            raise ValueError("Test error")

        with pytest.raises(ToolExecutionError):
            gateway.execute_tool("failing_tool", {"x": 5})

    def test_tool_validation_skipped_without_schema(self, gateway):
        """Test that validation is skipped when no schema is defined."""

        @gateway.register_tool(description="Tool without schema")
        def unvalidated_tool(**kwargs) -> dict:
            return kwargs

        # Should pass validation without checking
        assert gateway.validate_input("unvalidated_tool", {})
        assert gateway.validate_input("unvalidated_tool", {"any": "thing"})


class TestToolGatewayExecution:
    """Test tool execution behavior."""

    def test_execution_result_recorded(self):
        """Test that execution results are properly recorded."""
        gateway = ToolGateway()

        @gateway.register_tool(description="Simple return")
        def return_value(v: str) -> str:
            return v.upper()

        result = gateway.execute_tool("return_value", {"v": "hello"})
        history = gateway.get_execution_history()

        assert result == "HELLO"
        assert history[-1]["result"] == "HELLO"
        assert history[-1]["status"] == "success"

    def test_execution_inputs_recorded(self):
        """Test that execution inputs are recorded."""
        gateway = ToolGateway()

        @gateway.register_tool(description="Echo tool")
        def echo(msg: str) -> str:
            return msg

        inputs = {"msg": "test message"}
        gateway.execute_tool("echo", inputs)

        history = gateway.get_execution_history()
        assert history[-1]["inputs"] == inputs
