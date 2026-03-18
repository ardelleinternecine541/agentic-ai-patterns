"""
Tool-Use Gateway Pattern Implementation.

Provides a centralized, secure interface for tool registration, input validation,
execution routing, and error handling. Allows agents to safely interact with
external systems and APIs through a controlled gateway.
"""

from typing import Any, Callable, Dict, Optional, List
from functools import wraps
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ToolMetadata:
    """Metadata about a registered tool."""
    name: str
    func: Callable
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class ToolGatewayError(Exception):
    """Base exception for tool gateway errors."""
    pass


class ToolNotFoundError(ToolGatewayError):
    """Raised when a requested tool is not registered."""
    pass


class ToolValidationError(ToolGatewayError):
    """Raised when tool input validation fails."""
    pass


class ToolExecutionError(ToolGatewayError):
    """Raised when tool execution fails."""
    pass


class ToolGateway:
    """
    Centralized gateway for tool registration and execution.

    Provides secure tool management with validation, error handling,
    and execution routing for agent systems.
    """

    def __init__(self):
        """Initialize the tool gateway."""
        self.tools: Dict[str, ToolMetadata] = {}
        self.execution_history: List[Dict[str, Any]] = []
        logger.info("tool_gateway_initialized")

    def register_tool(
        self,
        description: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator for registering a tool.

        Args:
            description: Human-readable description of the tool
            input_schema: Optional JSON schema for input validation
            output_schema: Optional JSON schema for output validation

        Returns:
            Decorator function for tool registration
        """

        def decorator(func: Callable) -> Callable:
            tool_name = func.__name__
            metadata = ToolMetadata(
                name=tool_name,
                func=func,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
            )
            self.tools[tool_name] = metadata

            logger.info(
                "tool_registered",
                tool_name=tool_name,
                description=description,
            )

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def validate_input(
        self, tool_name: str, inputs: Dict[str, Any]
    ) -> bool:
        """
        Validate inputs for a registered tool.

        Args:
            tool_name: Name of the tool
            inputs: Dictionary of input parameters

        Returns:
            True if inputs are valid

        Raises:
            ToolNotFoundError: If tool is not registered
            ToolValidationError: If validation fails
        """
        if tool_name not in self.tools:
            raise ToolNotFoundError(f"Tool '{tool_name}' is not registered")

        metadata = self.tools[tool_name]

        if metadata.input_schema is None:
            # No schema defined, accept any input
            logger.info("tool_validation_skipped", tool_name=tool_name)
            return True

        # Basic validation - check required fields
        required_fields = metadata.input_schema.get("required", [])
        for field in required_fields:
            if field not in inputs:
                raise ToolValidationError(
                    f"Missing required input '{field}' for tool '{tool_name}'"
                )

        logger.info("tool_input_validated", tool_name=tool_name)
        return True

    def execute_tool(
        self, tool_name: str, inputs: Dict[str, Any]
    ) -> Any:
        """
        Execute a registered tool with validation and error handling.

        Args:
            tool_name: Name of the tool to execute
            inputs: Dictionary of input parameters

        Returns:
            Result of tool execution

        Raises:
            ToolNotFoundError: If tool is not registered
            ToolValidationError: If input validation fails
            ToolExecutionError: If execution fails
        """
        # Validate tool exists
        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' is not registered"
            logger.error("tool_execution_failed", tool_name=tool_name, error=error_msg)
            raise ToolNotFoundError(error_msg)

        # Validate inputs
        try:
            self.validate_input(tool_name, inputs)
        except ToolValidationError as e:
            logger.error(
                "tool_validation_failed",
                tool_name=tool_name,
                error=str(e),
            )
            raise

        # Execute tool
        metadata = self.tools[tool_name]
        try:
            result = metadata.func(**inputs)

            execution_record = {
                "tool_name": tool_name,
                "inputs": inputs,
                "result": result,
                "status": "success",
            }
            self.execution_history.append(execution_record)

            logger.info(
                "tool_executed",
                tool_name=tool_name,
                result_type=type(result).__name__,
            )
            return result

        except TypeError as e:
            error_msg = f"Invalid arguments for tool '{tool_name}': {str(e)}"
            logger.error("tool_execution_failed", tool_name=tool_name, error=error_msg)
            raise ToolExecutionError(error_msg) from e
        except Exception as e:
            error_msg = f"Execution error in tool '{tool_name}': {str(e)}"
            logger.error("tool_execution_failed", tool_name=tool_name, error=error_msg)
            raise ToolExecutionError(error_msg) from e

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools.

        Returns:
            List of tool metadata dictionaries
        """
        return [
            {
                "name": metadata.name,
                "description": metadata.description,
                "input_schema": metadata.input_schema,
                "output_schema": metadata.output_schema,
            }
            for metadata in self.tools.values()
        ]

    def get_tool_description(self, tool_name: str) -> str:
        """
        Get the description of a registered tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool description

        Raises:
            ToolNotFoundError: If tool is not registered
        """
        if tool_name not in self.tools:
            raise ToolNotFoundError(f"Tool '{tool_name}' is not registered")
        return self.tools[tool_name].description

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the execution history.

        Returns:
            List of execution records
        """
        return self.execution_history.copy()

    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
        logger.info("execution_history_cleared")
