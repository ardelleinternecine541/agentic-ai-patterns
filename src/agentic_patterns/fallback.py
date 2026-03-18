"""
Fallback & Recovery Pattern Implementation.

Provides graceful degradation and error recovery with configurable retry
policies, exponential backoff, and fallback chains. Ensures agent systems
remain operational even when primary execution paths fail.
"""

from typing import Any, Callable, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import structlog

logger = structlog.get_logger(__name__)


class RetryStrategy(Enum):
    """Enumeration of retry strategies."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED = "fixed"
    IMMEDIATE = "immediate"


@dataclass
class FallbackStep:
    """Represents a single step in a fallback chain."""
    func: Callable
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 1.0
    max_delay: float = 60.0
    description: str = ""


class FallbackChainError(Exception):
    """Raised when all fallback steps fail."""
    pass


class FallbackChain:
    """
    Fallback chain implementation for graceful degradation.

    Provides a sequence of fallback steps with configurable retry logic.
    Each step is attempted with configured retry strategy until success
    or all retries exhausted, then moves to next step.
    """

    def __init__(self):
        """Initialize the fallback chain."""
        self.steps: List[FallbackStep] = []
        self.execution_history: List[dict] = []
        logger.info("fallback_chain_initialized")

    def add_step(
        self,
        func: Callable,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        description: str = "",
    ) -> None:
        """
        Add a step to the fallback chain.

        Args:
            func: Callable to execute
            max_retries: Maximum number of retry attempts
            retry_strategy: Strategy for retry timing
            initial_delay: Initial delay for first retry (seconds)
            max_delay: Maximum delay between retries (seconds)
            description: Description of this fallback step
        """
        step = FallbackStep(
            func=func,
            max_retries=max_retries,
            retry_strategy=retry_strategy,
            initial_delay=initial_delay,
            max_delay=max_delay,
            description=description or func.__name__,
        )
        self.steps.append(step)
        logger.info(
            "fallback_step_added",
            step_num=len(self.steps),
            description=step.description,
        )

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the fallback chain.

        Args:
            *args: Positional arguments to pass to each step
            **kwargs: Keyword arguments to pass to each step

        Returns:
            Result of the first successful step

        Raises:
            FallbackChainError: If all steps fail
        """
        if not self.steps:
            raise FallbackChainError("Fallback chain is empty")

        logger.info("fallback_chain_execution_started", step_count=len(self.steps))

        for step_idx, step in enumerate(self.steps):
            logger.info(
                "attempting_fallback_step",
                step_num=step_idx + 1,
                description=step.description,
            )

            try:
                result = self._execute_with_retry(step, *args, **kwargs)

                execution_record = {
                    "step": step_idx + 1,
                    "description": step.description,
                    "status": "success",
                    "result": result,
                }
                self.execution_history.append(execution_record)

                logger.info(
                    "fallback_step_succeeded",
                    step_num=step_idx + 1,
                    description=step.description,
                )
                return result

            except Exception as e:
                logger.error(
                    "fallback_step_failed",
                    step_num=step_idx + 1,
                    description=step.description,
                    error=str(e),
                )

                execution_record = {
                    "step": step_idx + 1,
                    "description": step.description,
                    "status": "failed",
                    "error": str(e),
                }
                self.execution_history.append(execution_record)

                if step_idx == len(self.steps) - 1:
                    # Last step failed
                    logger.error(
                        "all_fallback_steps_exhausted",
                        total_steps=len(self.steps),
                    )
                    raise FallbackChainError(
                        f"All {len(self.steps)} fallback steps failed"
                    ) from e

        raise FallbackChainError("Fallback chain execution failed unexpectedly")

    def _execute_with_retry(
        self,
        step: FallbackStep,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a step with retry logic.

        Args:
            step: FallbackStep to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the execution

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(step.max_retries + 1):
            try:
                result = step.func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        "retry_succeeded",
                        step=step.description,
                        attempt=attempt + 1,
                    )
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    "step_execution_failed",
                    step=step.description,
                    attempt=attempt + 1,
                    max_retries=step.max_retries,
                    error=str(e),
                )

                if attempt < step.max_retries:
                    delay = self._calculate_delay(
                        attempt,
                        step.retry_strategy,
                        step.initial_delay,
                        step.max_delay,
                    )
                    logger.info(
                        "waiting_for_retry",
                        step=step.description,
                        delay=delay,
                    )
                    time.sleep(delay)

        raise last_exception

    @staticmethod
    def _calculate_delay(
        attempt: int,
        strategy: RetryStrategy,
        initial_delay: float,
        max_delay: float,
    ) -> float:
        """
        Calculate delay based on retry strategy.

        Args:
            attempt: Current attempt number (0-indexed)
            strategy: RetryStrategy to use
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Delay in seconds
        """
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif strategy == RetryStrategy.FIXED:
            return initial_delay
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = initial_delay * (attempt + 1)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = initial_delay * (2 ** attempt)
        else:
            delay = initial_delay

        return min(delay, max_delay)

    def get_execution_history(self) -> List[dict]:
        """
        Get the execution history of the chain.

        Returns:
            List of execution records
        """
        return self.execution_history.copy()

    def clear_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
        logger.info("fallback_chain_history_cleared")
