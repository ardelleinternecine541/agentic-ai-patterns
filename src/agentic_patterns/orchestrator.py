"""
Multi-Agent Orchestration Pattern Implementation.

Manages multiple specialized agents working towards common goals. Handles
task distribution, result aggregation, and inter-agent communication with
dependency management.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    """Enumeration of task statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    task_id: str
    description: str
    agent_type: str
    dependencies: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None


@dataclass
class Agent:
    """Represents an agent in the orchestrator."""
    name: str
    agent_type: str
    execute: Callable
    capabilities: List[str] = field(default_factory=list)


class OrchestrationError(Exception):
    """Raised when orchestration fails."""
    pass


class AgentOrchestrator:
    """
    Multi-agent orchestration system.

    Manages a team of specialized agents, distributes tasks, handles
    dependencies, and aggregates results. Supports task coordination
    and result combination.
    """

    def __init__(self, agents: Optional[List[Agent]] = None):
        """
        Initialize the orchestrator.

        Args:
            agents: Optional list of agents to register
        """
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.execution_order: List[str] = []

        if agents:
            for agent in agents:
                self.register_agent(agent)

        logger.info("agent_orchestrator_initialized")

    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        logger.info(
            "agent_registered",
            name=agent.name,
            agent_type=agent.agent_type,
            capabilities=agent.capabilities,
        )

    def add_task(
        self,
        task_id: str,
        description: str,
        agent_type: str,
        dependencies: Optional[List[str]] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Add a task to the orchestration queue.

        Args:
            task_id: Unique task identifier
            description: Task description
            agent_type: Type of agent to execute task
            dependencies: Optional list of task IDs this task depends on
            inputs: Optional input parameters for the task

        Returns:
            The created Task object
        """
        task = Task(
            task_id=task_id,
            description=description,
            agent_type=agent_type,
            dependencies=dependencies or [],
            inputs=inputs or {},
        )
        self.tasks[task_id] = task
        self.task_queue.append(task_id)

        logger.info(
            "task_added",
            task_id=task_id,
            agent_type=agent_type,
            dependencies=task.dependencies,
        )
        return task

    def execute_task(self, task_id: str) -> Any:
        """
        Execute a single task.

        Args:
            task_id: ID of the task to execute

        Returns:
            Result of task execution

        Raises:
            OrchestrationError: If task execution fails
        """
        if task_id not in self.tasks:
            raise OrchestrationError(f"Task '{task_id}' not found")

        task = self.tasks[task_id]

        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                raise OrchestrationError(
                    f"Dependency '{dep_id}' not found for task '{task_id}'"
                )

            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                raise OrchestrationError(
                    f"Dependency '{dep_id}' not completed for task '{task_id}'"
                )

        # Find suitable agent
        agent = self._find_agent_for_task(task)
        if agent is None:
            raise OrchestrationError(
                f"No agent found for task type '{task.agent_type}'"
            )

        task.status = TaskStatus.RUNNING
        logger.info("task_execution_started", task_id=task_id, agent=agent.name)

        try:
            # Prepare inputs with dependency results
            execution_inputs = task.inputs.copy()
            for dep_id in task.dependencies:
                dep_result = self.tasks[dep_id].result
                execution_inputs[f"dep_{dep_id}"] = dep_result

            # Execute task
            result = agent.execute(task.description, **execution_inputs)
            task.result = result
            task.status = TaskStatus.COMPLETED

            logger.info(
                "task_execution_completed",
                task_id=task_id,
                agent=agent.name,
            )
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(
                "task_execution_failed",
                task_id=task_id,
                agent=agent.name,
                error=str(e),
            )
            raise OrchestrationError(f"Task '{task_id}' failed: {str(e)}") from e

    def execute_pipeline(self) -> Dict[str, Any]:
        """
        Execute all tasks in dependency order.

        Returns:
            Dictionary mapping task_id to result

        Raises:
            OrchestrationError: If any task fails
        """
        # Topologically sort tasks by dependencies
        execution_order = self._topological_sort()

        logger.info(
            "pipeline_execution_started",
            task_count=len(execution_order),
        )

        results = {}

        for task_id in execution_order:
            try:
                result = self.execute_task(task_id)
                results[task_id] = result
            except OrchestrationError as e:
                logger.error(
                    "pipeline_execution_failed",
                    task_id=task_id,
                    error=str(e),
                )
                raise

        logger.info(
            "pipeline_execution_completed",
            completed_tasks=len(results),
        )

        return results

    def aggregate_results(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Aggregate results from multiple tasks.

        Args:
            task_ids: List of task IDs to aggregate (None for all)

        Returns:
            Dictionary of aggregated results
        """
        if task_ids is None:
            task_ids = list(self.tasks.keys())

        aggregated = {}
        for task_id in task_ids:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                aggregated[task_id] = {
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                }

        logger.info("results_aggregated", task_count=len(aggregated))
        return aggregated

    def get_task_status(self, task_id: str) -> TaskStatus:
        """
        Get the status of a task.

        Args:
            task_id: ID of the task

        Returns:
            TaskStatus enum value
        """
        if task_id not in self.tasks:
            raise OrchestrationError(f"Task '{task_id}' not found")
        return self.tasks[task_id].status

    def reset(self) -> None:
        """Reset all tasks to pending status."""
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING
            task.result = None
            task.error = None

        logger.info("orchestrator_reset")

    def _find_agent_for_task(self, task: Task) -> Optional[Agent]:
        """
        Find a suitable agent for a task.

        Args:
            task: Task to find an agent for

        Returns:
            Suitable Agent or None
        """
        for agent in self.agents.values():
            if agent.agent_type == task.agent_type:
                return agent
        return None

    def _topological_sort(self) -> List[str]:
        """
        Topologically sort tasks by dependencies.

        Returns:
            List of task IDs in execution order

        Raises:
            OrchestrationError: If circular dependencies exist
        """
        # Build dependency graph
        in_degree = {task_id: len(self.tasks[task_id].dependencies) for task_id in self.tasks}
        graph = {task_id: [] for task_id in self.tasks}

        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id in graph:
                    graph[dep_id].append(task_id)

        # Kahn's algorithm
        queue = [task_id for task_id in self.tasks if in_degree[task_id] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.tasks):
            raise OrchestrationError("Circular dependency detected in tasks")

        self.execution_order = result
        return result
