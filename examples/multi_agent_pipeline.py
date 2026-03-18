"""
Multi-Agent Pipeline Example.

Demonstrates the orchestrator pattern with multiple specialized agents
working together on a complex task with dependencies.
"""

from agentic_patterns.orchestrator import AgentOrchestrator, Agent


def create_planner_agent() -> Agent:
    """Create a planning agent."""

    def execute_plan(task: str, **kwargs) -> dict:
        """Execute planning task."""
        plan_steps = [
            "1. Analyze requirements",
            "2. Break down into subtasks",
            "3. Define dependencies",
            "4. Estimate timeline",
        ]
        return {
            "task": task,
            "plan": plan_steps,
            "status": "planning_complete",
        }

    return Agent(
        name="planner_agent",
        agent_type="planner",
        execute=execute_plan,
        capabilities=["task_analysis", "planning", "breakdown"],
    )


def create_executor_agent() -> Agent:
    """Create an execution agent."""

    def execute_task(task: str, **kwargs) -> dict:
        """Execute implementation task."""
        dep_plan = kwargs.get("dep_planning_task")

        results = []
        if dep_plan:
            plan = dep_plan.get("plan", [])
            for step in plan:
                results.append(f"Executed: {step}")

        return {
            "task": task,
            "execution_results": results,
            "status": "execution_complete",
        }

    return Agent(
        name="executor_agent",
        agent_type="executor",
        execute=execute_task,
        capabilities=["implementation", "execution", "testing"],
    )


def create_validator_agent() -> Agent:
    """Create a validation agent."""

    def execute_validation(task: str, **kwargs) -> dict:
        """Execute validation task."""
        dep_execution = kwargs.get("dep_execution_task")

        validation_results = {
            "task": task,
            "checks_passed": 0,
            "checks_failed": 0,
            "status": "validation_complete",
        }

        if dep_execution:
            # Simulate validation checks
            validation_results["checks_passed"] = 4
            validation_results["quality_score"] = 0.95

        return validation_results

    return Agent(
        name="validator_agent",
        agent_type="validator",
        execute=execute_validation,
        capabilities=["validation", "testing", "quality_assurance"],
    )


def main():
    """Run multi-agent pipeline example."""
    print("Agentic AI Patterns - Multi-Agent Pipeline Example\n")

    # Create and register agents
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(create_planner_agent())
    orchestrator.register_agent(create_executor_agent())
    orchestrator.register_agent(create_validator_agent())

    print("Registered Agents:")
    for name, agent in orchestrator.agents.items():
        print(f"  - {name}: {agent.agent_type} ({', '.join(agent.capabilities)})")
    print()

    # Define task pipeline
    print("Building task pipeline...")
    orchestrator.add_task(
        task_id="planning_task",
        description="Plan the implementation of a web scraper",
        agent_type="planner",
    )

    orchestrator.add_task(
        task_id="execution_task",
        description="Execute the implementation plan",
        agent_type="executor",
        dependencies=["planning_task"],
    )

    orchestrator.add_task(
        task_id="validation_task",
        description="Validate the implementation",
        agent_type="validator",
        dependencies=["execution_task"],
    )

    print(f"Added {len(orchestrator.tasks)} tasks to pipeline\n")

    # Execute pipeline
    print("Executing pipeline...")
    print("-" * 60)

    try:
        results = orchestrator.execute_pipeline()

        print("\nPipeline execution completed successfully!\n")

        # Display aggregated results
        print("Aggregated Results:")
        aggregated = orchestrator.aggregate_results()
        for task_id, result in aggregated.items():
            print(f"\n{task_id}:")
            print(f"  Status: {result['status']}")
            print(f"  Result: {result['result']}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    main()
