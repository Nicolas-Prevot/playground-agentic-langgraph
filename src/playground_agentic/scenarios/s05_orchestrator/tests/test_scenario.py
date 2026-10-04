from playground_agentic.scenarios.s05_orchestrator.example import (
    example_1_database_migration,
    example_2_incident_handoff,
)
from playground_agentic.scenarios.s05_orchestrator.graph import (
    OrchestrationRequest,
    Plan,
    PlannedTask,
    Review,
    build_graph,
    run,
)


def _test_planner(_goal: str, _feedback: str, _iteration: int) -> Plan:
    return Plan(
        tasks=[
            PlannedTask(id="one", objective="First independent task"),
            PlannedTask(id="two", objective="Second independent task"),
        ]
    )


def _test_worker(task: PlannedTask) -> str:
    return f"Completed {task.objective}."


def test_example_1_replans_the_database_migration() -> None:
    migration = example_1_database_migration()
    assert migration["status"] == "completed"
    assert migration["iterations"] == 2
    assert migration["task_count"] == 3
    assert "Rollback if replication lag exceeds 30 seconds" in migration["answer"]


def test_example_2_completes_the_incident_handoff_in_one_iteration() -> None:
    incident = example_2_incident_handoff()
    assert incident["status"] == "completed"
    assert incident["iterations"] == 1
    assert incident["task_count"] == 3
    assert "Checkout p95 is 4.8 seconds" in incident["answer"]


def test_iteration_budget_returns_typed_partial_result() -> None:
    graph = build_graph(
        _test_planner,
        _test_worker,
        lambda _goal, _items: Review(approved=False, feedback="revise"),
    )
    result = run(
        OrchestrationRequest(goal="Prepare a report", max_iterations=1), compiled_graph=graph
    )
    assert result.status == "partial"
    assert result.iterations == 1
    assert "revise" in result.reason


def test_request_task_budget_caps_dynamic_plan() -> None:
    result = run(
        OrchestrationRequest(goal="Prepare a report", max_tasks=1, max_iterations=1),
        compiled_graph=build_graph(
            _test_planner,
            _test_worker,
            lambda _goal, _items: Review(approved=False, feedback="task cap reached"),
        ),
    )
    assert result.task_count == 1
    assert result.status == "partial"
