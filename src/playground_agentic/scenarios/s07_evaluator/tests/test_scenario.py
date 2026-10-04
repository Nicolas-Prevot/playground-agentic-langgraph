from pathlib import Path

from playground_agentic.scenarios.s07_evaluator.example import (
    example_1_release_notes_with_skill,
    example_2_incident_status_update,
)
from playground_agentic.scenarios.s07_evaluator.graph import (
    Evaluation,
    OptimizationRequest,
    build_graph,
    run,
)
from playground_agentic.scenarios.s07_evaluator.skill_adapter import SkillRegistry


def test_example_1_uses_skill_context_until_release_notes_are_accepted() -> None:
    release_notes = example_1_release_notes_with_skill()
    assert release_notes["skill"] == "release-notes"
    assert release_notes["status"] == "accepted"
    assert release_notes["iterations"] == 2
    assert release_notes["score"] >= 0.8


def test_example_2_optimizes_an_incident_update_without_a_skill() -> None:
    incident = example_2_incident_status_update()
    assert incident["status"] == "accepted"
    assert incident["iterations"] == 2
    assert "Next update: 15:30 UTC" in incident["content"]


def test_core_workflow_does_not_require_a_skill() -> None:
    result = run(
        OptimizationRequest(task="Write release notes"),
        compiled_graph=build_graph(),
    )
    assert result.status == "accepted"
    assert result.iterations == 2


def test_iteration_budget_returns_best_partial_artifact() -> None:
    graph = build_graph(
        evaluator=lambda _task, _content, _instructions, _resources: Evaluation(
            score=0.2, feedback="again"
        )
    )
    result = run(
        OptimizationRequest(task="Write release notes", max_iterations=1),
        compiled_graph=graph,
    )
    assert result.status == "partial"
    assert result.content


def test_skill_loader_rejects_path_traversal() -> None:
    registry = SkillRegistry(Path(__file__).parents[1] / "skills")
    try:
        registry.load("../outside")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path traversal should fail")
