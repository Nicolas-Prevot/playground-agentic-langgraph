from langchain_core.messages import AIMessage

from playground_agentic.scenarios.s09_deep_worker.example import (
    example_1_dependency_upgrade_assessment,
    example_2_incident_runbook_review,
)
from playground_agentic.scenarios.s09_deep_worker.graph import DeepTaskRequest, build_graph, run
from playground_agentic.testing import ScriptedChatModel


def test_example_1_uses_skill_subagent_and_filesystem_for_upgrade_assessment() -> None:
    upgrade = example_1_dependency_upgrade_assessment()
    assert upgrade["status"] == "completed"
    assert upgrade["artifact"].startswith("# Checkout API Dependency Upgrade Assessment")
    assert "## Evidence" in upgrade["artifact"]
    assert "## Limits" in upgrade["artifact"]
    assert upgrade["model_calls"] == 4


def test_example_2_uses_the_bounded_worker_for_a_runbook_review() -> None:
    runbook = example_2_incident_runbook_review()
    assert runbook["status"] == "completed"
    assert runbook["artifact"].startswith("# Checkout Latency Runbook Gap Review")
    assert "payment-provider errors" in runbook["artifact"]
    assert runbook["model_calls"] == 4


def test_parent_rejects_missing_artifact() -> None:
    model = ScriptedChatModel(responder=lambda _messages, _tools: AIMessage(content="done"))
    result = run(
        DeepTaskRequest(task="Do not create a file"),
        compiled_graph=build_graph(model, researcher_model=model),
    )
    assert result.status == "invalid_artifact"
    assert result.artifact == ""
    assert "Missing required sections" in result.validation_error


def test_artifact_path_is_a_fixed_boundary_contract() -> None:
    try:
        DeepTaskRequest(task="write", artifact_path="/outside.txt")
    except Exception as exc:
        assert "literal_error" in str(exc)
    else:
        raise AssertionError("arbitrary artifact paths must be rejected")
