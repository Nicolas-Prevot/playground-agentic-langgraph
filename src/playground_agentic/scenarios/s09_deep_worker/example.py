"""Run two open-ended artifact tasks inside a bounded Deep Agent node."""

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from playground_agentic.scenarios.s09_deep_worker.graph import DeepTaskRequest, build_graph, run
from playground_agentic.testing import ScriptedChatModel


def _tool_call(name: str, args: dict[str, Any], identifier: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identifier}])


def example_models(
    case: Literal["dependency-upgrade", "runbook-review"] = "dependency-upgrade",
) -> tuple[ScriptedChatModel, ScriptedChatModel]:
    if case == "dependency-upgrade":
        research_question = (
            "Identify migration and rollback risks for the checkout API dependency upgrade."
        )
        artifact = (
            "# Checkout API Dependency Upgrade Assessment\n\n"
            "Outcome: stage the FastAPI upgrade behind one checkout canary before fleet "
            "rollout.\n\n"
            "## Evidence\n"
            "The current service uses deprecated startup hooks and its payment callback tests "
            "exercise only the old lifecycle. Add lifespan-path tests and compare canary error "
            "rate, latency, and callback acknowledgements.\n\n"
            "## Limits\n"
            "This assessment does not replace a dependency lock diff, staging load test, or "
            "verified rollback rehearsal.\n"
        )
        research_evidence = (
            "Lifecycle changes and transitive dependency drift are the primary upgrade risks."
        )
    else:
        research_question = "Identify operational gaps in the checkout latency incident runbook."
        artifact = (
            "# Checkout Latency Runbook Gap Review\n\n"
            "Outcome: add provider isolation, queue saturation, and customer-impact branches.\n\n"
            "## Evidence\n"
            "The existing runbook checks application latency but omits payment-provider errors, "
            "queue depth, rollback ownership, and the threshold for a public status update.\n\n"
            "## Limits\n"
            "Validate the proposed branches in a game day and assign named service owners before "
            "treating the runbook as production-ready.\n"
        )
        research_evidence = (
            "Useful runbooks connect symptoms to thresholds, owners, mitigation, and escalation."
        )

    def responder(messages: list[BaseMessage], _tools: object) -> AIMessage:
        last_tool = next(
            (item for item in reversed(messages) if isinstance(item, ToolMessage)), None
        )
        if last_tool is None:
            return _tool_call(
                "task",
                {
                    "description": research_question,
                    "subagent_type": "researcher",
                },
                "delegate-1",
            )
        if last_tool.name == "task":
            return _tool_call(
                "read_file", {"file_path": "/skills/artifact-author/SKILL.md"}, "skill-1"
            )
        if last_tool.name == "read_file":
            return _tool_call(
                "write_file",
                {"file_path": "/artifacts/result.md", "content": artifact},
                "write-1",
            )
        return AIMessage(content="The requested artifact is ready.")

    parent = ScriptedChatModel(responder=responder, model_name="offline-deep-parent")
    researcher = ScriptedChatModel(
        responder=lambda _messages, _tools: AIMessage(content=research_evidence),
        model_name="offline-deep-researcher",
    )
    return parent, researcher


# Example 1: bounded dependency-upgrade artifact work
def example_1_dependency_upgrade_assessment() -> dict[str, object]:
    """Use bounded delegation and file work for a dependency-upgrade artifact."""

    upgrade_parent, upgrade_researcher = example_models("dependency-upgrade")
    result = run(
        DeepTaskRequest(
            task=(
                "Assess upgrading the checkout API from its deprecated FastAPI startup hooks to "
                "the current lifespan model. Produce a decision artifact with evidence, rollout "
                "risk, and explicit limits."
            )
        ),
        compiled_graph=build_graph(upgrade_parent, researcher_model=upgrade_researcher),
    )
    return result.model_dump()


# Example 2: bounded runbook-review artifact work
def example_2_incident_runbook_review() -> dict[str, object]:
    """Use the same bounded worker contract for an operational runbook review."""

    runbook_parent, runbook_researcher = example_models("runbook-review")
    result = run(
        DeepTaskRequest(
            task=(
                "Review the checkout latency incident runbook for missing diagnostic, mitigation, "
                "ownership, and communication steps. Produce one evidence-and-limits artifact."
            )
        ),
        compiled_graph=build_graph(runbook_parent, researcher_model=runbook_researcher),
    )
    return result.model_dump()


def run_example() -> dict[str, object]:
    return {
        "dependency_upgrade_assessment": example_1_dependency_upgrade_assessment(),
        "incident_runbook_review": example_2_incident_runbook_review(),
    }


if __name__ == "__main__":
    print(run_example())
