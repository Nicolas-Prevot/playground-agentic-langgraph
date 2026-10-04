"""Run two domain-specific planner/worker/reviewer examples."""

from playground_agentic.scenarios.s05_orchestrator.graph import (
    OrchestrationRequest,
    Plan,
    PlannedTask,
    Review,
    WorkerArtifact,
    build_graph,
    run,
)


def _example_planner(goal: str, feedback: str, iteration: int) -> Plan:
    if "incident" in goal.lower():
        return Plan(
            tasks=[
                PlannedTask(id="impact", objective="Quantify affected checkout traffic"),
                PlannedTask(id="mitigation", objective="Record the active mitigation"),
                PlannedTask(id="next-checks", objective="Define on-call validation checks"),
            ]
        )
    tasks = [
        PlannedTask(id="compatibility", objective="Inventory migration compatibility risks"),
        PlannedTask(id="rollout", objective="Design a zero-downtime rollout sequence"),
    ]
    if feedback or iteration > 1:
        tasks.append(PlannedTask(id="risks", objective="Define rollback triggers and mitigations"))
    return Plan(tasks=tasks)


def _example_worker(task: PlannedTask) -> str:
    artifacts = {
        "compatibility": (
            "Verify extension support, long-running transactions, and logical-replication lag "
            "against a production snapshot."
        ),
        "rollout": (
            "Create the target cluster, backfill through logical replication, shadow reads, "
            "then shift writes during a measured maintenance window."
        ),
        "risks": (
            "Rollback if replication lag exceeds 30 seconds or checkout errors exceed 1%; "
            "retain the source as read-only until reconciliation passes."
        ),
        "impact": "Checkout p95 is 4.8 seconds and 18% of EU payment attempts time out.",
        "mitigation": (
            "Traffic is pinned to the healthy payment provider and the new retry policy is off."
        ),
        "next-checks": (
            "On-call will check timeout rate, provider errors, and queue depth every 15 minutes."
        ),
    }
    return artifacts[task.id]


def _example_reviewer(goal: str, artifacts: list[WorkerArtifact]) -> Review:
    identifiers = {artifact.task_id for artifact in artifacts}
    required = (
        {"impact", "mitigation", "next-checks"}
        if "incident" in goal.lower()
        else {"compatibility", "rollout", "risks"}
    )
    missing = sorted(required - identifiers)
    if missing:
        return Review(approved=False, feedback=f"Add missing work: {', '.join(missing)}.")
    return Review(approved=True, feedback="The plan covers the required operational decisions.")


def _example_graph():
    return build_graph(_example_planner, _example_worker, _example_reviewer)


# Example 1: review-triggered replanning
def example_1_database_migration() -> dict[str, object]:
    """Replan a database migration after review finds missing rollback policy."""

    result = run(
        OrchestrationRequest(
            goal=(
                "Plan a zero-downtime PostgreSQL 15 to 17 migration for the checkout database, "
                "including compatibility checks, rollout, and rollback policy."
            )
        ),
        compiled_graph=_example_graph(),
    )
    return result.model_dump()


# Example 2: complete first-pass plan
def example_2_incident_handoff() -> dict[str, object]:
    """Plan and approve a complete operational handoff in one iteration."""

    result = run(
        OrchestrationRequest(
            goal=(
                "Prepare the checkout latency incident handoff before the EU on-call rotation "
                "changes in 30 minutes."
            )
        ),
        compiled_graph=_example_graph(),
    )
    return result.model_dump()


def run_example() -> dict[str, object]:
    return {
        "database_migration": example_1_database_migration(),
        "incident_handoff": example_2_incident_handoff(),
    }


if __name__ == "__main__":
    print(run_example())
