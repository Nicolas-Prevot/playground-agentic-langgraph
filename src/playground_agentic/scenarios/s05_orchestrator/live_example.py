"""Plan independent migration workstreams, draft them, and review coverage."""

from langchain_core.language_models.chat_models import BaseChatModel

from playground_agentic.example_support import generate_text, model_from_env, show, structured
from playground_agentic.scenarios.s05_orchestrator.graph import (
    OrchestrationRequest,
    OrchestrationResult,
    Plan,
    PlannedTask,
    Review,
    WorkerArtifact,
    build_graph,
    run,
)

BRIEF = (
    "Plan a migration from nightly invoice CSV imports to hourly API imports. "
    "The fictional API has stable invoice IDs, cursor pagination, and a 60-request/minute limit. "
    "Retain the CSV importer for rollback. Require reconciliation before switching readers. "
    "Cover deduplication, rate limiting, canary validation, and rollback ownership."
)


def run_example(model: BaseChatModel) -> OrchestrationResult:
    def planner(goal: str, feedback: str, iteration: int) -> Plan:
        return structured(
            model,
            Plan,
            "Plan at most three independent writing tasks with unique lowercase-hyphen IDs. "
            "Each task needs only the brief and its objective. Incorporate feedback. "
            "This is a parallel planning exercise; do not create dependent execution tasks.",
            {"goal": goal, "feedback": feedback, "iteration": iteration},
        )

    def worker(task: PlannedTask) -> str:
        return generate_text(
            model,
            "Draft a concrete migration workstream in under 180 words. "
            "Use only supplied facts; name assumptions and proposed validation.",
            {"brief": BRIEF, "task": task.model_dump()},
        )

    def reviewer(goal: str, artifacts: list[WorkerArtifact]) -> Review:
        return structured(
            model,
            Review,
            "Approve only if the combined plan covers deduplication, rate limits, canary "
            "reconciliation, and rollback ownership. Otherwise give actionable feedback.",
            {"goal": goal, "artifacts": [item.model_dump() for item in artifacts]},
        )

    return run(
        OrchestrationRequest(goal=BRIEF, max_tasks=3, max_iterations=2),
        compiled_graph=build_graph(planner, worker, reviewer),
    )


if __name__ == "__main__":
    show(run_example(model_from_env()))
