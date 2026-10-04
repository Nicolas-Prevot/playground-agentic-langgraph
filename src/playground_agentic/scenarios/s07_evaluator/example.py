"""Run skilled release notes and a skill-free incident update optimizer."""

from pathlib import Path

from playground_agentic.scenarios.s07_evaluator.graph import (
    Evaluation,
    OptimizationRequest,
    build_graph,
    run,
)
from playground_agentic.scenarios.s07_evaluator.skill_adapter import SkillRegistry


def _incident_generator(
    task: str, instructions: str, resources: list[str], feedback: str, iteration: int
) -> str:
    del task, instructions, resources
    if iteration == 1:
        return "Impact: EU checkout p95 reached 4.8 seconds."
    return (
        "Impact: EU checkout p95 reached 4.8 seconds; 18% of payments timed out.\n"
        "Current status: traffic is pinned to the healthy provider.\n"
        "Mitigation: the new retry policy remains disabled while queues drain.\n"
        "Next update: 15:30 UTC after error-rate and queue-depth validation.\n"
        f"Feedback addressed: {feedback}"
    )


def _incident_evaluator(
    task: str, content: str, instructions: str, resources: list[str]
) -> Evaluation:
    del task, instructions, resources
    required = ("Impact:", "Current status:", "Mitigation:", "Next update:")
    missing = [section for section in required if section not in content]
    if missing:
        return Evaluation(score=0.4, feedback=f"Add {', '.join(missing)}")
    return Evaluation(score=0.96, feedback="The update is actionable and time-bounded.")


# Example 1: evaluator loop with an Agent Skill rubric
def example_1_release_notes_with_skill() -> dict[str, object]:
    """Optimize release notes using an Agent Skill as the rubric source."""

    skill = SkillRegistry(Path(__file__).with_name("skills")).load("release-notes")
    result = run(
        OptimizationRequest(
            task=(
                "Create release notes for a resumable document-enrichment workflow that avoids "
                "reprocessing completed documents after restart."
            ),
            instructions=skill.instructions,
            resources=skill.resources,
        ),
        compiled_graph=build_graph(),
    )
    return {"skill": skill.name, **result.model_dump()}


# Example 2: evaluator loop with an application-owned rubric
def example_2_incident_status_update() -> dict[str, object]:
    """Optimize an incident update with an application-owned evaluator and no Skill."""

    result = run(
        OptimizationRequest(
            task="Write the next public checkout-latency incident update.",
            minimum_score=0.9,
        ),
        compiled_graph=build_graph(_incident_generator, _incident_evaluator),
    )
    return result.model_dump()


def run_example() -> dict[str, object]:
    return {
        "release_notes": example_1_release_notes_with_skill(),
        "incident_status_update": example_2_incident_status_update(),
    }


if __name__ == "__main__":
    print(run_example())
