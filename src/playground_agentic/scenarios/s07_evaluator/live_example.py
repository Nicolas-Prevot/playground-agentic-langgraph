"""Draft release notes from facts and optimize against a mixed deterministic/model rubric."""

from langchain_core.language_models.chat_models import BaseChatModel

from playground_agentic.example_support import generate_text, model_from_env, show, structured
from playground_agentic.scenarios.s07_evaluator.graph import (
    Evaluation,
    OptimizationRequest,
    OptimizationResult,
    build_graph,
    run,
)


def run_example(model: BaseChatModel) -> OptimizationResult:
    def generate(
        task: str, instructions: str, resources: list[str], feedback: str, iteration: int
    ) -> str:
        return generate_text(
            model,
            "Write concise release notes using only the supplied facts. "
            "Follow the rubric and address reviewer feedback.",
            {
                "task": task,
                "rubric": instructions,
                "facts": resources,
                "feedback": feedback,
                "iteration": iteration,
            },
        )

    def evaluate(task: str, content: str, instructions: str, resources: list[str]) -> Evaluation:
        missing = [
            label for label in ("Outcome:", "Compatibility:", "Validation:") if label not in content
        ]
        if missing:
            return Evaluation(score=0.2, feedback=f"Include these headings: {missing}")
        if len(content.split()) >= 160:
            return Evaluation(score=0.2, feedback="Shorten the release notes to under 160 words.")
        return structured(
            model,
            Evaluation,
            "Evaluate factual fidelity and usefulness against the supplied facts. "
            "Unsupported claims that tests passed or compatibility is unchanged "
            "must score below 0.8. Give actionable feedback.",
            {"task": task, "draft": content, "rubric": instructions, "facts": resources},
        )

    return run(
        OptimizationRequest(
            task="Write release notes for the hourly invoice importer.",
            instructions="Use Outcome:, Compatibility:, and Validation: headings. Under 160 words.",
            resources=[
                "Added cursor-based hourly invoice imports and stable-ID deduplication.",
                "CSV remains available for rollback. API credentials are a new requirement.",
                "Staging reconciliation passed for 200 fixture invoices; load testing is pending.",
            ],
            max_iterations=3,
        ),
        compiled_graph=build_graph(generate, evaluate),
    )


if __name__ == "__main__":
    show(run_example(model_from_env()))
