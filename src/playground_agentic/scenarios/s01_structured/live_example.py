"""Classify an actual intake with a model; routing stays deterministic."""

from langchain_core.language_models.chat_models import BaseChatModel

from playground_agentic.example_support import model_from_env, show, structured
from playground_agentic.scenarios.s01_structured.graph import Analysis, build_graph


def run_example(model: BaseChatModel) -> dict[str, object]:
    def analyze(text: str) -> Analysis:
        return structured(
            model,
            Analysis,
            "Classify an incoming customer message as support, sales, or risk. "
            "Risk means suspected abuse or fraud. Summarize only stated facts. "
            "Report uncertainty honestly; confidence is a heuristic, not a probability guarantee.",
            {"message": text},
        )

    return build_graph(analyze).invoke(
        {
            "text": "Our invoice INV-2048 charged us twice for the same September subscription. "
            "Please investigate and explain the correction.",
            "minimum_confidence": 0.8,
        }
    )


if __name__ == "__main__":
    show(run_example(model_from_env()))
