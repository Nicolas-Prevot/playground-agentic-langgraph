"""Run conflict and consensus examples through parallel specialist reviews."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from playground_agentic.scenarios.s04_specialists.graph import (
    SpecialistRequest,
    build_graph,
    run,
)
from playground_agentic.testing import ScriptedChatModel


def _specialist_responder(messages: list[BaseMessage], _tools: object) -> AIMessage:
    text = str(next(item for item in reversed(messages) if isinstance(item, HumanMessage)).content)
    role = text.split("SPECIALIST=", 1)[1].splitlines()[0]
    if role == "security":
        return AIMessage(content="REVISE: encrypt exports and require scoped access.")
    if role == "delivery":
        return AIMessage(content="APPROVE: use a staged rollout with rollback metrics.")
    return AIMessage(content="APPROVE: the change solves the stated customer problem.")


def example_model() -> ScriptedChatModel:
    return ScriptedChatModel(responder=_specialist_responder)


def _example_router(question: str) -> list[str]:
    selected = ["product", "delivery"]
    if "customer payment data" in question.lower():
        selected.append("security")
    return selected


# Example 1: sensitive decision with specialist disagreement
def example_1_customer_data_export() -> dict[str, object]:
    """Preserve disagreement among all specialists needed for a sensitive export."""

    result = run(
        SpecialistRequest(
            question=(
                "Should account owners be able to export customer payment data from the new "
                "billing portal, with a staged release next month?"
            )
        ),
        compiled_graph=build_graph(example_model(), _example_router),
    )
    return result.model_dump()


# Example 2: scoped specialist consensus
def example_2_staged_ui_release() -> dict[str, object]:
    """Reach consensus using only the product and delivery specialists."""

    result = run(
        SpecialistRequest(
            question=(
                "Can we release the redesigned invoice filters to 10% of tenants behind a "
                "feature flag, with rollback on error-rate regression?"
            ),
            specialists=["product", "delivery"],
        ),
        compiled_graph=build_graph(example_model(), _example_router),
    )
    return result.model_dump()


def run_example() -> dict[str, object]:
    return {
        "customer_data_export": example_1_customer_data_export(),
        "staged_ui_release": example_2_staged_ui_release(),
    }


if __name__ == "__main__":
    print(run_example())
