"""Use a real model to select a local shipment lookup and credit calculator."""

from langchain_core.language_models.chat_models import BaseChatModel

from playground_agentic.example_support import model_from_env, show
from playground_agentic.scenarios.s02_tools.example import (
    calculate_service_credit,
    lookup_shipment,
)
from playground_agentic.scenarios.s02_tools.graph import ChatRequest, ChatResult, build_graph, run


def run_example(model: BaseChatModel) -> ChatResult:
    return run(
        ChatRequest(
            message=(
                "Use the shipment lookup to check TRK-2048. Separately calculate the service "
                "credit for our EUR 1290 plan: 225 minutes downtime in a 43200 minute period. "
                "Report both results, with the shipment ETA and the calculated credit."
            ),
            max_steps=4,
        ),
        compiled_graph=build_graph(model, tools=[lookup_shipment, calculate_service_credit]),
    )


if __name__ == "__main__":
    show(run_example(model_from_env()))
