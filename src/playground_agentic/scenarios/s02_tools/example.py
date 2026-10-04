"""Run two realistic examples through the explicit tool loop."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool

from playground_agentic.scenarios.s02_tools.graph import (
    ChatRequest,
    build_graph,
    run,
)
from playground_agentic.testing import ScriptedChatModel


@tool
def lookup_shipment(tracking_id: str) -> str:
    """Look up the latest status and ETA for one shipment tracking ID."""

    shipments = {
        "TRK-2048": "delayed at Lyon hub; revised ETA 2026-09-01",
    }
    return shipments.get(tracking_id, "tracking ID not found")


@tool
def calculate_service_credit(
    monthly_fee_eur: float, downtime_minutes: int, billing_period_minutes: int
) -> str:
    """Calculate the proportional service credit for measured downtime."""

    credit = monthly_fee_eur * downtime_minutes / billing_period_minutes
    return f"EUR {credit:.2f}"


def _support_tool_responder(messages: list[BaseMessage], _tools: object) -> AIMessage:
    last = messages[-1]
    if isinstance(last, ToolMessage):
        return AIMessage(content=f"Tool result: {last.content}")
    text = str(last.content)
    tracking = re.search(r"TRK-\d+", text, re.IGNORECASE)
    if tracking:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "lookup_shipment",
                    "args": {"tracking_id": tracking.group(0).upper()},
                    "id": "shipment-1",
                }
            ],
        )
    if "service credit" in text.lower():
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_service_credit",
                    "args": {
                        "monthly_fee_eur": 1290,
                        "downtime_minutes": 225,
                        "billing_period_minutes": 43200,
                    },
                    "id": "credit-1",
                }
            ],
        )
    return AIMessage(content="I cannot answer this request with the allowed tools.")


def example_model() -> ScriptedChatModel:
    return ScriptedChatModel(responder=_support_tool_responder)


# Example 1: read-only operational lookup
def example_1_delayed_shipment_lookup() -> dict[str, object]:
    """Let the model select an authorized read-only shipment lookup."""

    result = run(
        ChatRequest(
            message="Where is shipment TRK-2048, and what ETA should support give the customer?"
        ),
        compiled_graph=build_graph(example_model(), tools=[lookup_shipment]),
    )
    return result.model_dump()


# Example 2: deterministic domain calculation
def example_2_sla_service_credit() -> dict[str, object]:
    """Let the model select a deterministic calculator with complete inputs."""

    result = run(
        ChatRequest(
            message=(
                "Calculate the proportional service credit for a EUR 1,290 monthly plan with "
                "225 minutes of measured downtime in a 43,200-minute billing period."
            )
        ),
        compiled_graph=build_graph(example_model(), tools=[calculate_service_credit]),
    )
    return result.model_dump()


def run_example() -> dict[str, object]:
    return {
        "delayed_shipment_lookup": example_1_delayed_shipment_lookup(),
        "sla_service_credit": example_2_sla_service_credit(),
    }


if __name__ == "__main__":
    print(run_example())
