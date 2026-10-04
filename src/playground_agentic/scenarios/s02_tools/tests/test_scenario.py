from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph

from playground_agentic.scenarios.s02_tools.example import (
    example_1_delayed_shipment_lookup,
    example_2_sla_service_credit,
    example_model,
    lookup_shipment,
)
from playground_agentic.scenarios.s02_tools.graph import (
    ChatRequest,
    build_graph,
    run,
    stream_text,
)
from playground_agentic.testing import ScriptedChatModel


@tool
def failing_lookup(item_id: str) -> str:
    """Look up an item in an unavailable test dependency."""

    raise RuntimeError(f"dependency unavailable for {item_id}")


def _failing_tool_responder(messages: list[BaseMessage], _tools: object) -> AIMessage:
    if isinstance(messages[-1], ToolMessage):
        return AIMessage(content="The lookup failed and was not hidden.")
    return AIMessage(
        content="",
        tool_calls=[{"name": "failing_lookup", "args": {"item_id": "item-7"}, "id": "failure-1"}],
    )


def test_example_1_executes_a_real_lookup_and_exposes_topology() -> None:
    shipment = example_1_delayed_shipment_lookup()
    assert shipment["answer"] == "Tool result: delayed at Lyon hub; revised ETA 2026-09-01"
    assert shipment["status"] == "completed"
    assert shipment["steps"] == 2

    compiled = build_graph(example_model(), tools=[lookup_shipment])
    assert isinstance(compiled, CompiledStateGraph)
    assert {"model", "tools", "done", "budget"} <= set(compiled.nodes)


def test_example_2_calculates_a_reproducible_service_credit() -> None:
    credit = example_2_sla_service_credit()
    assert credit["answer"] == "Tool result: EUR 6.72"
    assert credit["status"] == "completed"
    assert credit["steps"] == 2


def test_loop_stops_at_the_request_budget() -> None:
    exhausted = run(
        ChatRequest(message="Where is TRK-2048?", max_steps=1),
        compiled_graph=build_graph(example_model(), tools=[lookup_shipment]),
    )
    assert exhausted.status == "budget_exhausted"
    assert exhausted.steps == 1


def test_tool_errors_are_visible_in_state_and_streaming_yields_chunks() -> None:
    failure_model = ScriptedChatModel(responder=_failing_tool_responder)
    failed = run(
        ChatRequest(message="Look up item-7"),
        compiled_graph=build_graph(failure_model, tools=[failing_lookup]),
    )
    assert failed.status == "completed"
    assert failed.tool_errors
    assert "RuntimeError" in failed.tool_errors[0]
    chunks = list(
        stream_text(
            ChatRequest(message="Where is TRK-2048?"),
            compiled_graph=build_graph(example_model(), tools=[lookup_shipment]),
        )
    )
    assert len(chunks) > 1
    assert "".join(chunks) == "Tool result: delayed at Lyon hub; revised ETA 2026-09-01"
