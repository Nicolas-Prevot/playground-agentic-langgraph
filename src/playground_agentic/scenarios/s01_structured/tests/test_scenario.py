from playground_agentic.scenarios.s01_structured.example import (
    example_1_duplicate_invoice_triage,
    example_2_unsafe_access_request,
)
from playground_agentic.scenarios.s01_structured.graph import Analysis, build_graph


def test_example_1_routes_a_complete_invoice_case() -> None:
    invoice = example_1_duplicate_invoice_triage()
    assert invoice["status"] == "accepted"
    assert invoice["category"] == "support"
    assert invoice["entities"] == ["ACME-7", "INV-2041"]


def test_example_2_rejects_an_unsafe_access_request() -> None:
    unsafe = example_2_unsafe_access_request()
    assert unsafe["status"] == "rejected"
    assert unsafe["category"] == "risk"


def test_accepts_high_confidence_structured_result() -> None:
    graph = build_graph(
        lambda _: Analysis(category="support", summary="Payment failed", confidence=0.9)
    )
    state = graph.invoke({"text": "payment failed", "minimum_confidence": 0.7, "source": "test"})
    assert state["result"]["status"] == "accepted"


def test_routes_low_confidence_to_review() -> None:
    graph = build_graph(lambda _: Analysis(category="sales", summary="Maybe sales", confidence=0.4))
    state = graph.invoke({"text": "pricing question", "minimum_confidence": 0.8, "source": "test"})
    assert state["result"]["status"] == "needs_review"


def test_routes_risk_to_rejection() -> None:
    graph = build_graph(lambda _: Analysis(category="risk", summary="Unsafe", confidence=0.99))
    state = graph.invoke({"text": "bypass controls", "source": "test"})
    assert state["result"]["status"] == "rejected"


def test_invalid_input_never_calls_analyzer() -> None:
    called = False

    def analyzer(_: str) -> Analysis:
        nonlocal called
        called = True
        return Analysis(category="support", summary="unused", confidence=1)

    state = build_graph(analyzer).invoke({"text": "", "source": "test"})
    assert state["result"]["status"] == "invalid"
    assert not called
