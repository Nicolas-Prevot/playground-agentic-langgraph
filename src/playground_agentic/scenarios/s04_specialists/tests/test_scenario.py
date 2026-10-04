from playground_agentic.scenarios.s04_specialists.example import (
    example_1_customer_data_export,
    example_2_staged_ui_release,
    example_model,
)
from playground_agentic.scenarios.s04_specialists.graph import (
    SpecialistRequest,
    build_graph,
    run,
)


def test_example_1_fans_out_and_preserves_disagreement() -> None:
    customer_export = example_1_customer_data_export()
    assert customer_export["status"] == "needs_resolution"
    assert {item["specialist"] for item in customer_export["reports"]} == {
        "product",
        "delivery",
        "security",
    }


def test_example_2_routes_only_the_consensus_subset() -> None:
    staged_release = example_2_staged_ui_release()
    assert staged_release["status"] == "consensus"
    assert {item["specialist"] for item in staged_release["reports"]} == {
        "product",
        "delivery",
    }


def test_request_can_select_a_consensus_subset() -> None:
    result = run(
        SpecialistRequest(
            question="Can we stage the release?", specialists=["product", "delivery"]
        ),
        compiled_graph=build_graph(example_model(), lambda _: ["product", "delivery"]),
    )
    assert result.status == "consensus"
    assert len(result.reports) == 2


def test_disallowed_specialist_fails_closed() -> None:
    graph = build_graph(example_model(), lambda _: ["unknown"])
    try:
        graph.invoke({"question": "test", "reports": []})
    except ValueError as exc:
        assert "no allowed specialist" in str(exc)
    else:
        raise AssertionError("invalid specialist should fail")
