import pytest

from playground_agentic.scenarios.s06_rag.example import (
    example_1_audit_log_retention,
    example_2_unsupported_vendor_approval,
)
from playground_agentic.scenarios.s06_rag.graph import (
    GroundedAnswer,
    InMemoryKnowledgeClient,
    RagRequest,
    build_graph,
    run,
)
from playground_agentic.scenarios.s06_rag.mcp_adapter import McpKnowledgeClient


@pytest.mark.asyncio
async def test_example_1_returns_a_grounded_policy_citation() -> None:
    retention = await example_1_audit_log_retention()
    assert retention["status"] == "answered"
    assert retention["citations"] == ["audit-retention-policy"]
    assert "365 days" in retention["answer"]


@pytest.mark.asyncio
async def test_example_2_refuses_when_policy_evidence_is_missing() -> None:
    unsupported = await example_2_unsupported_vendor_approval()
    assert unsupported["status"] == "not_found"
    assert unsupported["citations"] == []
    assert unsupported["rewrites"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_optional_mcp_adapter_satisfies_the_same_retriever_contract() -> None:
    result = await run(
        RagRequest(question="What does LangGraph persistence enable?"),
        compiled_graph=build_graph(McpKnowledgeClient()),
    )
    assert result.status == "answered"
    assert "persistence" in result.citations


@pytest.mark.asyncio
async def test_direct_route_skips_retrieval() -> None:
    result = await run(RagRequest(question="hello"), compiled_graph=build_graph())
    assert result.status == "direct"
    assert result.citations == []


@pytest.mark.asyncio
async def test_missing_evidence_stops_after_rewrite_budget() -> None:
    result = await run(
        RagRequest(question="unknown topic", max_rewrites=1),
        compiled_graph=build_graph(InMemoryKnowledgeClient(documents=[])),
    )
    assert result.status == "not_found"
    assert result.rewrites == 1


@pytest.mark.asyncio
async def test_ungrounded_generation_is_rejected() -> None:
    graph = build_graph(
        generator=lambda _question, _documents: GroundedAnswer(
            answer="unsupported", cited_ids=["invented"]
        )
    )
    result = await run(RagRequest(question="What does persistence enable?"), compiled_graph=graph)
    assert result.status == "not_found"
