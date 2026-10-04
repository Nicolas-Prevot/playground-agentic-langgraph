"""Run an answered and an evidence-insufficient knowledge example."""

from playground_agentic.scenarios.s06_rag.graph import (
    InMemoryKnowledgeClient,
    KnowledgeDocument,
    RagRequest,
    build_graph,
    run,
)


def _policy_client() -> InMemoryKnowledgeClient:
    return InMemoryKnowledgeClient(
        documents=[
            KnowledgeDocument(
                id="audit-retention-policy",
                text=(
                    "Production audit logs are retained for 365 days in immutable storage; "
                    "security investigations may place a legal hold before expiry."
                ),
            ),
            KnowledgeDocument(
                id="backup-policy",
                text="Backup restore tests run quarterly; stored copies expire after 35 days.",
            ),
        ]
    )


# Example 1: grounded answer with an allowed citation
async def example_1_audit_log_retention() -> dict[str, object]:
    """Answer a policy question only from cited internal evidence."""

    result = await run(
        RagRequest(question="How long are production audit logs retained?"),
        compiled_graph=build_graph(_policy_client()),
    )
    return result.model_dump()


# Example 2: explicit refusal when evidence is insufficient
async def example_2_unsupported_vendor_approval() -> dict[str, object]:
    """Return not-found when policy evidence cannot authorize a risky action."""

    result = await run(
        RagRequest(
            question="Can temporary vendors authorize emergency privilege elevation?",
            max_rewrites=1,
        ),
        compiled_graph=build_graph(_policy_client()),
    )
    return result.model_dump()


async def run_example() -> dict[str, object]:
    return {
        "audit_log_retention": await example_1_audit_log_retention(),
        "unsupported_vendor_approval": await example_2_unsupported_vendor_approval(),
    }


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(run_example()))
