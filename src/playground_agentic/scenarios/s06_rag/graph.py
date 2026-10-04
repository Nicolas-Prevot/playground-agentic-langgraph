"""Scenario 06: grounded retrieval with explicit grading and rewrite routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    id: str
    text: str


class RagRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5_000)
    max_rewrites: int = Field(default=1, ge=0, le=3)


class GroundedAnswer(BaseModel):
    answer: str
    cited_ids: list[str]


class RagResult(BaseModel):
    status: Literal["answered", "not_found", "direct"]
    answer: str
    citations: list[str]
    rewrites: int


class RagState(TypedDict, total=False):
    question: str
    active_query: str
    max_rewrites: int
    rewrites: int
    route: str
    documents: list[dict[str, str]]
    relevant_documents: list[dict[str, str]]
    answer: str
    citations: list[str]
    grounded: bool
    status: str


class KnowledgeClient(Protocol):
    async def search(self, query: str) -> list[KnowledgeDocument]: ...


class InMemoryKnowledgeClient:
    def __init__(self, documents: list[KnowledgeDocument] | None = None) -> None:
        self.documents = (
            documents
            if documents is not None
            else [
                KnowledgeDocument(
                    id="persistence",
                    text=(
                        "LangGraph persistence enables memory, human-in-the-loop, time travel, "
                        "and fault-tolerant resume through checkpoints."
                    ),
                ),
                KnowledgeDocument(
                    id="send", text="LangGraph Send creates dynamic parallel worker tasks."
                ),
            ]
        )

    async def search(self, query: str) -> list[KnowledgeDocument]:
        terms = {part.strip(".,?!").lower() for part in query.split() if len(part) > 3}
        return [
            item
            for item in self.documents
            if terms.intersection(item.text.lower().replace("-", " ").split())
            or item.id in query.lower()
        ]


Router = Callable[[str], Literal["direct", "knowledge"]]
Grader = Callable[[str, list[KnowledgeDocument]], list[str]]
Generator = Callable[[str, list[KnowledgeDocument]], GroundedAnswer]
Rewriter = Callable[[str, str, int], str]


def offline_router(question: str) -> Literal["direct", "knowledge"]:
    return "direct" if question.lower().strip() in {"hello", "hi", "thanks"} else "knowledge"


def offline_grader(question: str, documents: list[KnowledgeDocument]) -> list[str]:
    terms = {part.strip(".,?!").lower() for part in question.split() if len(part) > 3}
    return [item.id for item in documents if terms.intersection(item.text.lower().split())]


def offline_generator(question: str, documents: list[KnowledgeDocument]) -> GroundedAnswer:
    del question
    citations = [item.id for item in documents]
    answer = " ".join(item.text for item in documents)
    return GroundedAnswer(answer=answer, cited_ids=citations)


def offline_rewriter(question: str, _previous: str, _attempt: int) -> str:
    return f"LangGraph {question}"


def build_graph(
    client: KnowledgeClient | None = None,
    router: Router = offline_router,
    grader: Grader = offline_grader,
    generator: Generator = offline_generator,
    rewriter: Rewriter = offline_rewriter,
):
    client = client or InMemoryKnowledgeClient()

    def route(state: RagState) -> dict[str, object]:
        request = RagRequest.model_validate(state)
        selected = router(request.question)
        if selected not in {"direct", "knowledge"}:
            raise ValueError("Router must select direct or knowledge")
        return {
            "active_query": request.question,
            "max_rewrites": request.max_rewrites,
            "rewrites": 0,
            "route": selected,
        }

    def route_query(state: RagState) -> Literal["direct", "retrieve"]:
        return "direct" if state["route"] == "direct" else "retrieve"

    def direct(state: RagState) -> dict[str, object]:
        del state
        return {
            "status": "direct",
            "answer": "Hello! Ask me about the knowledge base.",
            "citations": [],
        }

    async def retrieve(state: RagState) -> dict[str, object]:
        documents = await client.search(state["active_query"])
        return {"documents": [item.model_dump() for item in documents]}

    def grade(state: RagState) -> dict[str, object]:
        documents = [KnowledgeDocument.model_validate(item) for item in state["documents"]]
        relevant_ids = set(grader(state["question"], documents))
        relevant = [item.model_dump() for item in documents if item.id in relevant_ids]
        return {"relevant_documents": relevant}

    def route_grade(state: RagState) -> Literal["generate", "rewrite", "not_found"]:
        if state["relevant_documents"]:
            return "generate"
        return "rewrite" if state["rewrites"] < state["max_rewrites"] else "not_found"

    def rewrite(state: RagState) -> dict[str, object]:
        attempt = state["rewrites"] + 1
        query = rewriter(state["question"], state["active_query"], attempt)
        if not query.strip():
            raise ValueError("Rewriter returned an empty query")
        return {
            "active_query": query,
            "rewrites": attempt,
        }

    def generate(state: RagState) -> dict[str, object]:
        documents = [KnowledgeDocument.model_validate(item) for item in state["relevant_documents"]]
        generated = generator(state["question"], documents)
        available = {item.id for item in documents}
        # Citation membership is a structural gate, not proof of semantic entailment.
        grounded = (
            bool(generated.answer.strip())
            and bool(generated.cited_ids)
            and set(generated.cited_ids) <= available
        )
        return {
            "answer": generated.answer,
            "citations": generated.cited_ids,
            "grounded": grounded,
        }

    def route_grounding(state: RagState) -> Literal["answered", "not_found"]:
        return "answered" if state["grounded"] else "not_found"

    def answered(state: RagState) -> dict[str, object]:
        del state
        return {"status": "answered"}

    def not_found(state: RagState) -> dict[str, object]:
        return {
            "status": "not_found",
            "answer": "The available evidence is insufficient for a grounded answer.",
            "citations": [],
        }

    builder = StateGraph(RagState)
    builder.add_node("route", route)
    builder.add_node("direct", direct)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_node("answered", answered)
    builder.add_node("not_found", not_found)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", route_query)
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_grade)
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges("generate", route_grounding)
    builder.add_edge("direct", END)
    builder.add_edge("answered", END)
    builder.add_edge("not_found", END)
    return builder.compile()


async def run(request: RagRequest, *, compiled_graph) -> RagResult:
    state = await compiled_graph.ainvoke(request.model_dump())
    return RagResult(
        status=state["status"],
        answer=state["answer"],
        citations=state.get("citations", []),
        rewrites=state["rewrites"],
    )
