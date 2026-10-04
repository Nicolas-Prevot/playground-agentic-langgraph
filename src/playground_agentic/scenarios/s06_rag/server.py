"""Small real MCP knowledge server used by Scenario 06's runnable example."""

from __future__ import annotations

import logging
import warnings

from mcp.server.fastmcp import FastMCP
from pydantic_settings.exceptions import IncompleteFieldDefinitionWarning

warnings.filterwarnings("ignore", category=IncompleteFieldDefinitionWarning)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

server = FastMCP("langgraph-knowledge")

DOCUMENTS = [
    {
        "id": "persistence",
        "text": (
            "LangGraph checkpointers save graph state by thread. Persistence enables memory, "
            "human-in-the-loop interrupts, time travel, and fault-tolerant resume."
        ),
    },
    {
        "id": "send",
        "text": "LangGraph Send dynamically creates worker tasks with isolated worker inputs.",
    },
    {
        "id": "subgraphs",
        "text": "Subgraphs can use per-invocation, per-thread, or stateless persistence modes.",
    },
]


@server.tool()
def search_knowledge(query: str) -> dict[str, object]:
    """Return knowledge documents sharing meaningful terms with the query."""

    terms = {part.strip(".,?!").lower() for part in query.split() if len(part) > 3}
    matches = [
        document
        for document in DOCUMENTS
        if terms.intersection(str(document["text"]).lower().replace("-", " ").split())
        or str(document["id"]) in query.lower()
    ]
    return {"documents": matches[:3]}


if __name__ == "__main__":
    server.run(transport="stdio")
