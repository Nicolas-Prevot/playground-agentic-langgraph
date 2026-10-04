"""Optional MCP implementation of the scenario's ``KnowledgeClient`` contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from playground_agentic.scenarios.s06_rag.graph import KnowledgeDocument


class McpKnowledgeClient:
    """Use a maintained LangChain MCP client without coupling MCP to the RAG graph."""

    def __init__(self, server_script: Path | None = None) -> None:
        script = server_script or Path(__file__).with_name("server.py")
        self.client = MultiServerMCPClient(
            {
                "knowledge": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(script)],
                }
            }
        )

    async def search(self, query: str) -> list[KnowledgeDocument]:
        tools = await self.client.get_tools()
        tool = next(item for item in tools if item.name == "search_knowledge")
        return _parse_documents(await tool.ainvoke({"query": query}))


def _parse_documents(raw: Any) -> list[KnowledgeDocument]:
    if isinstance(raw, str):
        payload = json.loads(raw)
    elif isinstance(raw, dict):
        payload = raw
    elif (
        isinstance(raw, list)
        and raw
        and isinstance(raw[0], dict)
        and isinstance(raw[0].get("text"), str)
    ):
        payload = json.loads(raw[0]["text"])
    else:
        raise TypeError(f"Unexpected MCP result type: {type(raw).__name__}")
    return [KnowledgeDocument.model_validate(item) for item in payload["documents"]]
