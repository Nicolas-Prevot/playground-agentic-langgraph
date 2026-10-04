"""Scenario 09: a Deep Agent constrained to one parent StateGraph worker node."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import FileData
from deepagents.backends.utils import create_file_data, file_data_to_string
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class DeepTaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4_000)
    artifact_path: Literal["/artifacts/result.md"] = "/artifacts/result.md"


class DeepTaskResult(BaseModel):
    status: Literal["completed", "invalid_artifact"]
    artifact_path: str
    artifact: str
    model_calls: int
    validation_error: str = ""


class DeepWorkerState(TypedDict, total=False):
    task: str
    artifact_path: str
    artifact: str
    model_calls: int
    status: str
    validation_error: str


def _skill_files() -> dict[str, Any]:
    path = Path(__file__).with_name("skills") / "artifact-author" / "SKILL.md"
    return {"/skills/artifact-author/SKILL.md": create_file_data(path.read_text(encoding="utf-8"))}


def build_graph(
    model: BaseChatModel,
    *,
    researcher_model: BaseChatModel,
):
    # Deep Agents accepts LangChain middleware here; its current invariant generic annotation
    # is narrower than the runtime contract.
    researcher = create_agent(
        researcher_model,
        tools=[],
        system_prompt="Analyze only the evidence supplied in the delegation. Never invent sources.",
        middleware=[ModelCallLimitMiddleware(run_limit=4, exit_behavior="error")],
    )
    # Compiled children receive exactly these capabilities. Declarative Deep Agent children
    # may receive built-in filesystem tools, even when tools=[] is supplied.
    researcher_spec: CompiledSubAgent = {
        "name": "researcher",
        "description": "Analyze one narrowly delegated evidence question.",
        "runnable": researcher,
    }
    bounded_default: CompiledSubAgent = {
        "name": "general-purpose",
        "description": "Evidence-only fallback with the same restrictions as the researcher.",
        "runnable": researcher,
    }
    deep_worker = create_deep_agent(
        model=model,
        system_prompt=(
            "Complete only the scoped task. Use the researcher once, read the artifact-author "
            "skill, and write exactly /artifacts/result.md. Do not write elsewhere."
        ),
        backend=StateBackend(),
        skills=["/skills/"],
        subagents=[researcher_spec, bounded_default],
        permissions=[
            FilesystemPermission(operations=["read"], paths=["/skills/**"], mode="allow"),
            FilesystemPermission(
                operations=["write"], paths=["/artifacts/result.md"], mode="allow"
            ),
            FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
        ],
        middleware=cast(
            Any,
            [
                ModelCallLimitMiddleware(run_limit=8, exit_behavior="error"),
                ToolCallLimitMiddleware(tool_name="task", run_limit=1, exit_behavior="error"),
            ],
        ),
        name="bounded_deep_worker",
    )

    def scope(state: DeepWorkerState) -> dict[str, object]:
        request = DeepTaskRequest.model_validate(state)
        return {"task": request.task, "artifact_path": request.artifact_path}

    def call_deep_agent(state: DeepWorkerState) -> dict[str, object]:
        result = deep_worker.invoke(
            cast(
                Any,
                {
                    "messages": [{"role": "user", "content": state["task"]}],
                    "files": _skill_files(),
                },
            )
        )
        files = result.get("files", {})
        file_data = files.get(state["artifact_path"])
        artifact = (
            file_data_to_string(cast(FileData, file_data)) if isinstance(file_data, dict) else ""
        )
        parent_calls = sum(isinstance(item, AIMessage) for item in result.get("messages", []))
        return {
            "artifact": artifact,
            "model_calls": parent_calls,
        }

    def validate(state: DeepWorkerState) -> dict[str, object]:
        artifact = state.get("artifact", "")
        required = ("# ", "## Evidence", "## Limits")
        missing = [heading for heading in required if heading not in artifact]
        if missing:
            return {
                "status": "invalid_artifact",
                "validation_error": f"Missing required sections: {missing}",
            }
        return {"status": "completed"}

    builder = StateGraph(DeepWorkerState)
    builder.add_node("scope", scope)
    builder.add_node("deep_agent", call_deep_agent)
    builder.add_node("validate", validate)
    builder.add_edge(START, "scope")
    builder.add_edge("scope", "deep_agent")
    builder.add_edge("deep_agent", "validate")
    builder.add_edge("validate", END)
    return builder.compile()


def run(request: DeepTaskRequest, *, compiled_graph) -> DeepTaskResult:
    state = compiled_graph.invoke(request.model_dump())
    return DeepTaskResult(
        status=state["status"],
        artifact_path=state["artifact_path"],
        artifact=state.get("artifact", ""),
        model_calls=state.get("model_calls", 0),
        validation_error=state.get("validation_error", ""),
    )
