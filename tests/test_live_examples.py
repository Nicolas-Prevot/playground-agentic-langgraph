"""Exercise the live wiring offline. These fixtures do not evaluate model quality."""

import importlib
import json
import subprocess
import sys
from io import BytesIO, TextIOWrapper
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from playground_agentic.example_support import generate_text, model_from_env, show, structured
from playground_agentic.scenarios.s01_structured.graph import Analysis
from playground_agentic.testing import ScriptedChatModel


def demo_model() -> ScriptedChatModel:
    def respond(messages, tools):
        schema = next((item for item in tools if isinstance(item, type)), None)
        if schema:
            name = schema.__name__
            values = {
                "Analysis": {
                    "category": "support",
                    "summary": "Duplicate invoice",
                    "confidence": 0.9,
                },
                "SelectedSpecialists": {"specialists": ["product", "security"]},
                "Plan": {"tasks": [{"id": "dedup", "objective": "Define invoice deduplication"}]},
                "Review": {"approved": True, "feedback": "Covered"},
                "RelevantEvidence": {"ids": ["audit-retention"]},
                "GroundedAnswer": {
                    "answer": "Retain for 365 days.",
                    "cited_ids": ["audit-retention"],
                },
                "SearchQuery": {"query": "audit retention"},
                "Evaluation": {"score": 0.9, "feedback": "Facts retained"},
                "RefundPayload": {"order_id": "order-42", "amount": 25},
            }
            if name == "RelevantEvidence" and "vendors" in str(messages[-1].content):
                values[name] = {"ids": []}
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": values[name],
                        "id": "structured-result",
                    }
                ],
            )
        if any(getattr(item, "name", None) == "lookup_shipment" for item in tools):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content=f"Shipment: {messages[-1].content}")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_shipment",
                        "args": {"tracking_id": "TRK-2048"},
                        "id": "lookup",
                    }
                ],
            )
        return AIMessage(
            content="APPROVE: Scoped review.\nOutcome: Import invoices.\n"
            "Compatibility: API credentials required.\nValidation: Load test pending."
        )

    return ScriptedChatModel(responder=respond)


def test_live_config_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL"):
        model_from_env()


def test_blank_optional_dotenv_settings_use_provider_defaults(monkeypatch) -> None:
    captured = {}

    def construct(**settings):
        captured.update(settings)
        return demo_model()

    monkeypatch.setitem(sys.modules, "langchain_litellm", SimpleNamespace(ChatLiteLLM=construct))
    monkeypatch.setenv("LLM_MODEL", " openai/example-model ")
    monkeypatch.setenv("LLM_API_BASE", " ")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1024")
    model_from_env()
    assert captured["model"] == "openai/example-model"
    assert captured["api_base"] is None
    assert captured["api_key"] is None
    assert captured["max_tokens"] == 1024
    assert captured["streaming"] is False
    model_from_env(streaming=True)
    assert captured["streaming"] is True
    assert captured["stream_options"] == {"include_usage": True}


def test_structured_adapter_rejects_invalid_model_output() -> None:
    invalid = ScriptedChatModel(responder=lambda _m, _t: AIMessage(content="not a tool result"))
    with pytest.raises(ValueError, match="no Analysis tool result"):
        structured(invalid, Analysis, "Classify", {"text": "invoice"})


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_live_helpers_reject_truncated_model_output(finish_reason) -> None:
    model = ScriptedChatModel(
        responder=lambda _m, _t: AIMessage(
            content="Incomplete answer", response_metadata={"finish_reason": finish_reason}
        )
    )
    for helper in (
        lambda: generate_text(model, "Summarize", {"text": "invoice"}),
        lambda: structured(model, Analysis, "Classify", {"text": "invoice"}),
    ):
        with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
            helper()


def test_structured_adapter_preserves_schema_validation_failure() -> None:
    model = ScriptedChatModel(
        responder=lambda _m, _t: AIMessage(
            content="",
            tool_calls=[{"name": "Analysis", "args": {"category": "invented"}, "id": "invalid"}],
        )
    )
    with pytest.raises(ValueError, match="invalid Analysis structured output") as error:
        structured(model, Analysis, "Classify", {"text": "invoice"})
    assert error.value.__cause__ is not None


def test_cli_json_round_trips_unicode_through_an_ascii_pipe(monkeypatch) -> None:
    value = {"answer": "R\u00e9sum\u00e9 \u2192 \u4f60\u597d \U0001f680"}
    with TextIOWrapper(BytesIO(), encoding="ascii", write_through=True) as stream:
        with monkeypatch.context() as context:
            context.setattr(sys, "stdout", stream)
            show(value)
        stream.seek(0)
        assert json.load(stream) == value


@pytest.mark.parametrize(
    "scenario",
    [
        "s01_structured",
        "s02_tools",
        "s03_memory_chat",
        "s04_specialists",
        "s05_orchestrator",
        "s07_evaluator",
    ],
)
def test_live_example_runs_through_real_graph_with_injected_model(scenario) -> None:
    module = importlib.import_module(f"playground_agentic.scenarios.{scenario}.live_example")
    model = demo_model()
    result = module.run_example(model)
    assert result
    assert model.counter.value > 0
    if scenario == "s03_memory_chat":
        assert result["turns"][2]["profile"]["name"] == "Alex"
        assert result["turns"][3]["profile"] == {}


@pytest.mark.asyncio
async def test_live_rag_fixture_answers_and_abstains() -> None:
    from playground_agentic.scenarios.s06_rag.live_example import run_example

    results = await run_example(demo_model())
    assert results["supported"]["status"] == "answered"
    assert results["unsupported"]["status"] == "not_found"


def test_live_deep_worker_uses_same_injection_boundary() -> None:
    from playground_agentic.scenarios.s09_deep_worker.example import example_models
    from playground_agentic.scenarios.s09_deep_worker.live_example import run_example

    parent, researcher = example_models("runbook-review")
    original = parent.responder

    def respond(messages, tools):
        if "Analyze only the evidence" in str(messages[0].content):
            return researcher.responder(messages, tools)
        return original(messages, tools)

    parent.responder = respond
    result = run_example(parent)
    assert result.status == "completed"
    assert result.artifact.startswith("# ")
    assert result.model_calls == 4


def test_live_approval_resumes_without_model_and_rejects_second_resume(tmp_path: Path) -> None:
    from playground_agentic.scenarios.s08_hitl.live_example import run_example

    first = run_example(demo_model(), tmp_path)
    assert first["interrupted"] is True
    assert first["effects"] == 0
    resumed = run_example(None, tmp_path, decision="approve")
    assert resumed["result"]["status"] == "executed"
    assert resumed["effects"] == 1
    with pytest.raises(ValueError, match="No pending approval"):
        run_example(None, tmp_path, decision="approve")


def test_batch_resume_in_fresh_interpreter_reuses_completed_sibling(tmp_path: Path) -> None:
    # Separate processes eliminate Python closure/counter reuse as an explanation for resume.
    program = """
import json, sys
from pathlib import Path
from langchain_core.messages import AIMessage
from playground_agentic.testing import ScriptedChatModel
from playground_agentic.scenarios.s10_resumable.live_example import run_example
model = ScriptedChatModel(responder=lambda _m, _t: AIMessage(content='Fixture summary'))
print(json.dumps(run_example(model, Path(sys.argv[1]), resume=sys.argv[2] == 'resume')))
"""

    def invoke(phase):
        completed = subprocess.run(  # noqa: S603 -- fixed Python program and local fixture paths
            [sys.executable, "-c", program, str(tmp_path), phase],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return json.loads(completed.stdout)

    started = invoke("start")
    assert started["status"] == "simulated_crash"
    assert started["model_calls_this_process"] == ["policy"]
    resumed = invoke("resume")
    assert resumed["result"]["status"] == "completed"
    assert resumed["model_calls_this_process"] == ["runbook"]
