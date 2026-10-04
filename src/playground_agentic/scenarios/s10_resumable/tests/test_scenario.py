from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from playground_agentic.scenarios.s10_resumable.example import (
    example_1_resumed_document_batch,
    example_2_mixed_import_batch,
)
from playground_agentic.scenarios.s10_resumable.graph import (
    JobRequest,
    WorkItem,
    WorkOutcome,
    build_graph,
    run,
)


def test_parallel_job_retries_and_isolates_permanent_failure() -> None:
    graph = build_graph()
    result = run(
        JobRequest(
            items=[
                WorkItem(id="ok", payload="hello"),
                WorkItem(id="retry", payload="retry-once then work"),
                WorkItem(id="bad", payload="fail permanently"),
            ],
            max_attempts=2,
        ),
        compiled_graph=graph,
    )
    by_id = {record.item_id: record for record in result.records}
    assert result.status == "partial"
    assert by_id["ok"].status == "completed"
    assert by_id["retry"].status == "completed"
    assert by_id["retry"].attempts == 2
    assert by_id["bad"].status == "failed"
    assert isinstance(graph, CompiledStateGraph)
    assert {"initialize", "worker", "assess", "finalize"} <= set(graph.nodes)


def test_all_failed_job_has_explicit_terminal_status() -> None:
    result = run(
        JobRequest(items=[WorkItem(id="bad", payload="fail")]),
        compiled_graph=build_graph(),
    )
    assert result.status == "failed"


def test_checkpoint_resume_reuses_successful_parallel_write() -> None:
    saver = InMemorySaver()
    calls: dict[str, int] = {}
    crash_once = {"enabled": True}

    def processor(item: WorkItem, _attempt: int) -> WorkOutcome:
        calls[item.id] = calls.get(item.id, 0) + 1
        if item.id == "unstable" and crash_once["enabled"]:
            crash_once["enabled"] = False
            raise RuntimeError("crash")
        return WorkOutcome(status="completed", output=item.payload)

    graph = build_graph(processor, checkpointer=saver)
    config = {"configurable": {"thread_id": "resume-test"}}
    request = JobRequest(
        items=[WorkItem(id="stable", payload="a"), WorkItem(id="unstable", payload="b")]
    )
    try:
        graph.invoke(request.model_dump(), config=config)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the first invocation should simulate a process failure")

    resumed = graph.invoke(None, config=config)
    assert resumed["status"] == "completed"
    assert calls["stable"] == 1
    assert calls["unstable"] == 2
    assert len(list(graph.get_state_history(config))) >= 2


def test_example_1_resumes_after_sqlite_reopen_without_duplicate_work() -> None:
    resumed = example_1_resumed_document_batch()
    assert resumed["result"]["status"] == "completed"
    assert resumed["processor_calls"] == {"stable": 1, "unstable": 2}


def test_example_2_reports_mixed_import_outcomes() -> None:
    mixed = example_2_mixed_import_batch()
    assert mixed["status"] == "partial"
    records = {record["item_id"]: record for record in mixed["records"]}
    assert records["product_manual"]["status"] == "completed"
    assert records["vendor_datasheet"]["attempts"] == 2
    assert records["corrupt_scan"]["status"] == "failed"


def test_checkpoint_can_fork_an_alternative_job_input() -> None:
    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "fork-test"}}
    graph.invoke(
        JobRequest(items=[WorkItem(id="original", payload="first")]).model_dump(),
        config=config,
    )
    before_initialize = next(
        snapshot for snapshot in graph.get_state_history(config) if snapshot.next == ("initialize",)
    )
    fork_config = graph.update_state(
        before_initialize.config,
        {"items": [WorkItem(id="alternative", payload="second").model_dump()]},
        as_node="__start__",
    )
    forked = graph.invoke(None, config=fork_config)
    assert forked["records"][0]["item_id"] == "alternative"
    assert forked["records"][0]["output"] == "SECOND"
