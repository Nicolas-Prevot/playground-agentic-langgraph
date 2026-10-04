# Persistence, memory, and subgraph scope

These concepts are related but not interchangeable.

| Need | LangGraph primitive | Primary scenario |
|---|---|---|
| Resume graph execution | Checkpointer + `thread_id` | 08, 10 |
| Multi-turn conversation state | Checkpointer | 03 |
| Cross-thread user/domain facts | `Store` with explicit namespace | 03 |
| Replay or fork prior state | Checkpoint history + `update_state` | 10 |
| Private worker state | Compiled subgraph | 04 |

Inject persistence rather than constructing it inside graph nodes:

```python
graph = build_graph(model=model, checkpointer=application_checkpointer)
result = graph.invoke(inputs, config={"configurable": {"thread_id": run_id}})
```

For subgraphs, choose the scope explicitly:

- `compile()` / `checkpointer=None`: fresh per invocation, inheriting the parent checkpointer for
  durable execution and interrupts. This is the normal specialist-worker choice.
- `checkpointer=True`: state persists across calls in the same thread. Use only for a genuinely
  multi-turn subagent.
- `checkpointer=False`: stateless, with no durable execution or interrupts.

Do not call every checkpoint “memory.” Scenario 08 checkpoint state is durable action state;
scenario 10 checkpoint state is job progress; only scenario 03 treats conversation and profile
data as user memory.

## Adoption checks

- Derive `thread_id` and Store namespaces from authenticated application context.
- Select a persistent backend for resumable work; in-memory implementations are test fixtures.
- Make nondeterministic calls and side effects idempotent because failed work can be replayed.
- Test close/reopen resume, deletion, namespace isolation, and schema evolution.

## Replay and lifetime

Use a new thread ID for each independent job/action. A repeated `invoke(new_input)` on an existing
thread adds to saved state; returning `[]` does not clear an additive reducer. Resume interrupted
approval with `Command(resume=...)`; resume failed work with `invoke(None, config=...)`.

For durability demonstrations use `graph.invoke(..., durability="sync")`, which waits for checkpoint
writes between steps. Select the production durability mode deliberately; checkpointing still does
not make an external effect exactly once. A node may execute again after a crash or interrupt.
Separate nondeterministic work into recoverable nodes/tasks and design effect providers for safe
retries with durable keys or reconciliation.

Never change the approved payload while retaining an effect key. Keep checkpoint schema and node
compatibility for suspended runs through deployments. Replays/forks may call models or tools again,
so review the effect boundary before using time travel.

Scenarios 08 and 10 have [two-process live demos](../running-examples.md). The SQLite effect ledger
in 08 simulates an effect entirely inside a transaction; this does not solve atomic delivery to
an unrelated external service.
