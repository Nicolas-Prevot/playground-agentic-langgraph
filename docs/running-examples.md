# Run the scenarios

For traces, logs, metrics, and a local Grafana dashboard, see the optional
[scenario 10 observability lab](observability/README.md). It supports fixtures and real model calls.

Every scenario has two deterministic demonstrations in `example.py` and a model-backed
demonstration in `live_example.py`. The latter uses the same graph builder, Pydantic contracts,
and normal LangChain model interface. Only models require an endpoint. Tools, documents,
business services, and workspaces are local fixtures. Nothing sends real email or refunds money.

## Install

From the repository root, with Python 3.11+ and uv:

```sh
uv sync --frozen --all-extras
uv run python -m playground_agentic.scenarios.s01_structured.example
uv run pytest
```

Each offline module can be run the same way, replacing `s01_structured` with the directory name
in the table below. No credentials are used by offline examples or the default tests. `--all-extras`
also installs SQLite, MCP, Deep Agents, provider adapters, and developer tools; there are no external
database or MCP services to start. The MCP integration test starts its own local stdio fixture.

## Configure a model

Copy [.env.example](../.env.example) to `.env` in the repository root, then edit `.env` with your
model settings. The live examples use LiteLLM provider/model identifiers and an optional custom
base URL. See the official
[LangChain integration](https://docs.langchain.com/oss/python/integrations/chat/litellm).

For example, a local OpenAI-compatible endpoint would use these `.env` entries:

```dotenv
LLM_MODEL=openai/<model-served-by-your-endpoint>
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=not-required
LLM_MAX_TOKENS=2048
```

Replace placeholders. `not-required` is only for a local server without authentication. For a
hosted provider, leave `LLM_API_BASE` empty and set `LLM_API_KEY` or its standard provider variable
(such as `OPENAI_API_KEY`) in `.env`. Empty optional values use the provider defaults. The committed
template has no credentials or default model; `.env` and other local `.env.*` variants are ignored
by Git, while `.env.example` is retained.

Load the file when launching a live example:

```sh
uv run --env-file .env python -m playground_agentic.scenarios.s01_structured.live_example
```

`uv` loads `.env` before starting Python, so the same command works across operating systems.
The graph modules do not load files or alter global environment settings. Your application may
instead inject a configured model directly.

The endpoint must support chat, tool calling, and structured output through function calling.
An API-compatible HTTP surface alone does not establish those capabilities. The demonstrations
use `with_structured_output(..., method="function_calling")`; malformed output and unsupported
capabilities fail visibly. They never replace failed model calls with canned answers. See
[LangChain model capabilities](https://docs.langchain.com/oss/python/langchain/models).

The structured-output helper reports missing tool results and schema errors explicitly. Text
and structured helpers also reject responses marked as token-truncated. Reasoning models may
spend part of `LLM_MAX_TOKENS` before producing their visible answer; size the per-call cap for
your model and task. A missing tool result can also mean unsupported function calling or model
noncompliance, so raising the cap is not a universal fix. These failures are not silently retried.
An agent that fails to write its required artifact returns `invalid_artifact` with a diagnostic;
inspect statuses and content, not just the process exit code. A successful live run is a smoke
check, not evidence that stochastic model behavior will always satisfy the contract.

Calls have a 60-second provider timeout, at most one provider retry, and an output-token cap.
These are per-call settings, not a global wall-time or spending limit. Scenarios with agents may
use several calls. Run scenarios individually, starting with 01 and 02. No test contacts a live
endpoint by default. `example_support.py` is demo wiring, not a production provider framework.
To use a different LangChain integration, import a live module and pass your `BaseChatModel`
directly to its `run_example(model)` function.

## What to run and inspect

For 01–07 and 09, run
`uv run --env-file .env python -m playground_agentic.scenarios.<directory>.live_example`.
Commands for 08 and 10 follow the table. Output is readable JSON and model wording may vary.

| ID / directory | Meaningful task | Model work | Local simulation | Inspect this outcome |
|---|---|---|---|---|
| 01 `s01_structured` | Classify a duplicate-invoice ticket | Typed category, summary, confidence | Fixed intake | Accepted or review route; faithful invoice summary |
| 02 `s02_tools` | Shipment status plus proportional service credit | Choose tools and arguments, explain results | Shipment fixture and real arithmetic | Lyon delay and EUR 6.72 credit; no invented lookup result |
| 03 `s03_memory_chat` | Remember a support-workflow discussion and user preferences | Four chat turns | In-memory checkpointer and Store | Same-thread topic recall, cross-thread profile, empty second-user profile |
| 04 `s04_specialists` | Review a billing-export beta with missing isolation tests | Choose specialists and produce scoped reports | Product/delivery/security roles | Selected experts and preserved disagreement; consensus is not correctness |
| 05 `s05_orchestrator` | Plan a CSV-to-API invoice migration | Plan, draft independent workstreams, review/replan | Supplied API facts | Deduplication, rate limits, reconciliation and rollback; at most two rounds |
| 06 `s06_rag` | Answer retention policy and an absent vendor SLA question | Relevance, query rewrite, cited generation | Two-document lexical search | Supported answer cites `audit-retention`; absent SLA returns `not_found` |
| 07 `s07_evaluator` | Write release notes from migration facts | Generate and evaluate fidelity | Deterministic headings/length checks | Staging evidence distinguished from pending load tests; accepted or partial |
| 08 `s08_hitl` | Propose and review a EUR 25 duplicate-charge refund | Extract refund proposal once | SQLite checkpoint and atomic fake effect ledger | Zero effects before approval; one after a separate resume process |
| 09 `s09_deep_worker` | Review a supplied incident runbook excerpt | Bounded delegation and virtual file writing | StateBackend, local Skill, no external research | Markdown evidence/limits artifact; at most one delegation |
| 10 `s10_resumable` | Summarize two documents across a worker failure | Summarize each item | SQLite checkpoint and intentional second-item crash | Fresh process only calls the model for the unfinished item |

## Approval across two processes (08)

```sh
uv run --env-file .env python -m playground_agentic.scenarios.s08_hitl.live_example prepare
uv run python -m playground_agentic.scenarios.s08_hitl.live_example resume --decision approve
```

Read the first command's proposal before running the second. To decline it, use
`resume --decision reject`. Resume defaults to rejection and does not require a model endpoint.
To repeat, give both commands a new `--directory .demo-state/approval-2`.
The scripts refuse to overwrite an existing run or resume one without pending approval.

The reviewer identity is a clearly labeled fixture. Production authentication must come from the
server. SQLite records only a fake refund; the transaction can atomically record the entire simulated
effect. An actual payment service requires its own durable idempotency and reconciliation contract.
The graph also supports typed payload edits; deterministic tests exercise edits and policy denial.

## Crash and resume across two processes (10)

```sh
uv run --env-file .env python -m playground_agentic.scenarios.s10_resumable.live_example start
uv run --env-file .env python -m playground_agentic.scenarios.s10_resumable.live_example resume
```

`start` reports `simulated_crash` after summarizing `policy`. `resume` should report completed
records with `model_calls_this_process: ["runbook"]`. Both commands use a checkpoint directory
that survives process exit. Give both commands a new `--directory .demo-state/batch-2` to repeat.
Concurrency is one in this demo to make the partial-progress observation reproducible; the graph
still uses `Send`, and other tests cover parallel execution. A worker exception is simulated,
not an operating-system kill or database outage. Real provider failures are not swallowed.

## Verification and interpretation

```sh
uv run ruff check .
uv run pyright
uv run pytest
```

Default tests execute all ten live wiring paths with injected scripted models, including actual
schema parsing, actual tool execution, and SQLite resume in fresh Python processes. They prove
contracts and control flow, not live-model accuracy. When adapting a scenario, inspect model-backed
runs against the outcomes above and your own domain cases. Compare facts and behavior rather than
identical generated wording. The [evaluation recipe](recipes/evaluation.md) suggests useful checks.
