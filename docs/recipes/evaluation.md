# Evaluation without a playground evaluator

Evaluation belongs to the destination task and dataset. The catalog therefore exposes stable
contracts and inspectable trajectories instead of inventing a generic scoring platform.

Use four layers:

1. **Contract tests:** Pydantic input/output validation and explicit terminal statuses.
2. **Graph invariant tests:** selected route, maximum iterations, allowed tools, citations,
   interrupts, idempotency, and resume behavior.
3. **Task evaluation:** domain examples with expected facts, rubric scores, or human labels.
4. **Operational evaluation:** latency, token/tool counts, failures, approval rate, and recovery.

| Scenario | Important evaluation dimensions |
|---|---|
| 01 | classification accuracy, confidence calibration, invalid/review rate |
| 02 | tool selection, argument validity, unnecessary calls, budget termination |
| 03 | answer quality, memory precision/recall, isolation, deletion, poisoning |
| 04 | routing recall, specialist usefulness, conflict preservation, call cost |
| 05 | plan coverage, duplicate/dependent tasks, synthesis quality, replan rate |
| 06 | retrieval recall, evidence relevance, citation precision, unsupported claims |
| 07 | rubric improvement per iteration, evaluator agreement, cost versus gain |
| 08 | policy decisions, reviewer edits, duplicate-effect prevention, recovery |
| 09 | artifact contract, forbidden access attempts, delegation/call budget |
| 10 | item accuracy, retry classification, partial result, no duplicate completed work |

The `run` helpers return Pydantic results, so a destination project can use plain `pytest`,
LangSmith, or another maintained evaluation system without adapting a playground report format.
When trajectory detail is needed, inspect graph state history or consume `updates`/`tasks` streams.

Keep deterministic fakes for topology and failure tests. Use real model runs only for capabilities
that deterministic tests cannot measure, and record the exact endpoint, model identifier, prompt,
dataset version, and evaluator configuration.

