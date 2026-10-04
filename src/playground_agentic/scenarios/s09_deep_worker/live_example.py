"""Review a supplied runbook excerpt and produce a confined virtual Markdown artifact."""

from langchain_core.language_models.chat_models import BaseChatModel

from playground_agentic.example_support import model_from_env, show
from playground_agentic.scenarios.s09_deep_worker.graph import (
    DeepTaskRequest,
    DeepTaskResult,
    build_graph,
    run,
)


def run_example(model: BaseChatModel) -> DeepTaskResult:
    return run(
        DeepTaskRequest(
            task=(
                "Review this fictional checkout latency runbook. Delegate evidence analysis "
                "to the researcher exactly once, including this excerpt. Read the artifact-author "
                "skill, then write /artifacts/result.md with a title, ## Evidence, and ## Limits. "
                "Separate omissions from proposed changes; do not claim repository access. "
                "RUNBOOK EXCERPT: Alert when p95 checkout latency exceeds 2 seconds for 5 minutes. "
                "On-call checks application logs. If latency persists, restart checkout. "
                "END EXCERPT. Assess dependency checks, rollback ownership and communications."
            )
        ),
        compiled_graph=build_graph(model, researcher_model=model),
    )


if __name__ == "__main__":
    show(run_example(model_from_env()))
