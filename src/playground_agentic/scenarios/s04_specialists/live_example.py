"""Route a launch proposal to relevant specialists, preserving disagreement."""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from playground_agentic.example_support import model_from_env, show, structured
from playground_agentic.scenarios.s04_specialists.graph import (
    SpecialistRequest,
    SpecialistResult,
    build_graph,
    run,
)


class SelectedSpecialists(BaseModel):
    specialists: list[Literal["product", "delivery", "security"]] = Field(
        min_length=1, max_length=3
    )


def run_example(model: BaseChatModel) -> SpecialistResult:
    def route(question: str) -> list[str]:
        return list(
            structured(
                model,
                SelectedSpecialists,
                "Select the expertise needed to review this proposal: product for customer value, "
                "delivery for rollout feasibility, security for access and data risk.",
                {"proposal": question},
            ).specialists
        )

    return run(
        SpecialistRequest(
            question=(
                "Launch a read-only invoice export beta to 20 customers next Friday. Customers "
                "requested it; a feature flag and rollback exist. CSV exports contain billing "
                "addresses, but tenant-isolation tests and download expiry are still missing. "
                "Should we launch, revise, or reject? Assess only your assigned specialty."
            )
        ),
        compiled_graph=build_graph(model, route),
    )


if __name__ == "__main__":
    show(run_example(model_from_env()))
