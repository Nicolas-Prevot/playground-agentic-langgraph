"""Run two realistic structured-intake examples."""

from __future__ import annotations

from typing import Any

from playground_agentic.scenarios.s01_structured.graph import (
    Analysis,
    build_graph,
)


def _example_analyzer(text: str) -> Analysis:
    if "bypass SSO" in text:
        return Analysis(
            category="risk",
            summary="Request attempts to bypass authentication and export secrets.",
            confidence=0.99,
            entities=["ACME-7"],
        )
    return Analysis(
        category="support",
        summary="Tenant reports a duplicate invoice charge after a payment retry.",
        confidence=0.96,
        entities=["ACME-7", "INV-2041"],
    )


# Example 1: accepted support intake
def example_1_duplicate_invoice_triage() -> dict[str, object]:
    """Route a complete billing intake to support with its identifiers preserved."""

    state = build_graph(_example_analyzer).invoke(
        {
            "text": (
                "Tenant ACME-7 reports that invoice INV-2041 was charged twice after a retry; "
                "route the case to the correct queue."
            ),
            "source": "support-form",
            "minimum_confidence": 0.7,
        }
    )
    return state["result"]


# Example 2: rejected risk intake
def example_2_unsafe_access_request() -> dict[str, object]:
    """Reject a high-confidence request to bypass authentication controls."""

    state = build_graph(_example_analyzer).invoke(
        {
            "text": (
                "A caller asks support to bypass SSO and export customer secrets for tenant ACME-7."
            ),
            "source": "support-form",
            "minimum_confidence": 0.7,
        }
    )
    return state["result"]


def run_example() -> dict[str, Any]:
    return {
        "duplicate_invoice_triage": example_1_duplicate_invoice_triage(),
        "unsafe_access_request": example_2_unsafe_access_request(),
    }


if __name__ == "__main__":
    print(run_example())
