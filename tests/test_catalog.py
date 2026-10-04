import re
from pathlib import Path
from urllib.parse import unquote, urlparse

SCENARIOS = (
    "s01_structured",
    "s02_tools",
    "s03_memory_chat",
    "s04_specialists",
    "s05_orchestrator",
    "s06_rag",
    "s07_evaluator",
    "s08_hitl",
    "s09_deep_worker",
    "s10_resumable",
)

REQUIRED_SECTIONS = (
    "## Problem",
    "## Choose this when",
    "## Avoid when",
    "## Graph",
    "## State and contracts",
    "## Nodes",
    "## Features and decisions",
    "## Examples",
    "## Advantages and limits",
    "## Copy into a project",
    "## Tests",
)


def test_catalog_has_exactly_ten_documented_scenarios() -> None:
    root = Path("src/playground_agentic/scenarios")
    discovered = tuple(
        path.name for path in sorted(root.glob("s[0-9][0-9]_*")) if (path / "graph.py").is_file()
    )
    assert discovered == SCENARIOS
    for name in SCENARIOS:
        scenario = root / name
        assert (scenario / "graph.py").is_file()
        assert (scenario / "example.py").is_file()
        assert (scenario / "live_example.py").is_file()
        assert (scenario / "tests/test_scenario.py").is_file()
        example_source = (scenario / "example.py").read_text(encoding="utf-8")
        example_numbers = re.findall(
            r"^(?:async )?def example_([12])_", example_source, re.MULTILINE
        )
        assert example_numbers == ["1", "2"]
        readme = (scenario / "README.md").read_text(encoding="utf-8")
        assert "```mermaid" in readme
        assert "flowchart" in readme
        assert readme.count("```") % 2 == 0
        assert all(section in readme for section in REQUIRED_SECTIONS)


def test_root_catalog_has_problem_and_feature_views() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## Choose a scenario" in readme
    assert "## Feature matrix" in readme
    assert "## Feature-to-scenario index" in readme
    assert "s10_resumable" in readme


def test_cross_cutting_recipes_cover_optional_features() -> None:
    recipe_root = Path("docs/recipes")
    expected = {
        "models.md",
        "persistence.md",
        "mcp.md",
        "skills.md",
        "middleware.md",
        "resilience.md",
        "evaluation.md",
        "observability-security.md",
    }
    assert expected <= {path.name for path in recipe_root.glob("*.md")}


def test_local_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    markdown_files = [Path("README.md"), *Path("docs").rglob("*.md"), *Path("src").rglob("*.md")]
    for markdown in markdown_files:
        for raw_target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            target = raw_target.strip("<>").split("#", 1)[0]
            if not target or urlparse(target).scheme:
                continue
            resolved = (markdown.parent / unquote(target)).resolve()
            assert resolved.exists(), f"Broken link in {markdown}: {raw_target}"
