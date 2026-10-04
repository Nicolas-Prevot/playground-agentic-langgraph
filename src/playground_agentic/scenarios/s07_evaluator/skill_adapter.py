"""Optional Agent Skills loader for the evaluator-optimizer input boundary."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class SkillMetadata(BaseModel):
    name: str
    description: str


class SkillContext(BaseModel):
    name: str
    instructions: str
    resources: list[str]


class SkillRegistry:
    """Read an Agent Skills-compatible directory without changing graph topology."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load(self, name: str) -> SkillContext:
        directory = (self.root / name).resolve()
        if directory.parent != self.root or not directory.is_dir():
            raise ValueError("Skill path escapes the configured skills root")
        skill_path = directory / "SKILL.md"
        metadata = self._metadata(skill_path)
        if metadata.name != name:
            raise ValueError("Skill name and directory must match")
        resources: list[str] = []
        reference_root = (directory / "references").resolve()
        if reference_root.is_dir():
            for path in sorted(reference_root.rglob("*")):
                resolved = path.resolve()
                if path.is_file() and reference_root in resolved.parents:
                    resources.append(resolved.read_text(encoding="utf-8")[:20_000])
        return SkillContext(
            name=name,
            instructions=skill_path.read_text(encoding="utf-8"),
            resources=resources,
        )

    @staticmethod
    def _metadata(path: Path) -> SkillMetadata:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError(f"Skill has no YAML frontmatter: {path}")
        _, frontmatter, _body = text.split("---", 2)
        return SkillMetadata.model_validate(yaml.safe_load(frontmatter))
