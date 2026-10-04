# Agent Skills

Use a Skill for reusable, on-demand instructions or domain knowledge—not as a routing mechanism or
authorization boundary.

An Agent Skills-compatible directory contains:

```text
skills/
└── release-notes/
    ├── SKILL.md
    └── references/
        └── CHECKLIST.md
```

Scenario 07 demonstrates a read-only adapter: the application loads a selected Skill and passes
its instructions/resources into the evaluator workflow's input contract. The graph remains a
normal evaluator-optimizer when no Skill is used.

Scenario 09 demonstrates native Deep Agent Skills with progressive disclosure. The skill files
are seeded into a virtual `StateBackend`, and the Deep Agent receives only `/skills/` as a source.

## Adoption checks

- Validate `SKILL.md` frontmatter and keep the directory name aligned with `name`.
- Resolve paths under an allowlisted root; reject traversal.
- Pin and review Skills as code, especially scripts and linked resources.
- Scope Skills per bounded agent or subagent; do not inject an entire organization catalog.
- Keep tool authorization separate. Instructions cannot grant access.

