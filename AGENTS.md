@.agents/marin-style/AGENTS-core.md

# Slime (Marin fork)

This repository is `penfever/slime`, a fork of
[`THUDM/slime`](https://github.com/THUDM/slime). Keep the fork's changes small,
reviewable, and suitable for upstreaming unless a fork-only integration requires
otherwise.

The shared Marin standards above apply to work in this fork. Slime's established
architecture, test layout, and generated-workflow conventions remain authoritative
where they are more specific.

## Development

Use the shared lint entry point for Python changes:

```bash
infra/pre-commit.py --changed-files --fix
```

It runs Slime's pinned Ruff checks and Black formatter. The repository's
`.pre-commit-config.yaml` retains non-Python repository checks, and CI runs both.
Slime does not currently have a repository-wide type-checking gate, so the shared
kit's type-check is intentionally disabled.

Before adding or changing tests, read `TESTING.md`. CPU test files are normally
run directly, matching CI:

```bash
python tests/test_<feature>.py
```

New CPU test files must define `NUM_GPUS = 0`, include a `__main__` pytest entry
point, and be registered in `.github/workflows/pr-test.yml.j2`. Regenerate
`.github/workflows/pr-test.yml` with
`.github/workflows/generate_github_workflows.py` after changing its template.

## Repository map

- `slime/`: training, rollout, agent, and backend implementation.
- `slime_plugins/`: optional model and rollout extensions.
- `examples/`: runnable training and integration examples.
- `tests/`: CPU and accelerator-aware test programs.
- `docs/en/` and `docs/zh/`: English and Chinese documentation.
- `.claude/skills/`: Slime-specific and vendored shared agent procedures.

## Change policy

- Preserve user changes and unrelated work in a dirty tree.
- Prefer direct APIs and explicit branch conditions over thin wrappers.
- Do not edit `.agents/marin-style/` or the shared skills by hand; update the
  pinned `marin-style` revision in `infra/pre-commit.py` and run
  `marin-style sync`.
- Keep `.github/workflows/pr-test.yml.j2` as the source of truth for the generated
  PR workflow.
