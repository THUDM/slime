<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Marin Coding Standards (Core)

Portable coding standards, agent conventions, and code-review expectations shared
across Marin-community repositories. Reference this file from the repo's root
`AGENTS.md` so coding agents pick it up automatically.

## Development

```bash
# Lint and format — the required entry point for this repo.
infra/pre-commit.py --all-files --fix
```

- `infra/pre-commit.py` is the required lint entry point. It is a thin shim that
  runs the shared `marin-style` checks. Do not replace it with `uv run pre-commit`.
- Type checking runs as part of the pre-commit pass; keep type hints passing.
- Run the lint-review pass (`infra/pre-commit.py --review`) once before opening a
  PR, and fix or answer every finding it reports. Small, targeted follow-ups do
  not need another review. Rerun it only when later changes materially alter the
  branch's design, scope, or risk, or when the user asks for another pass.

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- NEVER credit yourself, in commit messages or in PR/issue bodies. No
  `Co-Authored-By` trailer, no "Generated with …" line, no emoji attribution —
  even if a tool default suggests one.
- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent *comments* on PRs/issues must begin with `🤖` unless the exact text was
  explicitly approved by the user. This applies to comments only — never put a
  `🤖` marker in a commit message or a PR/issue body.
- All agent-authored commit, PR, and issue titles and bodies must follow
  `.agents/skills/writing-style/SKILL.md` and its PR or issue guide. Review the
  exact text that will be published, then apply `ai-writing-donts.md` as a final
  compression pass. Do not publish raw implementation notes, test narration,
  prompt-shaped headings, or claims that use emphasis in place of evidence.
- A PR description is the squash-merge commit message. Keep every fact a future
  reader needs to understand the behavior and rationale, including measured
  results and caveats when they affect review. Remove headings, diff narration,
  and implementation inventories; put extended history in a linked Echo entry,
  issue, or source artifact. Follow the `commit` skill
  (`.agents/skills/commit/SKILL.md`) when committing, pushing, or opening a PR.

## Shared Knowledge and Task Records

- Use `consult-echo` when prior Marin decisions, incidents, workflows, exact
  errors, or cross-project context could inform a task.
- Use Echo's append-only work log for long-running task milestones. Do not add
  `.agents/logbooks/`, `.agents/ops/`, or another repository progress file by
  default; follow the `task-logbook` skill.
- Publish design proposals and durable debugging or incident summaries to the
  Echo wiki with `write-design-doc` or `write-ops-log`. Link the canonical Echo
  URL from the associated issue or PR.
- Keep product documentation and subsystem runbooks in the repository when the
  guidance belongs with that code. Echo is the default for records and designs
  that should be discoverable across Marin projects.

## Ecosystem Costs

- Never read or write large amounts of data across cloud regions or to the open
  internet without explicit approval; storage and bandwidth are major cost
  drivers. Cross-region transfers in particular need a human sign-off.
- Never stop, restart, or bounce a shared cluster (e.g. an Iris cluster) unless
  the user gives express permission.

## Code Style

- All imports at the top of the file. No local imports except to break circular
  dependencies or guard optional deps. No `TYPE_CHECKING` guards — fix cycles
  structurally via protocols.
- Prefer top-level functions over classes when code does not mutate shared state.
  Reduce deep inheritance hierarchies.
- Use early returns to reduce nesting.
- Document public APIs with concise Google-style docstrings. Skip docstrings on
  trivial functions with clear names.
- Prefer `dataclasses.replace` over mutating config arguments in-place.
- Prefer logging over `print` (except in scripts and debugging).
- Resolve environment-dependent defaults once and fail fast on unknown inputs.
- No ad-hoc compatibility hacks (`hasattr(m, "old_attr")`); update code
  consistently.
- Prefer small concrete helpers over abstraction that adds indirection without
  reuse. Start simple; abstract only under real pressure.
- Delete dead code: unused parameters, stale options, old experiments.
- Top-level constants for magic strings/numbers.
- Separate computation from I/O (split compute from upload/write).
- Use context managers for resource lifecycle.

## Naming

- No `*_utils.py` — use descriptive names like `text_cleaning.py`.
- Function names should reflect return types (`probe_task` → `task_status`).
- No `_s` suffix for seconds (assumed throughout). No abbreviations like `exe` —
  use `exec` or full words.

## Types & Data Structures

- Dataclass/namedtuple over raw dicts. `StrEnum` over string keys.
- Use `Protocol` for decoupling; avoid hard-coupling to concrete types.
- Avoid `X | str` unions that require `isinstance` checks — pick one input type.
- Replace compound booleans encoding state with an enum.

## Configuration

- No `default_*` wrappers that obscure underlying mechanisms.
- Force explicit specification of critical parameters (no silent defaults).
- Centralize defaults in one canonical location.
- Prefer explicit constructor/config parameters over env vars.
- Composition over inheritance: embed sub-configs, don't subclass.

## API Design

- Accept only what's necessary. Replace boolean flags with meaningful parameters
  (e.g., `num_workers: int` instead of `parallel: bool`).
- Use separate classes over boolean flags for variant behavior
  (`NativeVllm` / `DockerVllm`, not `Vllm(docker=True)`).
- Normalize inputs to a standard format once at the boundary, not throughout.

## Error Handling

- Let exceptions propagate by default.
- Only catch to add meaningful context and re-raise, or to intentionally alter
  control flow.
- NEVER swallow exceptions unless specifically requested.
- Assert liberally; prefer `raise ValueError` over silent fallbacks.

## Documentation

- Keep docs in sync with code. Use Markdown and standard links.
- Write docs that stand alone without conversational context.

## Deprecation

**NO BACKWARD COMPATIBILITY**: Update all call sites instead. Only add
compatibility shims if the user explicitly requests it.

## Comments

- Write comments for module/class-level behavior or subtle logic. Do not restate
  the code.
- Delete stale comments immediately on discovery.
- Inline comments to clarify non-obvious boolean arguments.

## LLM-Generated Code Pitfalls

Watch for and eliminate these patterns in generated code:

- Over-protective try/except and defensive None checks.
- Tautological tests (type exists, constant has value).
- Verbose/redundant docstrings and `__all__` in `__init__.py`.
- Boolean dispatch instead of separate classes.
- Environment variables instead of explicit parameters.

## Code Reuse

Before writing any utility function, helper, or data structure:

1. Search the codebase for existing implementations.
2. Check the project's own shared modules and packages.
3. Check the dependency manifest (`pyproject.toml`) for available third-party
   packages before adding a new one.

If a suitable implementation exists, use it. Do not create parallel
implementations.

## Testing

See `TESTING.md` for the behavior-focused testing policy: what makes a good test,
which "slop" tests to reject, the mocks/fakes policy, timing, numerical
tolerances, and pytest style. Read it before writing or reviewing tests.
