---
name: slime-code-review-preferences
description: Use when reviewing or editing slime code, especially refactors around helper APIs, branch selection, argument validation, or recurring reviewer preferences about avoiding unnecessary wrappers and making control flow self-explanatory.
---

# Slime Code Review Preferences

Apply these lightweight review heuristics when changing slime code.

## Prefer Direct APIs Over Thin Wrappers

- Remove helper layers that only rename a call, format one path, or forward arguments without owning meaningful behavior.
- Prefer calling the concrete reusable API directly, for example a `*_to_path` helper when the caller already knows the destination path.
- Keep a wrapper only if it owns a real boundary: compatibility, validation, nontrivial error policy, lifecycle management, metrics/logging semantics, async/retry behavior, or cross-module ownership.
- Avoid moving a redundant wrapper's body into another file just to preserve the wrapper shape. Inline the simple call at the natural ownership site.
- When removing a wrapper, search for sibling wrappers and nearby helpers with `rg` and delete confirmed dead functions in the same pass.
- Treat single-use convenience functions as suspicious when their only job is path formatting plus forwarding. Prefer the caller owning that one line.

## Make Branches Explain Themselves

- Order conditionals by semantic precedence: special transport/lifecycle modes first, then explicit mode choices, then default paths.
- Prefer predicates that fully describe the branch, such as `mode == "full" and transport == "disk"`, over a broad predicate followed by an assert that explains what the branch really meant.
- Use asserts as invariants for impossible states after validation, not as a substitute for clear branch conditions.

## Keep Abstractions Honest

- Add an abstraction only when it removes real duplication, hides fragile mechanics, or clarifies ownership.
- When a review comment points out repeated indirection, look for a smaller public surface rather than adding another alias.
- Preserve existing behavior intentionally. If cleanup changes error handling, logging, or failure visibility, call that out in the final response.
