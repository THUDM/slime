<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Reports

Use this file for technical reports and research writeups.

## Assume This Reader

- Assume the reader is a peer researcher working in the same general area.
- Do not assume the reader uses Marin or cares about Marin as a codebase.
- Assume the reader cares about the result, the experimental setup, and the high-level methodology.
- Do not assume familiarity with your specific method or terminology.

## Optimize For

- Give enough detail for someone to reproduce, critique, or extend the work in their own stack.
- Make the main result and methodological choice legible without forcing the reader through Marin-specific context.
- Preserve first-read clarity instead of chasing exhaustive completeness.

## Write This Way

- Lead with the result, then show how you got there, then state caveats.
- Include a doc-level TL;DR near the top.
- Add section-level takeaway lines when sections are long enough that readers need orientation.
- Define non-standard or ambiguous terms briefly before relying on them.
- Include concrete setup details, metrics, and comparison points.
- Describe methodology at the level needed for a peer to evaluate the work, not at the level of Marin implementation internals unless those internals matter to the result.
- Remove repo-local detail that does not help an outside researcher interpret the result.
- Link to repo internals when they help with reproduction or provide useful context (similar to an appendix in a paper).
- Clearly separate confirmed results from exploratory findings (e.g., label as exploratory, replicated, or stable).
- Cite the prior work that the report builds on or compares against, including relevant Marin experiments and external papers where appropriate.

## Test

- Ask whether a researcher in another codebase or organization could evaluate the work without outside context.
