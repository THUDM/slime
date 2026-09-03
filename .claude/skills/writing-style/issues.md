<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Issues

Use this file for GitHub issues. Marin uses two issue styles with different expectations.

## Pick The Right Issue Type

- Use a standard OSS issue for code behavior, bugs, feature requests, and refactors.
- Use an experiment issue for learning that evolves over time.

## Standard OSS Issues

See also the [file-issue](../file-issue/SKILL.md) skill which focuses
more on what should be included rather than tone.

### Assume This Reader

- Assume the reader is a contributor or user who knows the repo but not this problem.

### Write This Way

- State the problem clearly and early.
- Include the smallest useful repro, logs, stack traces, or examples.
- Avoid unexplained shorthand.
- Keep the discussion focused and actionable.
- Use a direct, practical tone with no narrative padding.
- Keep the title at or below 80 characters. State a factual symptom for a bug
  and an imperative outcome for a task.
- Use as many words as needed for the symptom, impact, reproduction, expected
  behavior, and completion criteria. Remove anything that does not help a
  contributor understand or act on the issue.
- Avoid section headings in normal bug and task issues. Use plain `Reproduce:`,
  `Expected:`, or `Done when:` labels only when they make the issue easier to
  act on.
- Put extended investigation history in Echo's work log or a linked source
  artifact.
- Do not prescribe an implementation unless the choice is part of the task.

Before publishing, apply [ai-writing-donts.md](ai-writing-donts.md) to the exact
title and body. Delete any sentence that does not add a symptom, impact,
reproduction step, observation, expected behavior, or completion criterion.

### Example Shape

- Title: `[zephyr] Handle chunk sizes above 256`
- Body: `Chunks above 256 fail with <error>. Reproduce with <command>. Expected:
  the stage accepts the configured chunk size. <link> contains the relevant
  coordinator log.`

## Experiment Issues

Treat experiment issues as part of the scientific record. Use them as a summary layer, a coordination surface, and a long-lived artifact.

### Assume This Reader

- Assume the reader understands LLM systems broadly.
- Assume the reader was not present for the work.
- Assume the reader may revisit the issue much later.
- For issue comments, assume the reader has read the summary and recent comments, but not necessarily every comments in the thread.

### Keep The Key Principle

- Write so someone else can understand what you did, reproduce it, and evaluate it without extra context.

### Keep The Expected Structure

- Maintain a clear TL;DR and evolving conclusion.
- Link the Echo work log, W&B runs, commits, and tags.
- Track the decision log and negative results.
- Reflect current understanding instead of pasting raw logs.
- Treat the Echo work log as the detailed record and the issue as the
  interpreted summary.

### Write Updates This Way

- Answer four questions in each update: what changed, what was the result, how confident are you, and what is next.
- Keep updates concise, append-only, and link-heavy.
- Emphasize what changed, what was learned, and current belief.

### Keep The Tone

- Stay calm and factual.
- Be comfortable reporting negative results.
- Avoid hype and exaggerated reactions.

### Avoid These Failure Modes

- Do not let the issue become a chat log.
- Do not forget to update the TL;DR or conclusion.
- Do not lose the baseline.
- Do not overly clutter the issue. Echo should hold the full work log; the issue
  is the interpreted summary.
- Do not assume knowledge of the conversation thread or Echo work log. Each
  comment should be understandable in the context of the issue.
