---
name: writing-style
description: Marin house writing style. Use when drafting or revising Marin-authored prose, including commit messages and GitHub PR, issue, or comment text.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Marin House Style

Start here for any non-trivial Marin-authored text. This file is the common layer; then read the medium-specific file that matches the deliverable.

## Read The Right File

- Read [blog-posts.md](blog-posts.md) for public explainers and blog posts.
- Read [reports.md](reports.md) for technical reports aimed at peer researchers outside Marin.
- Read [tutorials.md](tutorials.md) for learning-oriented documentation that introduces Marin.
- Read [reference-docs.md](reference-docs.md) for precise usage docs aimed at readers who already know Marin.
- Read [issues.md](issues.md) for standard OSS issues and experiment issues.
- Read [pull-requests.md](pull-requests.md) for commit messages and PR titles and bodies.
- Read [discord.md](discord.md) for Discord summaries and tactical replies.
- Read [ai-writing-donts.md](ai-writing-donts.md) for the final prose-only review pass that strips generic AI-writing patterns.
- Apply this file first, then apply the medium-specific file. If a piece spans multiple media, keep the stricter rule.

## Hold The Marin Positioning

- Write from the stance of a rigorous, open-science lab building frontier-level foundation models.
- Treat process as part of the work. Experiments, decisions, and mistakes all belong in the record.
- Let the work speak. Do not substitute tone for evidence.

## Keep The Core Vibe

- Be sober, not flashy.
- Project quiet confidence.
- Keep an open door, not a megaphone.
- Stay practical and hands-on.
- Aim to be helpful and respectful.
- Assume a baseline familiarity with ML systems.
- Don’t dilute discussions to accommodate every level of experience. (Different media will be intended for different levels of experience.)
- Don’t be overly formal; write like a technical peer, not an academic paper or a product blog.
- Use technical language where it helps, but keep the tone natural and direct.

## On Voice

The above defines the default voice; we allow some flexibility by author. These rules are stricter for agent-written text than for human-written text.

Researchers need not suppress their own voice. Some personality is fine, especially in retrospectives, blog posts, and narrative writeups, as long as the writing stays concrete, honest, and technically disciplined.

Agents should err toward more discipline. When reviewing human prose, allow deviations and flourishes that do not conflict with core Marin values.

## Enforce Hard Rules

- Remove hype, marketing copy, and launch-tweet energy.
- Use emoji very sparingly: for communication, not flair.
- Remove grand claims that outrun the evidence.
- Avoid AGI rhetoric and speculation.
- Avoid adjectives that try to do the work of numbers.
- If a sentence sounds like a product announcement, rewrite it.

## Follow The Writing Principles

- Show results instead of claiming them. Prefer concrete numbers, examples, and observed behavior.
- Prefer the simplest framing that is still correct.
- State uncertainty plainly. Use phrases like `we think`, `preliminary results suggest`, or `this seems to break down when...` when warranted.
- Treat the reader as a capable collaborator. Do not condescend or over-explain basics.
- Default to transparency. Include what worked, what failed, what surprised you, and what you would try next.
- Cite the relevant prior work and artifacts. This includes Marin experiments, reports, issues, PRs, papers, and other external work when they materially inform the piece.

## Add Structure For Longer-Form Writing

- For longer-form writing, include an easy-to-scan doc-level TL;DR near the top.
- For longer-form writing, add section-level takeaway lines or short TL;DRs where they help readers navigate.

- Do not force these patterns into short-form media that do not benefit from them.

## Set Audience By Default

- Assume the reader works in ML or LLMs unless the medium says otherwise.
- Do not assume deep specialization by default.
- Introduce non-standard terms before using them heavily.
- Add detail only when it helps the reader reproduce, evaluate, or act.

## Use Sentence-Level Defaults

- Prefer short, direct sentences.
- Lead with the result or takeaway.
- Follow with method or explanation.
- End with caveats, limits, or open questions when needed.
- Prefer phrases like `we found`, `this suggests`, `in practice`, and `one limitation is`.
- Avoid phrases like `clearly`, `obviously`, `groundbreaking`, and `state-of-the-art` unless you define and defend them.

## Review For AI-Writing Tells

Do one editing pass that looks only for generic, over-smoothed, LLM-sounding prose. Then apply [ai-writing-donts.md](ai-writing-donts.md) as the detailed checklist for what to remove or rewrite.

## Run A Quick Self-Check

- Did you include concrete evidence?
- Did you remove hype language?
- Would this sound normal spoken aloud to a colleague?
- Did you overstate certainty or scope?
- Did you remove generic AI-writing templates and filler?

## Keep The One-Line Summary

Write clearly, honestly, and with enough evidence for the work to stand on its own.
