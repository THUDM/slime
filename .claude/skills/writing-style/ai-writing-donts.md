<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# AI-Writing Don'ts

Use this file for the final prose-only review pass. It captures common LLM
writing tells that should be removed from Marin-authored text.

These are warning signs, not strict proofs, but agent-written text should be
scrubbed against them aggressively.

## Prefer Direct Replacements

- Replace abstract bridge sentences with a concrete fact, result, decision, or caveat.
- Replace `the main change is...` with the actual change.
- Replace `the question is...` with the actual open question, or delete the sentence.
- Replace `the current state is...` with the state itself.
- Replace meta-summary prose with direct claims such as `X happened`, `X is better`, `X is still unknown`, or `we have not tested X`.

## Apply A Paragraph-Level Concrete Test

- Every paragraph should contain at least one concrete noun, result, decision, configuration, issue number, model name, command, metric, or caveat.
- If a paragraph has no numbers, named configs, named models, named issues, explicit decisions, or explicit unknowns, rewrite or delete it.
- Introductions and conclusions are not exempt by default. They may stay abstract only if they provide a real summary, framing constraint, or decision.

## Delete Bridge Sentences

- If a sentence can be deleted without losing factual content, delete it.
- Transition by ordering the facts well, not by narrating the transition.
- Delete lines like `The main change here is...`, `So the current state is...`, `What changed is...`, `The right summary is...`, or `Stepping back, ...` unless they add a real caveat or decision.

## Do Not Add Empty Importance Framing

- Do not write sentences that announce significance without adding content.
- Delete openings like `This is the part many people miss`, `This is where things get interesting`, `It is worth noting that`, or `Importantly`.
- Delete claims that a detail `marks a pivotal shift`, `underscores the importance`, `reflects a broader trend`, or `serves as a testament` unless the next clause names a concrete mechanism, result, or decision.
- Do not inflate ordinary facts into milestone language. Most facts are just facts.
- Do not use adjectives to imply evidence you have not shown: `crucial`, `pivotal`, `significant`, `transformative`, `groundbreaking`, `robust`, `powerful`, `seamless`.
- Replace significance language with the concrete thing that made it matter.

## Do Not Use Stock AI Rhetorical Templates

- Do not write `not just X, but Y`, `not X, but Y`, `more than just X`, or `it's not only X, it's also Y`.
- Do not write `the main change is not X, it is Y`, `the question is no longer X`, `not that X is done, but Y`, or similar contrast framing in analysis prose.
- Do not use `rather than` for rhetorical contrast when there is no real side-by-side comparison.
- Do not use TED-talk transitions like `What this means is`, `The key takeaway here is`, `At its core`, or `In many ways`.
- Do not use stagey reveal lines like `The real story is`, `What matters most is`, or `The answer lies in`.
- Do not use vague thesis templates like `X is more than a tool; it is a framework for...`.
- Do not use balanced-sounding filler such as `both a challenge and an opportunity`, `while powerful, it also raises important questions`, or `this highlights the delicate balance between`.
- Replace contrast framing with plain statements of the actual result.
- Prefer rewrites like these:
- Instead of `the main change is...`, write the change directly: `Quadratic decay now leads the sweep.`
- Instead of `the question is no longer X, but Y`, split it into two plain statements: `X is settled in this sweep. Y is still open.` Or just "Y is still open" if X is not worth mentioning and not in context.
- Instead of `not just X, but Y`, write `X, and Y` or just `Y` if X is not worth mentioning and not in context.
- Instead of `What this means is...`, state the implication directly: `This makes restart time the main bottleneck.`
- Instead of `At its core...`, state the mechanism directly: `The kernel is memory-bound, not compute-bound.`

## Do Not Regress To Generic Abstractions

- Do not replace a specific fact with a broad abstraction.
- If there is not a specific fact to state, delete the sentence rather than writing a vague generalization. TL;DR sections are not exempt from this rule.
- Prefer `we shard the checkpoint by layer to reduce restart time` over `this improves system efficiency`.
- Do not end paragraphs by zooming out to `the broader landscape`, `the evolving ecosystem`, `the future of`, or `the wider field` unless that zoom-out is necessary for the argument.
- Do not add generic `legacy`, `impact`, `significance`, `importance`, or `future directions` paragraphs to sound complete. Replace them with the actual impact, significance, or future direction if it is known, or delete them if it is not.
- Do not write conclusions that could apply to almost any ML project. Replace them with the actual conclusion that applies to this project or delete them if there is no real conclusion. Especially in summaries, issues, and internal communication, it is better to have no conclusion than a generic one.
- Replace abstract claims with the named thing that changed, won, failed, or remains unknown.

## Do Not Pad With Vague Analysis

- Do not summarize a technical choice as `showcasing the interplay between performance and scalability`.
- Do not write `this demonstrates how innovation and collaboration can drive progress`. Let the facts speak for themselves.
- Do not use `various`, `numerous`, `many`, `widely`, or `in some cases` when you can name the thing. If you cannot name the thing, consider deleting the sentence.
- Do not use vague attribution such as `some people argue`, `many believe`, `it is often seen as`, or `critics point out` without naming who or citing a source.
- Do not manufacture a takeaway when the honest statement is simply `we have not tested this yet`.
- Replace mushy analysis with one of: the observed result, the current caveat, the decision taken, or the next unresolved question.

## Do Not Sound Like A Generic Explainer

- Do not narrate the act of writing: `In this section, we will explore`, `Let's dive in`, `Here we can see`, `Before we begin`. You can write an introduction that sets up the section, but it should not be a meta-commentary on the writing process.
- After a heading, lead with the behavior, result, or constraint, not with an announcement that the topic matters.
- Do not treat headings as cues to switch into announcement voice. Example: under `## Checkpointing`, do not write `Checkpointing is a critical part of any robust training pipeline.` Instead, write `We shard the checkpoint by layer to reduce restart time.`
- Do not add recap paragraphs after every section if the section was already clear.
- Do not use FAQ voice in prose that is not actually a FAQ.
- Do not write teacherly reassurance such as `don't worry if this seems complex` unless the medium specifically calls for it.

## Do Not Use Chatbot Meta-Communication

- Do not write `I hope this helps`, `let me know if you'd like`, `happy to`, or similar assistant-signoff language in Marin-authored prose. You can simply end with the last fact, decision, call-to-action, or caveat without adding a closing sentence.
- Do not leave prompt-shaped leftovers like `Here is a revised version`, `Below is a summary`, or `Key takeaways:`.

## Do Not Use AI Vocabulary Clusters

- Watch for clusters of words such as `crucial`, `pivotal`, `underscore`, `highlight`, `showcase`, `delve`, `foster`, `enhance`, `align with`, `valuable`, `vibrant`, `landscape`, `interplay`, `intricate`, `tapestry`, `testament`, `quiet`.
- One occurrence may be fine. Several in one page usually means the prose has drifted into generic LLM style.
- Replace them with direct verbs and nouns that name the actual action or result.

## Do Not Force Variety At The Cost Of Precision

- Do not rotate through synonyms for the same technical object just to avoid repetition.
- If the thing is a `checkpoint`, keep calling it a `checkpoint` unless there is a real distinction.
- Do not rename the same thing as `the system`, `the framework`, `the platform`, `the stack`, and `the pipeline` in adjacent paragraphs.
- Prefer exact repetition over fake elegance.

## Do Not Use Listicle Cadence

- Do not force the rule of three just because it sounds polished.
- Rewrite sequences like `fast, flexible, and scalable` into a concrete claim or drop them.
- Do not pile up adjective trios or noun trios to simulate completeness.
- Do not use inline-header list patterns such as `Why it matters:` followed by vague bullets unless the bullets add real information.
- Do not use boldface as a substitute for structure.

## Do Not Hide Simple Claims Behind Fancy Grammar

- Prefer `X is Y` when it is correct.
- Do not contort sentences to avoid plain copular statements.
- Do not write long dependent-clause openings when the main clause is short and concrete.
- Do not stack transitions like `Additionally`, `Furthermore`, `Moreover`, `Importantly`, `Notably` at the start of consecutive sentences.
- Do not use em dashes to splice in every aside. If the aside matters, make it its own sentence.
- When possible, rewrite as one of: `X changed`, `X is better`, `X is worse`, `X is still unknown`, `we have not tested X`, `we chose X because Y`.

## Do Not Fill Gaps With Speculation

- Do not write around missing evidence with phrases like `details are limited`, `based on available information`, or `while not widely documented`. Admit the gap instead of trying to paper over it.
- Do not speculate about motives, significance, private context, or hidden constraints to smooth over uncertainty.
- If a fact is unknown, say it is unknown or omit it.
- Do not hedge with polished filler that sounds informed but is unsupported.
- If there is not much progress to report in a progress report, say so.

## Do Not Add Cosmetic Structure That Looks Smart

- Do not add a summary, takeaway box, or future-work section unless it helps the reader act.
- Do not use tables for prose comparisons that read better as sentences. However, tables are preferred for actual data, ablation results, or configuration sweeps.
- Do not over-title-case headings or sprinkle bold labels through ordinary paragraphs.
- Do not add a `Challenges and opportunities` section by reflex.
- Do not end with a grand summary if a short concrete closing sentence will do.

## Do Not Over-Format The Page

- Do not use boldface to make ordinary prose feel more important.
- Do not skip heading levels or add structure that implies depth the piece does not have.
- Do not use emojis, flourish punctuation, or stylized subject-line prose to create energy.

## Quick Negative Examples

- `This is the part many people miss.` -> Just omit, and consider whether the next sentence is actually important.
- `This marks a pivotal shift in the broader landscape.` -> Just omit, and consider whether the prior sentence is actually important.
- `Marin is not just a framework, but a testament to open innovation.` -> Just omit. Let the facts speak for themselves.
- `This highlights the intricate interplay between scalability, flexibility, and performance.` -> Omit
- `As the ecosystem continues to evolve, this work will likely play an important role.` -> Just omit, and consider whether the next sentence is actually important.

## Bad / Better Rewrites

- Bad: `The main change here is that the surrounding schedule and capacity choices are becoming much better grounded.`
- Better: `Quadratic decay now leads the schedule sweep, and \`cf=1.0\` looks faster than \`cf=1.25\` with a small measured quality hit.`
- Bad: `The optimizer question is still open, but the main change this week was on schedules.`
- Better: `Quadratic decay now leads the schedule sweep. AdamH vs Adam is still open.`
- Bad: `This is still mostly procedural progress.`
- Better: `The code is ready, but the ablation has not run yet.` or `Issues have been created, but the ablation has not run yet.`
- Bad: `The question is no longer whether schedules matter, but how to integrate them into the broader training picture.`
- Better: `Schedules matter in this sweep. The open question is whether the win survives at larger scale.`
- Bad: `What changed is that the current state is much clearer than it was before.`
- Better: `The sweep removed two weak settings, and the remaining comparison is quadratic decay vs cosine.`

## Quick Rewrite Heuristic

- If a sentence could survive unchanged in a random SaaS blog post, delete or rewrite it.
- If a sentence mainly gestures at significance, replace it with the concrete fact.
- If you can swap in three different project names and the sentence still sounds right, it is too generic.

## Final Compression Pass

- After drafting, remove every sentence that only comments on the state of the writing, framing, significance, or process unless it adds a decision, result, or caveat.
- Read each paragraph and cut any sentence whose only job is to introduce, summarize, or soften the sentence that follows it.
- Keep transition sentences only when they carry real information.

## Paragraph Scoring Rubric

- For each paragraph, ask: does it state a result, setup, caveat, or decision?
- If not, rewrite or delete it.
- If yes, ask: is that information stated directly, or hidden behind framing language?
- Prefer deletion over polishing when a sentence adds no content.
