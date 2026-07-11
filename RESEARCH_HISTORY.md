# Online TAD Research History

This is the public landing page for the project's multi-round research discussion, external Pro reviews, rejected routes, active decisions, and current deep-review prompt.

## Public Anchor

```text
Repository: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702
Branch: codex/online-tad-clean-20260702
Code baseline before this history publication:
bfd0608b2996cba30d158d741ee193476a5078df
```

Branch URL:

https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702

## Current Decision

The task is fixed to standard fully supervised Online Temporal Action Detection/Localization:

- causal RGB stream input;
- no future observations;
- instance-level `{start, end, class, score}` output when an action end is detected;
- emitted detections are immutable;
- no new sensors, observability labels, VideoQA outputs, Online TAS task, or broader semantic-maintenance task.

PIVOT/Three-Clock Event Observability is rejected as out of scope. PCEH/CESR is retained only as audited infrastructure and negative baselines.

The current lead candidate is **PETAL-OnTAD**:

> Jointly train a causal raw-video backbone and persistent action-instance queries, use trajectory-level prefix supervision, and require batched causal training to agree with incremental cached inference.

PETAL has not passed novelty review and has not been formally trained. Raw-video training remains on hold until a dedicated Pro review and a matched feature-level persistent-query pilot pass.

## Canonical Navigation

Read these files in order:

1. [`research-wiki/query_pack.md`](research-wiki/query_pack.md)
   Compressed current state, failed routes, closest work, gates, and next action.

2. [`research-wiki/discussion_timeline.md`](research-wiki/discussion_timeline.md)
   Chronological record of major questions, corrections, route changes, and artifacts.

3. [`research-wiki/decision_register.md`](research-wiki/decision_register.md)
   One entry per major decision, including rationale, counterarguments, reversibility, and supersession.

4. [`research-wiki/source_map.md`](research-wiki/source_map.md)
   Maps Pro reviews, attachments, literature audits, code areas, and discussion prompts to wiki nodes.

5. [`research-wiki/index.md`](research-wiki/index.md)
   Full entity index for ideas, papers, experiments, claims, and navigation.

6. [`research-wiki/ideas/petal-ontad.md`](research-wiki/ideas/petal-ontad.md)
   Current candidate, closest-work boundary, method sketch, experiments, and kill criteria.

## Current Pro Prompt

Use the complete prompt without deleting its repository audit, novelty search, strongest rejection, reconstructed baselines, or output format:

[`PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md`](PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md)

Direct GitHub URL:

https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md

## Review History

### Repository and Method Audits

- [`PRO_REVIEW_20260705.md`](PRO_REVIEW_20260705.md)
- [`PRO_REVIEW_20260706.md`](PRO_REVIEW_20260706.md)
- [`PRO_REVIEW_20260707.md`](PRO_REVIEW_20260707.md)
- [`PRO_REVIEW_20260708.md`](PRO_REVIEW_20260708.md)
- [`PRO_REVIEW_20260709.md`](PRO_REVIEW_20260709.md)
- [`PRO_REVIEW_20260710.md`](PRO_REVIEW_20260710.md)
- [`PRO_REVIEW_20260711.md`](PRO_REVIEW_20260711.md)

### Absorbed Decision Layers

- [`PRO_REVIEW_ABSORPTION_20260706.md`](PRO_REVIEW_ABSORPTION_20260706.md)
- [`PRO_REVIEW_ABSORPTION_20260707.md`](PRO_REVIEW_ABSORPTION_20260707.md)
- [`PRO_REVIEW_ABSORPTION_20260708.md`](PRO_REVIEW_ABSORPTION_20260708.md)
- [`PRO_REVIEW_ABSORPTION_20260709.md`](PRO_REVIEW_ABSORPTION_20260709.md)
- [`PRO_REVIEW_ABSORPTION_20260710.md`](PRO_REVIEW_ABSORPTION_20260710.md)
- [`PRO_REVIEW_ABSORPTION_20260711.md`](PRO_REVIEW_ABSORPTION_20260711.md)

The absorption files are the local decision layer. They do not silently replace the original reviews; both are preserved.

### Divergent Idea Round

- [`PRO_DIVERGENT_IDEA_PROMPT_20260711.md`](PRO_DIVERGENT_IDEA_PROMPT_20260711.md)
- [`PRO_DIVERGENT_IDEA_REVIEW_20260711.md`](PRO_DIVERGENT_IDEA_REVIEW_20260711.md)
- [`PRO_DIVERGENT_IDEA_ABSORPTION_20260711.md`](PRO_DIVERGENT_IDEA_ABSORPTION_20260711.md)

This round generated 36 candidates. Its recommendation was only partially accepted. See DR-021 through DR-025 before reusing any route.

### Rejected Three-Clock/PIVOT Route

- [`THREE_CLOCK_TASK_METHOD_DESIGN_20260711.md`](THREE_CLOCK_TASK_METHOD_DESIGN_20260711.md)
- [`THREE_CLOCK_COMPETITION_REVIEW_20260711.md`](THREE_CLOCK_COMPETITION_REVIEW_20260711.md)
- [`PRO_THREE_CLOCK_DEEP_REVIEW_PROMPT_20260711.md`](PRO_THREE_CLOCK_DEEP_REVIEW_PROMPT_20260711.md)

These files are retained to prevent repetition, not to endorse the route. The user rejected PIVOT because it leaves standard On-TAD.

## Supporting Research Notes

- [`end-to-end-ontad-research.md`](end-to-end-ontad-research.md)
  Strict distinction between feature-level detector end-to-end and raw-video instance-level end-to-end On-TAD.

- [`online-action-models-2026-research.md`](online-action-models-2026-research.md)
  Current OAD, On-TAD, Online TAS, and streaming-video landscape.

- [`pceh-current-implementation-20260710.md`](pceh-current-implementation-20260710.md)
  Snapshot of the implementation state before the later scientific demotion.

## Evidence Rules

The repository uses these evidence labels:

```text
verified from primary paper
verified from official code
verified from target GitHub branch
author-provided / local-unverified
inference
unknown
requires experiment
```

Historical statements such as “current best route” are time-local. Resolve conflicts by using the newest timeline entry and decision-register resolution.

No checkpoint, dataset, feature dump, remote log, generated result archive, private attachment, or credential belongs in this history layer.

## Immediate Next Step

1. Submit the PETAL Pro prompt.
2. Archive the full response byte-identically.
3. Create a separate absorption document that accepts or rejects each major recommendation.
4. Update the timeline, decision register, source map, PETAL node, and query pack.
5. Only then decide whether the feature-level P0 pilot should be implemented.
