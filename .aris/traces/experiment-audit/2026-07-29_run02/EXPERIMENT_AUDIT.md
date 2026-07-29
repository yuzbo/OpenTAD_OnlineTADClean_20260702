# EventMATR D1 strict-causal checkpoint replay audit

Date: 2026-07-29

Overall verdict: **WARN**

Integrity verdict: **PASS**

Scientific verdict: **root cause localized; current model not sufficient**

## Exact evidence

- training checkpoints: commit
  `1f4bb29ad58dddcc33f6ff2bdc57a5934ee5c53d`, tree
  `aad757531cfcbfb79b616756466251f946574035`, seed 52, epoch 5;
- replay source: commit
  `6a23ab3a3711bc1ecb5a2fe442e302964ee3afb8`, tree
  `8564dca45850675ce9ac366ee051aacbc4d97dc1`;
- replay manifest SHA-256:
  `b8980f21b34f3a862d8306c3a48249085351e4bd166f08d561016d29d3be58ed`;
- remote tests: Slurm `1203223`, `54 passed in 37.75s`;
- replay: Slurm array `1203224`, four `COMPLETED 0:0` tasks;
- combined finalizer: `PASS`.

Two earlier test-launch attempts (`1203220/1203221`) failed before Python because
the scheduler's default shell could not source the environment. They loaded no
model or data and are not experimental runs. The operative Bash submission is
`1203223`.

## Integrity checks

| Check | Verdict | Evidence |
|---|---|---|
| train-only scope | PASS | 200 official train/validation videos; no test loader |
| model boundary | PASS | evaluation payload excludes duration, full-video timing and GT-derived `segment_flag`; no targets passed |
| observed EOS | PASS | exactly 200 current-time EOS markers per lane |
| source identity | PASS | start/final source receipts are byte-identical in every lane |
| checkpoint immutability | PASS | size, mtime and SHA-256 unchanged before/after |
| thresholds | PASS | no numeric birth/end threshold; no search or lowering |
| ledger | PASS | ends = emissions = ledger rows in live lanes; unique IDs and positive intervals |
| capacity | PASS | zero exhaustion; no silent eviction |
| numerical health | PASS | finite summaries, margins, calibration and systems metrics |

This replay is a real-data, train-only, predicted-track inference diagnostic.
It is not a paper result and does not estimate held-out generalization.

## Root-cause result

| Lane | Birth | Cancel | End | Emit | Reacquire | Active EOS | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | 46,935 | 46,892 | 0 | 0 | 0 | 43 | 0 / 200 |
| T | 5,670 | 5,667 | 0 | 0 | 4,017 | 3 | 0 / 200 |
| H | 2,648 | 2,599 | 46 | 46 | 0 | 3 | 6 / 200 |
| TH | 5,621 | 5,072 | 519 | 519 | 3,611 | 30 | 26 / 200 |

R/T have many predicted START transitions. Their zero output therefore is not
candidate START starvation. Owner BACKGROUND cancels almost every record and
owner END never wins. H/TH have exact END-to-emit equality, excluding the writer
as the bottleneck. The native memory gate is similar across lanes and is not
the factor-specific first cause.

Censored owner risk is a causal intervention on liveness, not a solved result.
H/TH recall at temporal overlap 0.3 is only `0.166%/0.998%`. Only `2/6`
ground-truth end prefixes, respectively, have any owner END; even those are not
identity-matched. T/TH reacquisition counts together with high cancellation
show cancel/rebirth churn.

## Code-to-result audit

The intended mixed-track supervision is incomplete:

1. training creates oracle paths from prefix-visible targets;
2. a predicted birth inherits a target ID only through exact query-index
   equality with the oracle path;
3. every other predicted record retains `target_event_id=None`;
4. those records receive owner BACKGROUND/no-class supervision and are excluded
   from true-event end hazard;
5. inference has no oracle records.

This is a plausible upstream cause of the replayed cancellation prior. It is a
strong mechanistic inference supported by code and counts, not yet an isolated
counterfactual.

The temporal assignment history and hazard risk set are also batch-local.
Because each video spans multiple 64-prefix batches, the nominal pre-birth
window is truncated at every boundary. This must be isolated before judging the
hazard formulation itself.

## Independent review

A fresh zero-history reviewer independently confirmed:

- cancellation plus END starvation is the immediate zero-output cause;
- END-to-writer is live;
- predicted/oracle target association and batch-local history are important
  structural risks;
- current evidence cannot separate decoder capacity, background prior and
  association noise without counterfactuals.

The reviewer initially described replay proposals as oracle proposals. That
statement was rejected by direct boundary inspection: replay passes no targets
and is predicted-only. The valid concern is mixed-oracle training exposure.

Fresh Claude and Gemini review attempts failed authentication
(revoked OAuth token / invalid API key). No cross-model verdict is claimed.

## Limitations and required actions

- one seed, five epochs and train-only data cannot establish generalization;
- background-dominated Brier/ECE can look good while event recall is near zero;
- owner-switch correctness is not identifiable when almost every track cancels;
- the replay's 64 feature-prefix nearby window is not the same coordinate as the
  earlier 64 source-frame annotation audit;
- no preregistered quantitative effect threshold existed, so this is not
  retroactively labelled a failed numeric scientific gate.

Hold unchanged 10/20-epoch training. Repair predicted-track association,
source-stratified owner supervision, explicit cancellation and cross-batch
history; then run synthetic, real-batch and one-epoch counterfactuals before a
new registered five-epoch pilot.
