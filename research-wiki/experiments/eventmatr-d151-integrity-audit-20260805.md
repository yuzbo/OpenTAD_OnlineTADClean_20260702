---
type: experiment-audit
date: 2026-08-05
status: pass-diagnostic-only
scope: EventMATR D1.5.1 frozen-trace endpoint-margin result
---

# EventMATR D1.5.1 Integrity Audit

## Verdict

Job `1215356` is accepted as an integrity-valid, train-only, read-only mechanism
diagnostic. It is not a model-training run, an official comparison or a paper
performance result. The only released action is to preregister a
policy-independent three-state competing-risk model repair. Model implementation
and training remain blocked until that new design is prospectively approved.

## Immutable identities

- append-only result root:
  `/data/run01/sczc063/yuzibo/runs/eventmatr_d151/endpoint_margin_adcb0db_20260803_r1`;
- analyzer source commit/tree:
  `adcb0db3153e701644af5338335f266e80badeff` /
  `cd4da1f7a1f2866a39c399bccecc71262b4390db`;
- start and final source-identity SHA-256:
  `3ce9cbe9c608c15207d2e5149e22a0a03463c82ae55dca04b2a932f2321fb6a6`;
- receipt SHA-256:
  `ef5a437d613c92b08111ceb3764c10ed39872d532948b8892c2464461e22e4c5`;
- submission-script SHA-256:
  `d79aa09243b06787539fa651f74b3a0db8c08937f7e61434c4c13b6d20e65fb6`;
- analyzer/wrapper SHA-256:
  `0c48440f6eca91c0f48eff2407686e9bfc36c7273fb8a010a69fccf9f55c0779` /
  `b3d471997733618c5dd711a9da3d8f13eacd5e719122561d41f9b5b1c2c154f7`;
- independent recomputation summary SHA-256:
  `0c790d69abd324bc3c4b6afab8a3c057cfcb8650e61d2400cffcde31100d05dc`.

The cluster accounting record was observed as `COMPLETED 0:0` before artifact
collection. The retained stderr is empty and stdout verifies all three frozen
D1.5 input hashes before emitting `PASS_DIAGNOSTIC`. A separate local reviewer
could not re-query the expired Slurm accounting row, so the exit code is not
claimed to be recoverable from the downloaded files alone.

## Independent recomputation

A standard-library implementation, independent of the production analyzer,
reloaded all `24,008` event rows. It verified exactly `3,001` common fully
observed event identities in each of eight routes and `200` video clusters. It
recomputed every route summary, temporal bin, duration stratum, nearest-rank
median/95th percentile, all `10,000` common video-cluster bootstrap resamples at
seed `52017`, and the frozen routing decision. The mismatch list is empty.

| Route | Positive endpoint margins / 3,001 | 95% video-bootstrap interval | Median margin | 95th-percentile margin |
|---|---:|---:|---:|---:|
| OF formal | 0 | [0, 0] | -2.989564 | -1.827708 |
| OF no-cancel shadow | 7 | [0, 0.005997] | -1.602913 | -0.574103 |
| OR formal | 0 | [0, 0] | -2.986168 | -1.833560 |
| OR no-cancel shadow | 6 | [0, 0.004984] | -1.673983 | -0.586817 |
| PF formal | 0 | [0, 0] | -3.336526 | -1.736283 |
| PF no-cancel shadow | 3 | [0, 0.002837] | -1.907199 | -0.574290 |
| PR formal | 0 | [0, 0] | -3.328273 | -1.709434 |
| PR no-cancel shadow | 3 | [0, 0.002837] | -2.015220 | -0.591448 |

The oracle-refreshed minus oracle-free shadow difference is
`-0.0003332223`, with interval `[-0.0011223345, 0]`. Every route has zero
endpoint-positive winner without a matching target-backed immutable emission,
so state-machine accounting does not select the route. Both oracle no-cancel
shadows remain far below the frozen `5%` materiality floor; their upper
intervals are below `0.6%`, and their median and 95th-percentile endpoint margins
remain negative.

## Causality and release audit

The receipt records `test_access=false`, `train_only=true`, no loaded model, no
forward pass, no optimizer, no optimizer step, no checkpoint update, no threshold
search and no model-data resampling. Ground truth is used only by the
post-forward evaluation sidecar to locate annotated endpoints and is never
presented as deployable inference state. Consequently this diagnostic contains
no model-inference ground-truth leak, but it is deliberately annotation-aided
and therefore records `strict_causal_paper_result_valid=false`,
`paper_performance_valid=false`, `paper_claim_release=false`,
`locked_test_release=false` and `official_comparison_release=false`.

An independent read-only reviewer checked the analyzer, tests, protocol, source
identities and result object. It agreed with the integrity and scope verdict. It
used the available independent default reviewer rather than a different
high-tier model family; this reviewer-family limitation is recorded rather than
misrepresented as a Pro-model audit.

## Scientific disposition

The result rules out cancellation policy, identity refresh and ledger emission
accounting as sufficient primary repairs. It does not separately identify frozen
owner representation versus competing-risk learning, but the prospectively
frozen decision table resolves that case by authorizing only design and
preregistration of a policy-independent three-state competing-risk repair. The
receipt explicitly keeps `model_implementation_authorized=false` and
`model_training_authorized=false`.
