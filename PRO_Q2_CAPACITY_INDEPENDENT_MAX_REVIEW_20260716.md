# Q2 Capacity Independent Max Review

Date: 2026-07-16
Reviewer ID: `019f6b39-277a-7972-8ecb-beeb8a738605`
Reviewed commit: `1441219707b24719c2859f0dfa7c33d4b8190cbf`
Role: unique read-only independent max reviewer
Final choice: `C) KILL_Q2_R1`

## Evidence Integrity

Verdict: `PASS`.

- summary SHA-256:
  `dad8174f149240d8bba52eff9121b80588a90a4594061f7563fca476af4f30f9`;
- trace SHA-256:
  `b5ca44d240d22a43365ecaa6e6fe07a5cd564d079b351b8807b970176a46a0ed`;
- commitment SHA-256:
  `6cf2c0dc66ac6cc2beab9cb2e0eb64f8645914f28b218a47eb6717b93f83abb1`;
- summary and commitment canonical content hashes recompute exactly;
- all three checkpoint byte hashes, sizes, seeds, schema versions, and tensor
  fingerprints verify;
- resolved/scientific config identity and the 160-video data identity verify;
- strict replay parsed 524,258 trace rows: 123,940 annotation rows, 371,820
  actual rows, and 28,498 counterfactual-exhaustion rows. Actual and sparse
  counterfactual exhaustion aggregates reproduce exactly.

The trace does not include every bin for zero-exhaustion counterfactuals.
Therefore the `birth_prior_bias_m2` emission count of ten is bound by the
summary and commitment but cannot be recomputed from the trace alone. This is
an independent-replayability limitation, not an integrity failure.

## Mechanism Finding

The additive `-2` shift changes birth logits used by both the runtime birth
gate and the birth-assignment cost. Runtime slot availability is observed
before GT canonical assignment; decoding happens afterward. Lowering birth
logits therefore keeps runtime slots artificially free while GT births can
still receive canonical assignments.

Observed same-logits evidence:

```text
actual:               5517 emissions, 2206 exhaustions
birth bias -1:        1722 emissions,   80 exhaustions
birth threshold .75:  1270 emissions,   33 exhaustions
birth bias -2:          10 emissions,    0 exhaustions
```

Seeds 705 and 706 emit no event under `-2`. Annotation-side oracle minimum K is
two and `TRUE_CANONICAL_CAPACITY` causes zero cases. Exhaustion disappears
monotonically as birth suppression grows; the lifecycle/capacity mechanism is
not repaired.

## Kill Argument

The capacity gate is trivially satisfiable by never starting an event. A
sufficiently negative birth prior leaves every slot available and permits zero
GT canonical-assignment exhaustion even when runtime emits almost nothing.
This is a mechanism-identifiability defect, not an inference about downstream
mAP from random initialization.

The proposed `-2` is only an initialization. If training raises birth logits,
the frozen grid already shows that exhaustion returns at `-1` and increases
sharply near the native logits. Thus `-2` is not a stable shared lifecycle
contract; it is a silent low-activation operating point.

A new post-outcome emission floor would be chosen after seeing these results.
An effectiveness-labelled threshold would also violate the capacity-only
contract. Adding threshold, release-order, refractory, or availability
coupling would be an undeclared multi-factor revision. No further `-2`
confirmation audit is authorized.

## Final Authorization

```json
{
  "review_role": "independent_max_q2_capacity_gate",
  "commit_sha": "1441219707b24719c2859f0dfa7c33d4b8190cbf",
  "evidence_integrity_verdict": "PASS",
  "published_gate_status_verified": "REVISE_REQUIRED",
  "actual_exhaustions": 2206,
  "oracle_minimum_k": 2,
  "true_canonical_capacity_cases": 0,
  "legal_zero_exhaustion_candidates": ["birth_prior_bias_m2"],
  "actual_emissions": 5517,
  "birth_prior_bias_m2_emissions": 10,
  "mechanism_diagnosis": "DEGENERATE_BIRTH_SUPPRESSION",
  "choice": "C) KILL_Q2_R1",
  "selected_contract": null,
  "additional_zero_gpu_gate_required": false,
  "additional_zero_gpu_gate_authorized": false,
  "code_config_changes_authorized": false,
  "r1_implementation_allowed": false,
  "gpu_profile_allowed": false,
  "formal_training_allowed": false,
  "gpu_hours_authorized": 0
}
```

The `REVISE_REQUIRED` audit remains permanent evidence and is never relabelled
as PASS. Q2 and R1 terminate. GPU profile and formal training remain blocked.
