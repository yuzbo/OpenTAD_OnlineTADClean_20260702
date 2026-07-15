# Full PETAL CRS-EPS Independent Implementation Review

## Reviewed Anchor

- Commit: `0731f070e7d7f982a1df20cfaff801b21bed96fe`
- Branch: `codex/full-petal-implementation`
- B0 artifact SHA-256:
  `51fb0f2966df1e2f0672c90346fc39400c092cdb695ffdc1e2a8f35c133826db`
- B0 result: `548/548 PASS`, zero failures, errors, skips, blocking
  findings, or protocol violations.
- Reviewer: `019f5abd-5104-79b3-882e-354ca796f2c1`

## Verdict

```text
VERDICT=REVISE
PROFILE=BLOCK
FORMAL_TRAINING=BLOCK
blocking_findings=4
protocol_violations=4
```

## Accepted Blocking Findings

1. `P0-GATE-001`: profile tickets did not require an immutable signed G0 PASS
   artifact, so the documented G0 order was not launch-enforced.
2. `P1-TRAIN-001`: the optimizer group identity did not prove exact membership
   and order of all `M` manifest draws; duplicate/missing draws could still
   reach one optimizer event.
3. `P1-TRAIN-002`: pre-boundary failures rolled back gradients and online
   state but did not restore model buffers mutated during a failed draw.
4. `P1-G0-002`: G0 accepted caller-claimed provenance and outcome-blind margin
   status without signed preregistration or reconciliation against the loaded
   dataset bytes.

## Required Correction

- Sign selection and margins before G0 execution and bind commit, config, data
  identity, manifest, sampling population, and selection digest.
- Reconcile manifest provenance with the annotation, cache manifest, split
  files, and sampling specs actually loaded by the runner.
- Sign the final terminal G0 audit, recompute it at launch, and require G0 PASS
  in every CRS-EPS profile ticket before CUDA/DDP.
- Bind each video optimizer event to exact `0..M-1` membership and an immutable
  episode-sequence digest.
- Restore the group-start buffer snapshot and staged state on every failure
  path, including forward exception, buffer leak, slot exhaustion, non-finite
  loss/gradient, and incomplete group.

## Gate Consequence

No profile or formal training is authorized from this review. After correction,
the exact new commit must receive a new complete B0 artifact and be reviewed by
the same reviewer. Only `PASS / PROFILE=ALLOW` can authorize the real G0/profile
sequence; formal training remains a later gate.
