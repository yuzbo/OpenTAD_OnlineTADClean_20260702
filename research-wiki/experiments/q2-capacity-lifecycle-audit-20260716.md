# Q2 Capacity and Lifecycle Audit Preregistration

Date: 2026-07-16
Status: terminal `REVISE_REQUIRED`; independent disposition `KILL_Q2_R1`
GPU authorization: blocked; this experiment is CPU-only

## Question

Can the frozen Q2 controller support every first-crossing GT birth on the full
160-video fit core under one fixed/rematch-shared, prefix-observable lifecycle
contract, or are the observed exhaustions caused by structural canonical
capacity, false runtime ACTIVE occupancy, refractory occupancy, or same-bin
non-reuse?

This is a readiness and root-cause audit. It is not an effectiveness test and
cannot establish mAP, latency, novelty, or training efficiency.

## Immutable Inputs

- config: `configs/causaltad/thumos_pes_q2_persist_fixed.py`;
- route: `q2_persistent_binding_one_factor`;
- fit core: exact locked 160-video manifest;
- deterministic initialization seeds: `705`, `706`, `707`;
- native head: `K=4`, dropout `0.1`, thresholds `0.5/0.5/0.5`, refractory `2`;
- model mode: CPU FP32 `train`, so the frozen dropout path remains active;
- video and packet order: lexicographic video order, then immutable
  chronological packet order;
- checkpoint bytes: generated once per seed in an external immutable bundle
  and rebound to commit, config, tensor fingerprint, and file SHA-256.

The runner must reject a dirty checkout, config drift, seed drift, checkpoint
drift, split drift, feature-cache drift, GT-tainted head metadata, visible GPU,
or overwrite of an existing evidence directory.

## Annotation Audit

For every fit-core bin, record GT active, births, ends, same-bin birth/end,
same-class and cross-class concurrency, canonical occupancy before/after,
declared refractory reserve, current-order required K, and minimum oracle-free
K. Report a structural K census for capacities `2..max(8, observed demand)`.

## Frozen Same-Logits Grid

The numerical head trace is computed once per seed/video. The current head's
logits/living numerical recurrence must first be dynamically revalidated as
independent of lifecycle tensors. The following controller policies are then
replayed without inspecting effectiveness:

```text
actual
refractory_0
birth_threshold_0p25
birth_threshold_0p75
alive_threshold_0p25
alive_threshold_0p75
end_threshold_0p25
end_threshold_0p75
birth_prior_bias_m1
birth_prior_bias_m2
release_before_birth
release_before_birth_refractory_0
canonical_only_privileged
capacity_k2
capacity_k3
```

`canonical_only_privileged`, `capacity_k2`, and `capacity_k3` are diagnostics,
not selectable trainer contracts. Same-logits expansion beyond native K=4 is
scientifically undefined because no frozen logits exist for extra slots; K>4
is therefore represented only in the annotation structural census, never by
fabricated slot logits.

## Exhaustion Closure

Every exhausted instance receives exactly one label:

```text
TRUE_CANONICAL_CAPACITY
FALSE_ACTIVE_OCCUPANCY
REFRACTORY_OCCUPANCY
SAME_BIN_NON_REUSE
MIXED_CAUSE
```

An unexplained exhausted instance, fixed/rematch canonical divergence, missing
seed, partial fit core, or malformed trace makes the evidence invalid.

## Frozen Resource Contract

```text
execution platform: CPU only
threads: 8
runner wall limit: 21,000 seconds
declared maximum: 46.667 CPU-hours
hard CPU cap: 48 CPU-hours
GPU hours: 0
```

The runner stops fail-closed at its wall limit and reports
`BUDGET_EXCEEDED`; it may not reduce videos, seeds, policies, or trace fields
after observing outcomes.

### Pre-outcome platform amendment

The first N16R4 submission attempt failed before queue admission and before
any checkpoint/logit/capacity outcome existed. N16R4 exposes only a `gpu`
partition and its submission Lua rejects both an omitted GPU request and
`--gpus=0`. Allocating an unused GPU would still violate the frozen zero-GPU
budget. Long execution on the login node is also prohibited.

Therefore the execution platform is amended, before outcomes, to:

```text
host CPU: 13th Gen Intel Core i5-13600KF
physical/logical cores: 14/20
OS: Windows x86-64, recorded exactly in the output artifact
Python: 3.10.20
PyTorch: 2.6.0+cu124, CPU tensors only
CUDA_VISIBLE_DEVICES: -1 before process start
threads: 8
runner wall limit: 21,000 seconds
hard cap: 48 CPU-hours
```

The public THUMOS/Q2 data were copied byte-for-byte from N16R4 to the exact
Windows resolution of the unchanged Linux-style config paths. Before launch,
the local runner requires these identities:

```text
cache manifest:
bca3528b82858cd47a2f9581158dd192dfa65975d2031d009a1a68ec637f6796
annotation:
ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad
fit-core manifest:
36ebd89b583d259989b0260ca7314a308a2ba98ab3bc7bcbf79cc8ec40b628bc
development split:
847893621aa44cd665429dcdd6b41f4516f1034de2be555ea9b66d5a5384020f
feature cache inventory: 823 files, 499,846,443 bytes
```

The dataset still verifies every selected feature against the original cache
manifest. No policy, seed, video, threshold, K, cause label, gate, or CPU
budget changed. The failed Slurm attempt consumed zero GPU-hours and is not an
experiment result.

The first Windows launch also failed before loading the dataset or producing
logits because Windows removes an environment variable assigned the empty
string. The fail-closed runner therefore saw CUDA as visible and stopped after
checkpoint generation. The frozen value is amended to the runner-supported
sentinel `-1`; the failed directory is retained and never reused. This is an
execution preflight correction, not an outcome-dependent policy change.

## Evidence

The external, non-overwritten bundle contains:

```text
checkpoints/manifest.json
checkpoints/seed_705.pth
checkpoints/seed_706.pth
checkpoints/seed_707.pth
evidence/capacity_trace.jsonl.gz
evidence/capacity_summary.json
evidence/commitment.json
job.sbatch
slurm.<job>.out
slurm.<job>.err
```

The compressed trace contains all annotation bins, all actual-policy bins,
and every counterfactual exhaustion. The summary binds source commit, resolved
config, checkpoint manifest, data identity, policy contract, trace bytes,
platform, resource use, and gate output.

## Decision Rule

```text
INVALID_EVIDENCE:
  incomplete or unclosed evidence

Q2_INVALID_STRUCTURAL_CAPACITY:
  minimum oracle-free K > 4

PASS_FREEZE_ACTUAL:
  annotation demand <= 4
  AND actual policy has zero exhaustion for all 160 x 3
  AND every attribution/equality/provenance invariant passes

REVISE_REQUIRED:
  actual policy exhausts
  AND at least one non-privileged shared counterfactual is zero-exhaustion

Q2_INVALID_NO_SHARED_POLICY:
  actual exhausts
  AND no legal shared policy has zero exhaustion
```

Only `PASS_FREEZE_ACTUAL` directly authorizes R1/CSFSB implementation. A
`REVISE_REQUIRED` result authorizes review and a separate one-factor lifecycle
revision commit, not automatic policy selection. No outcome of this audit
authorizes a GPU profile or formal training.

## Launch

```bash
powershell -NoProfile -ExecutionPolicy Bypass \
  -File tools/run_q2_capacity_audit_local.ps1 \
  -RunDir C:/data/run01/sczc063/yuzibo/full_petal/q2_capacity_<commit>
```

The N16R4 launcher now requires an explicitly named CPU-only partition and
fails before publication when none exists. It must not fall back to a GPU
partition.

The next step after a valid capacity contract remains R1 implementation,
local/target-Linux B0, same-reviewer PASS, and a newly preregistered unseen CPU
G0. GPU profile remains blocked until that G0 passes.

## Pre-outcome Windows Finalization Failure

The clean-commit run rooted at
`q2_capacity_3e20a01_20260716` completed all `160 x 3` replay units before
failing during evidence publication. No terminal gate line or valid evidence
directory was published. Windows rejected the atomic directory rename because
the externally opened raw file beneath `capacity_trace.jsonl.gz` remained open
after the text and gzip wrappers were closed. Cleanup then deleted the summary
before failing on the same locked trace, so the retained `.partial-*` directory
is explicitly invalid and cannot be reconstructed or cited as an outcome.

This is a pre-outcome portability defect. The correction only closes the raw
trace handle explicitly and preserves the original exception if cleanup also
fails. It does not change data, checkpoints, seed order, logits, policies,
thresholds, K, lifecycle semantics, attribution, gate rules, or resource caps.
The complete audit must be rerun from a new clean commit into a new
non-overwritten run root before any capacity conclusion or R1 implementation.

## Terminal Clean Rerun

The clean rerun bound to commit
`1441219707b24719c2859f0dfa7c33d4b8190cbf` completed all 160 videos for
seeds `705/706/707`, atomically published its evidence directory, and exited
without stderr. The evidence self-hashes, summary/trace commitment, checkpoint
manifest, config identity, data identity, GT-taint audit, cause closure, and
fixed/rematch canonical equality all verify.

```text
STATUS=REVISE_REQUIRED
ACTUAL_EXHAUSTIONS=2206
R1_IMPLEMENTATION_ALLOWED=false
GPU_PROFILE_ALLOWED=false
FORMAL_TRAINING_ALLOWED=false

summary_sha256=dad8174f149240d8bba52eff9121b80588a90a4594061f7563fca476af4f30f9
trace_sha256=b5ca44d240d22a43365ecaa6e6fe07a5cd564d079b351b8807b970176a46a0ed
commitment_sha256=6cf2c0dc66ac6cc2beab9cb2e0eb64f8645914f28b218a47eb6717b93f83abb1
elapsed_seconds=2251.922
gpu_hours=0
```

The annotation census gives `minimum_oracle_free_k=2`; no exhaustion is
attributed to true canonical capacity. Actual-policy causes are 1 pure false
ACTIVE, 1,144 refractory, and 1,061 mixed false-ACTIVE/refractory cases. The
actual policy emits 5,517 events and exhausts 2,206 births. Refractory zero,
release-before-birth, and their combination still exhaust 2,199, 1,893, and
228 births respectively. A `-1` birth-logit bias exhausts 80, while the only
eligible zero-exhaustion candidate, `birth_prior_bias_m2`, emits only 10 events
over the same trace. The privileged canonical-only diagnostic also reaches
zero but remains forbidden as a trainer contract.

The result rejects current Q2 capacity readiness. It does not authorize
automatic adoption of `-2`: zero exhaustion may be achieved by suppressing
almost every model birth rather than by repairing lifecycle capacity. A unique
independent max reviewer must adjudicate whether a predeclared one-factor
birth-prior revision plus a zero-GPU nondegeneracy gate is scientifically
valid, or whether Q2/R1 must terminate. R1 code remains blocked meanwhile.

## Independent Terminal Disposition

Reviewer `019f6b39-277a-7972-8ecb-beeb8a738605` independently verified the
entire evidence chain and strictly parsed all 524,258 trace rows. The reviewer
selected `C) KILL_Q2_R1` with mechanism diagnosis
`DEGENERATE_BIRTH_SUPPRESSION`.

The zero-exhaustion `-2` arm is rejected because the gate can be passed by
silencing runtime births while GT canonical assignment still receives free
slots. Its ten emissions include zero for seeds 705 and 706. The monotonic
progression from actual, to `-1`, to `-2` demonstrates suppression rather than
lifecycle repair, and training can move logits back into the measured
exhaustion regime.

No new nondegeneracy threshold may be selected after observing this result. No
model/config revision, repeat `-2` audit, R1 implementation, GPU profile, or
formal training is authorized. The audit status itself remains
`REVISE_REQUIRED`; the independent scientific disposition is terminal `KILL`.
