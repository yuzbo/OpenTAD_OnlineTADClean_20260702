# Q2 Capacity and Lifecycle Audit Preregistration

Date: 2026-07-16
Status: preregistered; implementation validation pending
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
CUDA_VISIBLE_DEVICES: empty before process start
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
