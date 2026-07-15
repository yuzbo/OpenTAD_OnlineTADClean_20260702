# Full PETAL Code and Science Review Absorption, 2026-07-15

## Source

- Archived response: `PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_20260715.md`
- Source SHA-256: `C256F68ADD9B3316D90D76557357550054C9B9B75D8BF85F91D11471E422CA49`
- Reviewed immutable commit: `6d88610da34a695e07d29c5e08b50fb57d2aa5e9`
- Review verdict: `REVISE`, `PROFILE=BLOCK`, `FORMAL_TRAINING=BLOCK`
- Review next step: `FIX_BEFORE_PROFILE`
- Review open P0: `P0-LAUNCH-WORKDIR`
- Review open P1s: feature provenance, chunk clock, one-factor trace,
  reporting population, and transaction/DDP.

The response available to the reviewer did not include the external B0 bundle
or the prior signed review artifact. Locally, the exact commit was bound to B0
artifact SHA-256
`817445589263408B1B0C0A5D59B698EC7437032FFD47FFF6227111446E93534D`,
with 508 tests, zero blocking findings, and zero protocol violations. That
evidence remains useful, but it does not invalidate a launch scenario that the
B0 matrix did not exercise.

## Independent Verdict

**Verdict: REVISE; profile and formal training remain blocked. I accept the
central gate decision, but I do not accept every recommendation as written.**

The review found a real pre-profile defect:

> The launch ticket freezes an exact runtime override map before the Slurm
> helper creates its timestamped run directory, while the helper later injects
> that directory as `work_dir`; the pre-CUDA validator requires exact equality.

The previous `PASS / PROFILE=ALLOW` is therefore superseded. A passing B0 is
not enough when its test manifest lacks the real submit-shell composition.

The scientific scope is also correctly narrowed:

> Q2 is a cached-feature, chunk-truncated temporal binding study. It is not a
> raw-video end-to-end method, and it has no effectiveness or novelty result.

The review overstates one engineering blocker. The registered route is locked
to one process and one GPU. It already checks finite loss and gradients,
restores model/optimizer/scheduler/scaler/runtime mutations after injected
failures, rejects skipped authenticated optimizer steps, and blocks formal
resume. Multi-rank consensus and exact resume continuation are therefore future
capabilities, not P1 requirements for the current Q2 experiment.

## Point-by-Point Disposition

| Review item | Disposition | Independent reason |
|---|---|---|
| `P0-LAUNCH-WORKDIR` | **Accept; profile blocker** | `build_full_petal_launch_ticket.py` records exact `cfg_overrides`; `submit_full_petal_q2_n16r4.sh` creates `RUN_DIR` later and always passes a new `work_dir`; `validate_full_petal_launch` compares the complete runtime identity. The current order has no deterministic single source for the value. |
| External B0 and prior review were `UNKNOWN` to Pro | **Accept as reviewer-local epistemic status; not a project unknown** | The artifacts were not attached to that Pro session. They were independently validated locally, but they missed this shell integration case and cannot override the new P0. |
| Current Q2 is not raw-video end-to-end | **Accept** | `raw_video_finetuning=False`; the detector consumes fixed 768-dimensional cache tokens. Visual-backbone adaptation is not in the graph. |
| Cross-chunk state is detached | **Accept** | `detach_stream_state=True` preserves forward state values but prevents later chunks from backpropagating through earlier state construction. The honest scope is cached-feature truncated temporal training. |
| `P1-FEATURE-PROVENANCE` | **Accept for strict-online and paper claims; qualify for cost profile** | The cache manifest binds arrays, hashes, source frames, stride, and an encoder ID, but not an immutable extractor commit/config/checkpoint, raw support interval, normalization scope, or a raw-future invariance test. This blocks genuine-video-level causality claims. It does not by itself prevent a compute-only profile after P0 is fixed, provided the profile is explicitly non-effectiveness evidence. |
| `P1-CHUNK-CLOCK` | **Accept** | The model is numerically episode/step equivalent, but the test/evaluation config delivers up to 64 tokens at once and records `emit_time_sec=emit_frame/fps`. It does not represent when the packet became available. Source-clock latency must not be presented as wall-clock deployment latency. |
| `P1-ONE-FACTOR-TRACE` | **Qualify and retain as a formal-experiment gate** | Existing code and tests already compare matched weights, birth assignments, canonical lifecycle, birth/alive masks, endpoint slots, losses, and changed gradients. The review is wrong if read as saying no one-factor evidence exists. It is right that a single paired trace should additionally lock negative masks, ignore/censor masks, denominators, reset events, RNG consumption, overflow behavior, and inference equality. |
| `P1-POPULATION-211` | **Accept for formal reporting** | The code intentionally distinguishes a historical 211-ID lock from an observed 213-ID universe and requires an explained comparison artifact. The actual two IDs and reasons remain external. Until verified, results must be named as a 211 subset and every baseline must be reevaluated on the same population. This does not block a training-cost profile. |
| `P1-TRANSACTION-DDP` | **Reject as a current P1; retain a scope guard** | Q2 locks `world_size=1` and the launcher uses `--nproc_per_node=1`. Single-rank nonfinite-gradient, scaler, optimizer-step, commit-failure, rollback, scheduler, and event-count paths already have fault-injection tests. Formal resume is explicitly blocked. The correct action is to keep multi-rank and resume unsupported and fail closed, not implement DDP consensus before the one-GPU kill test. |
| `P2-END-TO-END-SCOPE` | **Accept** | Only `CACHED_FEATURE_TEMPORAL` is defensible. `RAW_VIDEO` and full-stream BPTT claims are prohibited. |
| `P2-IDENTITY` | **Accept as a claim gate, not a code bug** | Persistent slot addresses and hidden states exist; learned instance identity requires permutation, rebinding, overlap, fragmentation, and switch interventions. |
| `P2-LEGACY-HEAD` | **Partially accept** | Pointer/hazard modes are inactive in Q2 and must not be counted as contributions. Deprecating or deleting them is optional; a Q2 call-graph assertion and honest documentation are sufficient. |
| `P2-METRIC` | **Accept with hierarchy** | Standard temporal AP/mAP and explicit completion delay remain primary. Budgeted/identity metrics are diagnostics unless formally shown equivalent to an accepted protocol. |
| `P2-FINEACTION` | **Qualify** | `None` is unclear in the base config, but the result gate already requires a source-derived PASS artifact when a FineAction-dependent claim is present. Add an explicit `disabled` or `required` mode; do not treat this as a THUMOS-only Q2 blocker. |
| `P2-DOC-DRIFT` | **Accept** | The wiki demotes and retires Full PETAL while execution files reuse the name. Q2 must be recorded as an experimental binding study under the demoted parent route, not as a revived paper method. |
| `P3-PYTHON-UNROLL` | **Accept as a measured cost risk** | The current temporal path is sequential Python execution, not prefix-parallel training. This is precisely what the fixed-step profile should quantify after P0 closes. |
| Full package is `RECONSTRUCTION` | **Provisionally accept** | MATR/ActionSwitch occupy direct On-TAL memory/lifecycle space, E2E-LOAD occupies raw-video end-to-end OAD, and TrackFormer establishes persistent identity queries. The fixed-binding delta remains a marginal hypothesis until a current, cited closest-work matrix and matched experiments are complete. |
| Run the entire proposed baseline/ablation matrix before any pilot | **Reject** | The list is useful as a paper-stage inventory but too broad for the present kill gate. Start with the smallest matched set that distinguishes fixed binding, rematching, nonpersistent temporal state, and a simple recurrent control. Expand only if the mechanism survives. |
| Adopt the proposed `-1.0 pp` LCB or `20%` effect threshold as scientific law | **Reject** | Those values lack a power analysis and mix noninferiority with innovation. Freeze a resource gate only after defining the primary metric, expected variance, minimum meaningful effect, and paired analysis. |

## Verified Code Corrections to the Review

The following implementation facts narrow the review's open claims:

1. `tests/test_full_petal_detector_contracts.py` already proves episode-versus-
   incremental output equivalence for the detector and checks future-token
   perturbation at the cached-feature level.
2. `tests/test_prefix_trajectory_supervision.py` already covers fixed/rematch
   births, same-class overlap, endpoint retirement, slot exhaustion, ties,
   delayed births, and transactional rollback of supervision state.
3. `opentad/cores/train_engine.py` checks finite scalar costs and parameter
   gradients both before and after unscale/normalization/clipping.
4. `tests/test_full_petal_training_transaction.py` injects nonfinite costs,
   unscale/step/update faults, commit failures, and verifies rollback of the
   mutable training components.
5. `opentad/utils/full_petal_runtime_attestation.py` rejects an authenticated
   event when the optimizer post-step hook did not fire, including a skipped
   scaler step.
6. The current launch contract and Slurm helper both lock execution to one
   process. No DDP scalability claim is active.

These facts do not close the new P0, raw-feature provenance, real packet clock,
or reporting-population questions.

## Revised Gate Order

1. Fix `P0-LAUNCH-WORKDIR` by deriving `RUN_DIR`, `WORK_DIR`, ticket overrides,
   and Slurm argv from one immutable value before ticket publication, or by
   reading the runtime override directly from the ticket.
2. Add a real submit-shell integration test with deterministic time and a fake
   `sbatch`; mutate `work_dir` and prove pre-CUDA rejection.
3. Rename the executable route to a neutral Q2 binding-study identity and add a
   versioned route-status contract.
4. Add a cache-extractor provenance artifact and raw-future prefix-invariance
   test before any strict-online or effectiveness claim.
5. Separate training chunk size from serving/evaluation packet size. Use
   stepwise serving or record source, availability, and completion clocks
   explicitly.
6. Complete the one-factor paired trace, reusing the audit fields that already
   exist rather than rebuilding supervision.
7. Bind the two 211-versus-213 IDs and reasons; use one identical population for
   every reported comparison.
8. Keep world size one and resume disabled. Do not add DDP/resume engineering
   unless a later experiment actually requires them.
9. Freeze a new clean commit, regenerate the full B0 artifact, and send the
   exact commit and B0 to the same independent review gate.
10. Only a new `PASS / PROFILE=ALLOW` authorizes the 50-warmup plus 200-measured
    optimizer-event profile. Formal training remains separately blocked.

## Research Status After Absorption

- Task: standard fully supervised completion-triggered On-TAD remains fixed.
- Active scientific object: post-birth fixed supervision binding versus prefix
  rematching on one fixed cached-feature detector.
- Current method name: use `Q2 Persistent Binding Study`; do not use Full PETAL
  as a paper-method claim.
- Genuine raw-video online status: unproven until extractor provenance closes.
- End-to-end scope: cached-feature temporal, truncated across chunks.
- Effectiveness: no result.
- Novelty: full package provisionally reconstruction; fixed binding is an
  unproven marginal candidate.
- Profile: blocked by `P0-LAUNCH-WORKDIR`.
- Formal training: blocked by P0, scientific protocol gaps, and lack of profile.

## Final Disposition

```text
INDEPENDENT_VERDICT=REVISE
ACCEPT_PRO_CENTRAL_GATE=YES
ACCEPT_ALL_PRO_RECOMMENDATIONS=NO
PROFILE=BLOCK
FORMAL_TRAINING=BLOCK
NEXT_STEP=FIX_P0_LAUNCH_WORKDIR
OPEN_P0=P0-LAUNCH-WORKDIR
OPEN_P1=P1-FEATURE-PROVENANCE,P1-CHUNK-CLOCK,P1-ONE-FACTOR-TRACE,P1-POPULATION-211
DOWNGRADED=P1-TRANSACTION-DDP
```
