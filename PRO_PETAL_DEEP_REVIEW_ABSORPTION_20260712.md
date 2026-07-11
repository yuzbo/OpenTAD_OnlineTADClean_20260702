# PETAL Deep Review Absorption, 2026-07-12

## Source

- Archived response: `PRO_PETAL_DEEP_REVIEW_20260712.md`
- Source SHA256: `74A5A88D0F0BFE389C974290FC6B118B834B13FE59E5D22231985F7018256D95`
- Prompt: `PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md`
- Public repository reviewed by Pro: commit `8165970c339fd10de9b1e6378c7261f754c3a2a0`
- Local code inspected during absorption: branch `codex/online-tad-clean-20260702`, with uncommitted PCEH correctness and finetune-candidate edits kept separate.

## Independent Verdict

**Verdict: REVISE, confidence 0.91.**

I accept the review's central research decision but do not accept every statement as written.

Accepted central decision:

> Do not implement or train Full PETAL. First run a matched-feature, low-cost falsification study comparing fresh queries, a faithful Temporal TrackFormer reconstruction, and a minimal prefix-observable persistent event-set decoder.

The surviving scientific question is narrower than the previous PETAL story:

> Under identical causal features, compute, capacity, thresholds, and evaluator, does On-TAD-specific persistent instance state improve detection because it preserves action identity, rather than because it adds generic tracking capacity, stronger visual representations, future-assisted assignment, or proposal suppression?

No PETAL effectiveness or novelty claim currently survives unconditionally.

## Point-by-Point Disposition

| Review point | Disposition | Independent reason |
|---|---|---|
| Full PETAL is largely reconstructible from StreamFormer/E2E-LOAD, MATR/ActionSwitch, and TrackFormer/MOTR | **Accept** | Primary sources verify raw causal representation, direct On-TAL instance decoding/state, and persistent identity queries separately. The intersection gap is not itself an irreducible method. |
| Raw-video training must remain frozen before the mechanism gate | **Accept** | Current cost is prohibitive and visual adaptation would confound the persistence claim. |
| Route C must be a kill gate, not a presumed paper method | **Accept** | This is the cheapest experiment that can falsify the proposed causal mechanism. |
| Current PCEH class-keyed state cannot represent repeated or concurrent same-class instances | **Accept; code-verified** | `PrefixEmissionState.active_tracks` is keyed by class label. |
| Current PCEH endpoint and emission targets are coupled | **Accept; code-verified** | `end_crossed` sets both `end_event` and `emit_event`; `emit_allowed` and `late_target` continue positive emission pressure. |
| Current PCEH predicts `end = emit = current_frame` | **Accept; code-verified** | `PrefixEventEmissionHead.decode_step` writes both fields from `current_frame`. |
| Full stream GT should not be passed through the detector's runtime-facing kwargs | **Accept as an audit boundary** | Training labels are legal, but the current mixed interface makes GT-taint review unnecessarily difficult. The replacement route uses a training-only prefix schedule constructed by the dataset. |
| Model forward must not receive EOF or terminal duration | **Accept for the new route** | Orchestration may clear state after return; prediction computation must not depend on terminal metadata. |
| Late predictions must remain in the ledger | **Accept** | Dropping them makes latency and false-positive accounting optimistic. |
| Temporal TrackFormer is the novelty-killer baseline | **Accept** | TrackFormer directly establishes static birth queries, autoregressive identity-preserving track queries, and set interaction. |
| ActionSwitch is the strongest direct-task competitor | **Accept** | Its stated purpose includes simultaneous and same-class overlapping actions in streaming On-TAL. |
| Prefix-parallel execution is not a main scientific contribution | **Accept** | It is an execution property unless a new parallel scan or learning principle is required. It remains a measured systems property. |
| Full-trajectory assignment is automatically future-label leakage | **Do not accept literally** | Offline supervised training may legally use complete annotations while keeping model inputs causal. The real risk is privileged assignment and attribution. Prefix-observable matching is the main method; full-trajectory matching remains a clearly labeled upper-bound ablation, not a protocol violation by definition. |
| Class supervision should be delayed until endpoint observation | **Reject for the Stage-1 main setting** | The review is internally inconsistent: its method loss delays class loss until end, while its technical audit allows supervision from GT start. Stage 1 supervises class after the start becomes observable and includes an end-only ablation. |
| Start must use a distributional pointer | **Partially accept** | It is a plausible On-TAD-specific mechanism for long and pre-memory actions, but must be compared against a matched scalar-offset TrackFormer baseline. It is not assumed superior. |
| End hazard and emission should happen at the same decision step | **Accept for the completion-triggered main protocol** | This minimizes avoidable decision delay. A delayed confirmation policy can be an ablation only. |
| No explicit duplicate loss in the first pilot | **Accept** | Adding repair losses before the set mechanism works would obscure whether persistence itself helps. |
| `+2.0 avg-mAP` or `20%` error reduction is mandatory scientifically | **Accept only as a project gate** | These thresholds are useful resource-allocation rules, not universal significance laws. Paired uncertainty and non-inferiority remain required. |
| Three seeds are enough for a final claim | **Partially accept** | Three seeds are the minimum kill test. A retained headline result should use five seeds when affordable and paired per-video uncertainty. |
| Single-rank Stage 1 is acceptable | **Accept** | The current sampler rejects DDP. Correct video-owned DDP needs equal-step coordination and is deferred rather than improvised. |
| The PETAL acronym must be retired | **Accept** | A prior TAL method already uses PETAL. The implementation uses the descriptive name `PersistentEventSet` until evidence justifies a paper name. |

## Primary-Source Verification

Verified on 2026-07-12:

- TrackFormer describes static object queries for births and autoregressive identity-preserving track queries.
- ActionSwitch defines On-TAL as emitting instances when actions conclude and targets simultaneous, including same-class, actions.
- MATR defines immutable past On-TAL predictions, current-segment end detection, memory-based start localization, and query-based instance decoding.
- E2E-LOAD provides trainable-backbone raw-video online action detection, but its task is frame-level OAD.
- StreamFormer provides a causal streaming video backbone and evaluates OAD/OVIS/VQA, not standard instance-level On-TAL.
- `PETAL` is already the acronym for Prior-enhanced Temporal Action Localization.

These sources support the review's obvious-combination attack. They do not prove that the proposed On-TAD-specific mechanism is useless; that is the purpose of the matched pilot.

## Repository Verification

Focused tests before the new implementation:

```text
32 passed, 4 skipped
```

Passing tests validate the current contracts, not scientific correctness. Several tests encode the current coupled target and class-keyed state and therefore do not refute the Pro diagnosis.

Remote N16R4 audit:

- the main remote checkout is dirty and must not be reused or reset;
- no standard THUMOS I3D cache is currently ready at the configured path;
- THUMOS annotations and raw videos are present;
- a public-feature download is incomplete;
- deployment must use a fresh checkout and a separate run directory;
- existing unrelated Slurm jobs must not be cancelled or modified.

## Previously Missing Negative Evidence

The N16R4 workspace contains an earlier, different-task experiment:

```text
R105 Temporal Persistence Event-Set
task: query-free full-video event-set decoding
status: fail
```

Its full persistence model failed to beat a matched plain-energy capacity control and several destructive controls. This is not a direct On-TAD result and cannot kill the present route, but it is strong anti-hype evidence:

> Temporal persistence is not intrinsically useful; visual evidence, assignment, lifecycle semantics, and matched controls determine whether it helps.

The Stage-1 route must therefore include feature-shuffled/identity-shuffled diagnostics and cannot treat persistence as a contribution before results.

## Frozen Research Decision

1. Full PETAL is demoted from lead method to a conditional long-term integration route.
2. The immediate route is `PersistentEventSet` Stage 1 on fixed cached causal features.
3. Main comparison: fresh queries vs Temporal TrackFormer vs persistent event-set.
4. Main assignment: prefix-observable birth matching with fixed identity after birth.
5. Full-trajectory matching: privileged upper bound only.
6. Main protocol: completion-triggered, immutable On-TAD; no future endpoint prediction.
7. Main metrics: standard temporal mAP plus recall/FN, duplicate, fragmentation, same-class overlap, chunk crossing, slot exhaustion, latency, VRAM, wall time, and GPU-hours.
8. Raw-video PEFT remains blocked until the feature mechanism passes.
9. The route is killed if Temporal TrackFormer is equivalent or stronger within uncertainty, or if any gain lacks the claimed error-mode improvement.

## Immediate Engineering Scope

Implement only:

- prefix-observable instance schedules;
- strict no-EOF feature packets;
- a shared query decoder with controlled fresh/track/persistent variants;
- first-event endpoint targets and one-time immutable emission;
- synthetic same-class, overlap, chunk-crossing, slot-exhaustion, duplicate, and rearm tests;
- a frozen feature cache extractor and manifest;
- matched configs and a fail-closed Slurm smoke/pilot launcher;
- result artifacts that can enforce the kill rule.

Do not implement visual finetuning, LoRA, extra lifecycle heads, duplicate repair losses, full-future assignment as the main method, or a 30-epoch raw-video run.
