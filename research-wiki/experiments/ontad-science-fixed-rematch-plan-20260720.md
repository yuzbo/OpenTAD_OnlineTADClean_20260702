# Clean Scientific FIXED/REMATCH Implementation Plan

Date: 2026-07-20

Active recovery and execution evidence is append-only in
[`ontad-science-fixed-rematch-execution-20260720.md`](ontad-science-fixed-rematch-execution-20260720.md).
That record must be updated at each core-repair, local-validation, Slurm-smoke,
paired-profile, and final-decision checkpoint.

## Phase 1: Branch and Scope

- [x] Create `codex/ontad-science-fixed-rematch` from `b974f3d`.
- [x] Freeze the scientific design and experiment gates.
- [x] Record the exact curated source-file map.

## Phase 2: Minimal Scientific Port

- [x] Port prefix trajectory supervision and its focused tests.
- [x] Port the persistent trajectory detector without unrelated dependencies.
- [x] Reuse and revise the existing persistent event-set head.
- [x] Port only the chronological feature-data changes required by the route.
- [x] Port and rename the required instance metrics.
- [x] Add FIXED and REMATCH configs that differ only in binding mode.
- [x] Update registries with only the new scientific classes.

## Phase 3: Lifecycle Repair

- [x] Make canonical supervision capacity independent of runtime occupancy.
- [x] Replace refractory occupancy with one-step CANDIDATE confirmation.
- [x] Freeze the entry-free birth pool and reuse released slots next step.
- [x] Limit same-bin candidate admission to the frozen causal birth census.
- [x] Add explicit dropped-target, capacity, and non-degeneracy counters.

## Phase 4: Verification

- [x] Run source compilation.
- [x] Run supervision, lifecycle, metric, dataset, and config tests in the
      N16R4 PyTorch environment (55 focused tests passed on 2026-07-20).
- [x] Cover repeated and overlapping same-class actions with focused synthetic
      supervision and lifecycle tests.
- [ ] Run an end-to-end adjacent-action synthetic stream through dataset,
      detector, emission serialization, and evaluator.
- [x] Verify no inference annotation, terminal, or future fields.
- [x] Verify the two resolved configs differ only in binding mode.

## Phase 4.5: Scientific-Contract Repair

- [x] Archive and independently absorb the readiness review pinned to
      `27a59de`; recheck every P0/P1/P2 item against `95fa963`, smoke job
      `1176737`, and profile job `1176983`.
- [x] Make binary endpoint emission equal the current decision frame and
      remove the unsupervised endpoint offset from the primary output.
- [x] Hold newborn loss binding on the canonical birth slot for the birth
      decision; permit REMATCH only from the next causal decision.
- [x] Commit a newborn whose birth and end cross in the same step, including a
      final-token short action, exactly once.
- [x] Implement the exact adjacent-action synthetic stream, including
      same-step old-end/new-birth and released-slot deferral.
- [ ] Execute that end-to-end regression in the N16R4 PyTorch environment.
- [x] Split video identity from runtime stream identity and normalize all
      instance metrics to one explicit coordinate system.
- [x] Replace chronological greedy primary matching with the frozen global
      one-to-one policy and define fragmentation by disjoint covered
      components, not merely distinct bounds.
- [x] Freeze a standard-mAP percentage-point schema and make the result gate
      reject unit, metric, tIoU, or provenance mismatches.
- [x] Implement fit-only training, symmetric calibration/checkpoint freezing,
      and a one-shot locked reporting command.
- [x] Generate one repository-owned result artifact that joins audit counters,
      standard and instance metrics, latency, resource use, and SHA-256
      provenance.
- [x] Separate supervision exhaustion, runtime birth collision, arbitration
      suppression, cancellation, and abandonment; make formal failures
      atomic.
- [x] Generate and consume a hashed split census for births per step,
      same-bin end+birth, last-token coverage, delayed-reuse headroom, and
      clipped starts.

## Phase 5: Feature Falsification

- [x] Verify the annotation, class map, cached-feature manifest, and frozen
      160/40/211 split files on N16R4.
- [x] Run a Slurm end-to-end train/inference smoke job
      (`1176737`, commit `097bc72`, gate passed).
- [x] Run the frozen paired 50-warmup/200-measured deterministic profiler
      (`1176983`, commit `95fa963`).
- [x] Apply the frozen cost gate: stability and causal-equivalence passed, but
      the 12-epoch pair was estimated at 12.572 GPU-hours versus the 2-hour cap.
- [x] Complete the implemented Phase 4.5 contracts and rerun the repaired
      Slurm smoke (`1177438`) plus strict paired profiler (`1177511`).
- [x] Register a one-epoch seed-705 convergence/non-degeneracy screen at
      `1.691339 GPU·hours`, including the frozen safety factor and a
      conservative locked-report reserve, under the unchanged 2-hour cap.
- [x] Implement and locally validate the screen-only launcher, result
      artifact, calibration-only gate, and hash-linked evidence contract.
- [ ] Rerun smoke and strict profiling after model-optimization commit
      `0258b853aa284f9650d931e680116b63b289e192`.
- [ ] Validate the adjacent-action and complete Torch test bundle on N16R4.
- [ ] Run the registered seed-705 technical screen without accessing the
      reporting split.
- [ ] BLOCKED: freeze validation-selected thresholds.
- [ ] BLOCKED: run seeds 705, 706, and 707 through Slurm.
- [ ] Report paired standard and instance metrics.
- [ ] Apply the frozen technical and scientific gates without post-result edits.

## Phase 6: Conditional Raw RGB

- [x] Keep raw RGB blocked after the pre-training budget gate failed.
- [ ] Stop or demote the method only if a valid feature effectiveness gate fails.
- [ ] If both gates pass, add the causal raw-frame encoder path.
- [ ] Compare frozen encoder and PEFT/joint training without changing lifecycle.
