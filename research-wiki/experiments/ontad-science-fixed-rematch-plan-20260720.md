# Clean Scientific FIXED/REMATCH Implementation Plan

Date: 2026-07-20

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

## Phase 5: Feature Falsification

- [x] Verify the annotation, class map, cached-feature manifest, and frozen
      160/40/211 split files on N16R4.
- [x] Run a Slurm end-to-end train/inference smoke job
      (`1176737`, commit `097bc72`, gate passed).
- [ ] Run one matched seed as a convergence and non-degeneracy screen.
- [ ] Freeze validation-selected thresholds.
- [ ] Run seeds 705, 706, and 707 through Slurm.
- [ ] Report paired standard and instance metrics.
- [ ] Apply the frozen technical and scientific gates without post-result edits.

## Phase 6: Conditional Raw RGB

- [ ] Stop or demote the route if either feature gate fails.
- [ ] If both gates pass, add the causal raw-frame encoder path.
- [ ] Compare frozen encoder and PEFT/joint training without changing lifecycle.
