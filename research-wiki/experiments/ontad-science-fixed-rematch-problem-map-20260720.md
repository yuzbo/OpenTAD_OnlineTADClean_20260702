# Clean Scientific FIXED/REMATCH Problem Map

Date: 2026-07-20

## Research Question

Does fixing each visible action instance to its first-crossing supervision slot
reduce duplicate and fragmented Online-TAD outputs compared with per-prefix
rematching, when every other factor is held constant?

## Confirmed Facts

- The final task is standard, fully supervised, strictly causal On-TAD.
- The immediate experiment is feature based, not raw RGB.
- Raw-RGB joint training is conditional on the feature gate.
- The clean branch starts from `b974f3d`.
- The historical implementation cannot be merged wholesale.
- FIXED and REMATCH must differ only in post-birth loss binding.
- The previous four-slot controller exhausted 2,206 GT births.
- The annotation census required only two slots.
- False ACTIVE and refractory occupancy caused the exhaustion.
- A nearly silent birth controller is not a valid zero-exhaustion solution.

## Code Problems to Close

Closed structural problems:

1. The clean detector no longer imports the historical route as a unit.
2. Canonical supervision capacity is independent of runtime prediction
   occupancy.
3. Post-completion refractory occupancy has been removed.
4. Runtime births are bounded by a shared two-candidate arbitration rule.

Open scientific-contract problems, independently confirmed against commit
`95fa963`:

1. Binary endpoint emission still depends on an `endpoint_offset_head` that has
   no endpoint-offset loss.
2. REMATCH may move a newborn's class/start/end loss away from its canonical
   birth slot on the birth step itself.
3. A newborn with an end crossing in the same decision step is admitted only
   after end processing and therefore cannot commit; last-token short actions
   may disappear.
4. Entry-free slot freezing makes a slot released in the current step
   unavailable to a same-step adjacent birth. This is a deliberate controller
   choice but is not yet proven harmless.
5. The generic training runner constructs the locked reporting dataset and,
   with `val_eval_interval=1`, evaluates it every epoch.
6. The instance evaluator mismatches composite runtime stream keys with
   ground-truth video IDs and can compare prediction frames with GT seconds.
7. Its chronological greedy matching and "different bounds" fragmentation
   rule do not implement the frozen best-match/disjoint-component definition.
8. `OnlineAPBudgeted.average_mOnlineAP` and
   `persistent_binding_gate.average_map` do not share a frozen metric schema,
   range, tIoU set, or percentage-point unit.
9. A formal run cannot yet emit one repository-owned, hash-linked result row
   joining checkpoint, config, manifest, ledger, audit counters, standard mAP,
   instance metrics, latency, and resource use.
10. `formal_training_ready=False`, canonical exhaustion, and the distinction
    among supervision exhaustion, runtime collision, arbitration suppression,
    cancellation, and abandonment are not all enforced fail-closed by the
    generic runner.
11. The exact birth/end/cache-coverage/start-history census is not yet a hashed
    launcher-consumed artifact.
12. Threshold calibration and final-checkpoint selection are not yet frozen as
    one symmetric, reporting-blind algorithm.
13. The strict deterministic profiler estimates the registered 12-epoch pair
    at 12.572 GPU-hours, so the current protocol cannot enter the frozen
    2-hour one-seed gate.
14. No completed formal FIXED/REMATCH effectiveness result exists.

The complete source and point-by-point disposition are archived in
`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md` and
`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`.

## Curated Scientific File Map

- supervision: `opentad/utils/prefix_trajectory_supervision.py`;
- runtime head: `opentad/models/dense_heads/persistent_event_set_head.py`;
- detector: `opentad/models/detectors/persistent_trajectory_ontad.py`;
- chronological feature data:
  `opentad/datasets/streaming_feature.py`;
- instance metrics: `opentad/evaluations/online_instance_metrics.py`;
- frozen result gate:
  `opentad/evaluations/persistent_binding_gate.py`;
- matched configs:
  `configs/causaltad/thumos_persistent_binding_fixed.py` and
  `configs/causaltad/thumos_persistent_binding_rematch.py`;
- focused tests: `tests/test_prefix_trajectory_supervision.py`,
  `tests/test_persistent_event_set_head.py`,
  `tests/test_persistent_trajectory_detector.py`,
  `tests/test_online_instance_metrics.py`,
  `tests/test_persistent_binding_gate.py`,
  `tests/test_persistent_binding_configs.py`, and
  `tests/test_streaming_feature_dataset.py`.

## Required Decisions Encoded by the Design

- Supervision and runtime states are independent.
- Candidate births require one causal step of support.
- At most two candidates enter in one step.
- A slot released this step becomes birth-eligible at the next decision.
- Completed slots return directly to FREE.
- `E_id` and its 20% relative-reduction formula are frozen.
- Silent-output rejection uses prediction/GT ratio and minimum recall.
- Three matched seeds are required before a raw-RGB stage can begin.

## Stop Conditions

Stop or demote the FIXED hypothesis if:

- the repaired lifecycle still drops supervised births;
- a valid run requires more slots or output suppression to avoid exhaustion;
- either arm becomes silent or nearly silent;
- FIXED fails to reduce combined identity error by 20%;
- FIXED loses more than 0.5 mAP points;
- the comparison changes more than the binding rule;
- the proposed run exceeds its preregistered GPU-hour budget.
