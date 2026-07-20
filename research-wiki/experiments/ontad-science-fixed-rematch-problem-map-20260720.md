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

1. The historical detector imports components outside the scientific method.
2. Runtime predicted occupancy can remove a real training target.
3. Refractory slots consume capacity after an interval is complete.
4. Independent birth logits can activate too many free slots in one step.
5. The old result gate does not fully reject silent output.
6. The combined duplicate/fragmentation improvement needs one frozen formula.
7. The historical Q2 configs are cached-feature only and cannot support a
   raw-RGB claim.
8. No completed formal multi-seed FIXED/REMATCH effectiveness result exists.

## Required Decisions Encoded by the Design

- Supervision and runtime states are independent.
- Candidate births require one causal step of support.
- At most two candidates enter in one step.
- End and release happen before same-bin birth admission.
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
- the comparison changes more than the binding rule.
