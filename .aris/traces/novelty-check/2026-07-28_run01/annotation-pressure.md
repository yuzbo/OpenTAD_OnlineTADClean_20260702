# THUMOS14 train identity-pressure audit

Date: 2026-07-28

Input:

- `/data/run01/sczc063/yuzibo/eventmatr_data/thumos14_v2.json`
- bytes: `1,575,585`
- SHA-256:
  `8eb3e61cc758bcc08aea1d17cfbf1acb2fed8c2a51ed884116d766e9e4c04e66`
- subset: all and only records with `subset == "train"`

Method:

1. Read each annotation's integer `segment_frame=[start,end]` and
   `labelIndex`.
2. Enumerate unordered annotation pairs within each video.
3. A same-class overlap has
   `min(end1,end2)-max(start1,start2) > 0`.
4. For non-overlapping same-class pairs, gap is
   `max(start1,start2)-min(end1,end2)`.
5. Count pairs, distinct participating annotations, and videos at overlap,
   overlap-or-gap-`<=16`, and overlap-or-gap-`<=64`.
6. Compute concurrency with end events ordered before start events at the same
   frame, so touching intervals are not counted as overlap.

Results:

| Quantity | Count |
|---|---:|
| train videos | 200 |
| annotations | 3,007 |
| classes | 20 |
| videos with repeated same-class instances | 180 |
| true same-class overlap pairs | 5 |
| videos with true same-class overlap | 2 |
| annotations participating in true same-class overlap | 9 (0.299%) |
| overlap or same-class gap <=16 frame pairs | 453 |
| participating annotations at <=16 | 677 (22.51%) |
| videos at <=16 | 56 |
| overlap or same-class gap <=64 frame pairs | 1,348 |
| participating annotations at <=64 | 1,509 (50.18%) |
| videos at <=64 | 102 |
| different-class overlap pairs | 245 |
| videos with any overlap | 23 |
| maximum simultaneous annotated instances | 2 |

The five natural same-class overlaps occur in two videos and two classes:
three `FrisbeeCatch` pairs and two `VolleyballSpiking` pairs.

Interpretation:

- The standard THUMOS14 train split does not support an “abundant simultaneous
  same-class actions” premise.
- It does support a substantial nearby repeated-instance association problem.
- T/TH should therefore be judged primarily on nearby-repeat owner stability,
  fragmentation, duplicate rebirth, cancellation, and reacquisition.
- Same-class overlap remains a hard correctness/stress stratum, but a positive
  result on five natural pairs cannot carry a headline empirical claim.
