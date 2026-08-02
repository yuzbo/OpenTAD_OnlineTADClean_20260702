# EventMATR D1.5 v2 Final Integrity Audit — 2026-08-02

Scope: independent read-only audit of the source-exact D1.5 v2 scan, the
finalizer-only recovery, its final receipt, the registered statistics and the
paper-validity boundary. This audit does not reclassify the diagnostic as a
performance result.

## Verdict

Overall grade: **A for diagnostic integrity; not eligible as paper performance**.

The evidence supports `PASS_DIAGNOSTIC`: provenance, checkpoint immutability,
train-only data access, chronological route consumption, positive-control
closure, trace reconstruction and preregistered event-level statistics all
close. The routing result is valid. The same evidence explicitly forbids model
implementation, model training, official comparison, locked-test release and
paper claims.

## A. Source and provenance — A

- start and final identities are byte-identical, clean and `PASS`;
- diagnostic source is
  `dc530e2d99f928b1455a6e137f708306cc00d91c`, tree
  `14e9aeb886e85df5dfb2fd3b321f39597eb040c9`;
- manifest SHA-256 is
  `27d6e1143812c3c4033c5a0d39cc17d58667f6727cde3d7c1d4e61f84747f9af`;
- start/final identity SHA-256 is
  `87d1c195f061fbcf10c36409b745ce08fa6b7764bdbb12ec740136ffff25c00c`;
- training source remains
  `92cf34aa07bebee2a7a7e3661431d5055804b29b` and the registered D1.4 source
  remains separately bound.

The permanent wrapper repair is later commit
`0419a0ff94ea26513a5282f2be90e184cdffd3c6`, tree
`94dcb480575231737cad922a23585bb72d5ad521`. Its diff from the scan source is
limited to module-mode finalizer invocation and a regression assertion; the
statistics and finalizer scientific logic are unchanged.

## B. Data partition and test access — A with naming disclosure

- receipt reports `train_only=true` and `test_access=false`;
- locked-test, official-comparison and paper-claim releases are all false;
- official census closes at 200 videos, 3,007 annotations, 3,003 visible births,
  3,001 observable endpoints, two right-censored events, four unobservable
  events and 56,551 alive-supervision rows;
- the feature artifact filename contains `val`. This is the manifest-bound
  THUMOS training/validation split, not evidence of locked-test access. The
  naming is disclosed because it can otherwise be misread.

## C. Execution and checkpoint immutability — A

- `optimizer_step_count=0`, `checkpoint_updated=false`;
- start/final checkpoint size, timestamp and SHA checks are enforced by the scan;
- scan SHA-256 is
  `92d816b1e6c8d4890757c0147f41d3b33ab3652223b3ab71e42340da1d3d90c1`;
- trace is 4,359,073,296 bytes with SHA-256
  `4ed11c6ed002cccb05191f23eb5e9d50eb5d438c2b91bbde6526fc56fc05f528`;
- receipt is 33,961,025 bytes with SHA-256
  `f28d2ca243ed5bfc68f4cdf2d0dd20eb13899c6ccd3edc929f81d1208990a358`;
- recovery job `1214235` completed `0:0` with empty stderr.

## D. Causal and lifecycle contracts — A for the diagnostic

- all eight channel/route consumers bind to common query and route-consumption
  hashes;
- the finalizer reconstructs chronological per-video route traces, unique owner
  decisions, state-logit margins, target distances, transitions, EOS survivors
  and sidecar linkage;
- oracle admission has exact target coverage and zero semantic duplicates;
- positive control closes `3003 = 3001 + 2` without force-ending the censored
  events;
- there is no capacity exhaustion, silent record loss or END-without-emission
  closure fault.

Ground-truth sidecars deliberately intervene after model forward. Therefore the
experiment is a causal mechanism diagnostic, not deployable inference and not
paper performance.

## E. Statistical contract — A

- primary outcome is one paired target event with target-backed END plus
  immutable emission inside a symmetric 64-step window;
- right-censored events are excluded; unresolved and ambiguous identity remain
  explicit failures rather than being relabelled as false actions;
- formal comparisons and no-cancel comparisons are separate families;
- five-percentage-point effect floor, 10,000 video-cluster bootstrap samples,
  10,000 video-cluster sign flips, deterministic seeds and Holm familywise
  correction are frozen in code and protocol;
- an independent recomputation equals the stored paired object exactly; both
  canonical objects hash to
  `65c9471037baa3880caff9c8600a1db47d701c0ecc59456b161ba606cc0250a2`.

## F. Recovery-chain integrity — A with evidence-scope note

Job `1213435` completed the scan but failed only at direct-file finalizer import.
Recovery reused the unchanged scan, identities and trace, validated their exact
hashes, reread the full trace, and wrote the previously absent receipt. The
local audit copy contains the scan, receipt, identities, launch script and
Slurm logs, but not a second 4.36-GB trace copy; trace contents are covered by
the finalizer's full reconstruction and the frozen remote hash chain. This is a
scope disclosure, not evidence that the trace changed.

## Final boundary

Accepted conclusion: no preregistered admission, identity-refresh or no-cancel
contrast produced a material, uncertainty-qualified endpoint effect. The only
authorized next action is a preregistered read-only diagnostic refinement.

Rejected conclusions: model improvement, performance gain, paper-ready result,
novelty confirmation, locked-test validity, or authorization to train.
