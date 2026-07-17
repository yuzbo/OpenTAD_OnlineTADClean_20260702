# Protocol V2 Academic Method and Implementation Reassessment

## Fixed scope

- Reviewed at: `2026-07-17T11:07:51Z`
- Scope: academic method, statistical definition, code implementation, and
  reproducibility only
- Repository: `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`
- Branch: `codex/full-petal-implementation`
- Frozen commit: `18cc27b8496f5effa7e4e93abca58da4e7ea3d0b`
- Frozen tree: `60ccc875203788af28ac54989defa443fc53d269`
- Protocol SHA-256:
  `b948dd3212a02ad11ba1d7b9ab9010407a8e991624d558631f53a5acc325fe8a`
- Source manifest SHA-256:
  `c23f00c71e72b4f4de3d27407a9879b035f2d34c122ba44bede22952e0eaa437`
- Verdict: `REVISE_PROTOCOL_BEFORE_COLLECTION`

This was a read-only, outcome-blind, zero-GPU review. No real R0/R1 result,
model output, checkpoint, profile, training output, or GPU workload was read or
generated. Temporary synthetic inputs were used only for validator contract
tests. This report does not create, validate, or deliver identity or signature
artifacts.

## Findings

### P0: Population provenance can be fabricated

`validate_population_bundle` verifies caller-supplied 211- and 213-ID files,
recomputes their set difference, checks coverage by caller-supplied reason
rows, and hashes each cited source. It does not parse a cited source or derive
any ID membership or reason from it. The only content condition is that the
source payload is nonempty
(`opentad/utils/prefix_route_protocol_v2.py:1523-1667`, especially
`:1635-1647`).

The bound positive fixture uses arbitrary synthetic IDs and an opaque text file
containing `official canonical membership source`
(`tests/test_prefix_route_protocol_v2.py:316-376`), then expects
`EXPLAINED_MISMATCH` (`:379-390`). An independent synthetic probe reduced that
source to two bytes, retained arbitrary 211/213 lists and permitted reason
codes, and still obtained `EXPLAINED_MISMATCH`.

This is a false scientific state: the exact source-derived 211/213 resolution
has not been established. Because R0 consumes the resulting canonical IDs,
the reporting estimand can be changed by the submitter.

Required correction:

1. Freeze a machine-readable authoritative source format and release/revision.
2. Parse its exact bytes and derive both ID lists and all difference reasons.
3. Resolve aliases through explicit source fields rather than reason labels.
4. Verify every canonical ID against the reporting annotation and required
   reporting subset.
5. Reject opaque or merely nonempty provenance files.

### P0: R1 can certify fabricated cache causality

The validator requires `repository_commit` only to match Git-SHA syntax and
requires `hf_snapshot_revision` only to be nonempty
(`opentad/utils/prefix_route_r1_v2.py:520-525`). It hashes each raw-video file
and checks the filename stem, but never decodes the video
(`:615-625`). Feature arrays are loaded for shape and dtype (`:626-664`), but
the baseline dynamic token is never compared with the corresponding row in
the committed cache.

Dynamic input hashes and token bytes come from the submitted JSON. Validation
checks digest syntax, input-hash inequality, invariant token equality, and a
nonzero support-token difference (`:362-476`); it never reconstructs the
selected RGB frames, perturbations, or extractor output. The final linkage
booleans and `PASS_R1_EXISTING_CACHE_CERTIFIED` are then emitted
(`:686-721`).

The positive test fixture makes the problem explicit:

- `.mp4` files contain ASCII text (`tests/test_prefix_route_protocol_v2.py:897-901`);
- the five input hashes are repeated characters (`:946-950`);
- invariant and support-sensitive token vectors are caller-selected constants
  (`:933-955`);
- repository and model revision identities are arbitrary (`:983-990`);
- the validator is expected to PASS (`:1049-1061`).

An independent probe replaced every raw video with one byte, used a 40-zero
repository commit and arbitrary revision, and retained fabricated hash/token
records. It returned `PASS_R1_EXISTING_CACHE_CERTIFIED`.

The terminal-token rule no longer permits a literal empty/no-op future hash:
making future and baseline input hashes equal is rejected at
`opentad/utils/prefix_route_r1_v2.py:451-464`. However, an arbitrary unequal
hash plus caller-asserted invariant tokens still passes. Thus the empty-future
syntax defect is closed, but the scientific terminal intervention remains
unverified.

Required correction:

1. Bind the extraction commit to an exact reviewed or separately frozen
   commit, not any syntactically valid SHA.
2. Decode every raw video and derive frame count, selected frames, decision
   supports, and perturbation tensors.
3. Execute the exact bound extractor/config/weights in a controlled process.
4. Recompute every input hash and output token.
5. Compare every baseline token byte-for-byte with its identified cache row.
6. Treat submitted dynamic JSON only as a transcript to compare against
   recomputation.

### P1: R0 accepts the wrong subset and understates exposure

`_parse_video` requires `subset` only to be nonempty
(`opentad/utils/prefix_route_r0_v2.py:181-190`). The collector checks that each
reporting ID exists in the annotation database but does not require a canonical
reporting/validation subset (`:548-586`). A synthetic video marked `training`
was accepted as `PASS_R0_COMPLETE`.

The protocol validator freezes reporting exposure as
`DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK`
(`opentad/utils/prefix_route_protocol_v2.py:415-423`), while the R0 report emits
`DESIGN_EXPOSED_ROUTE_SELECTION_ONLY`
(`opentad/utils/prefix_route_r0_v2.py:765-780`). The emitted evidence therefore
weakens the declared exposure status.

The interval implementation itself is substantially improved:

- half-open intervals and end-before-start ties are explicit
  (`opentad/utils/prefix_route_r0_v2.py:150-178`);
- malformed, unknown, Ambiguous, and exact duplicate handling is explicit
  (`:181-304`);
- overlap and same-bin unordered pair counting is explicit (`:308-402`);
- the bootstrap unit is video and one shared resample is used for targets
  (`:499-545`);
- the R0 envelope is recomputed from source bytes and compared exactly
  (`opentad/utils/prefix_route_protocol_v2.py:1732-1850`).

The power gate remains scientifically underdefined. Code uses an independent
two-proportion normal approximation with a heuristic ICC design effect
(`opentad/utils/prefix_route_r0_v2.py:425-469`). The later route comparison is
paired by seed/video and often by the same ground-truth instances. The protocol
does not justify the independent-group covariance assumption or establish that
this power calculation targets the downstream estimand.

Required correction:

1. Derive and enforce the exact reporting subset for every canonical ID.
2. Emit the exact protocol exposure value from the validated protocol record.
3. Add training/mixed-subset and exact-exposure regression tests.
4. Define and justify the power estimand, allocation, covariance, and cluster
   model, or explicitly limit it to descriptive R0 eligibility.

### P1: B2 is not uniquely executable and fairness is caller-asserted

The human contract describes B2 as newborn object queries plus propagated track
queries (`PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md:153-168`). No manifest-bound
source fixes a concrete B2 architecture, newborn/track query counts,
initialization, temporal assignment, termination, loss, or capacity allocation.
Consequently, multiple materially different temporal-MOTR baselines satisfy
the prose.

`derive_fairness_audit` accepts caller-provided resource scalars and parameter
inventories, including booleans that claim each parameter was used in forward
and observed a gradient
(`opentad/utils/prefix_route_fairness_v2.py:45-143`). It then derives PASS from
those assertions (`:209-222`). The positive fixture supplies identical made-up
rows and `True` booleans (`tests/test_prefix_route_protocol_v2.py:641-680`).
The same asserted payload produced
`PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED` in an independent probe without any
model, graph, trace, counter, or measurement source.

This does not establish B0-B4 fairness, capacity matching, resource matching,
or a faithful B2 attack.

Required correction:

1. Bind executable B0-B4 contracts, including one exact B2 reconstruction.
2. Derive inventories from actual model objects and forward/autograd hooks.
3. Derive MACs, state bytes, memory, latency, update counts, token counts,
   calibration count, trials, and seeds from hash-bound measurement records.
4. Reject caller booleans and unproven scalar rows as fairness evidence.

### P1: Controls improve, but R5 is internally inconsistent

The semantic derangement implementation now has split scope, multiset
preservation, no fixed labels, no global consistent rename, deterministic
serialization inputs, and fail-closed unconstructibility
(`opentad/utils/prefix_route_controls_v2.py:122-249`). The eight relocked
semantic weakening checks also reject their mutations. This part is materially
improved.

R5 remains unusable:

1. The public iterator cycles finite factor cells for any requested count
   (`opentad/utils/prefix_route_ood_v2.py:451-481`).
2. Scientific identity excludes seed/index/set metadata but includes declared
   factor labels (`:365-381`).
3. `audit_sequence_sets` rejects any repeated scientific hash, including a
   repetition within one set, but does not enforce required set names, sizes,
   shifted support, or max-minus-min balance (`:484-524`).
4. The protocol-required `TRAIN=1000` generated by the public iterator fails
   its own audit as `sequence sets overlap: TRAIN and TRAIN`.
5. A one-sequence `TRAIN` set is accepted as pairwise disjoint.
6. With empty event topology, identity and held-out semantic mappings produce
   identical events and observations but distinct hashes because
   `factor_spec` differs. A no-effect shift can evade overlap/no-op detection.
7. `same_bin_handoff` places event 0 end and event 1 start at the same
   half-open boundary (`:158-163`). Event 0 occupies through bin 31 and event 1
   starts at bin 32, so this is adjacent-bin touching, not an R0
   same-decision-bin end/start case.
8. The public iterator refuses compound OOD (`:460-465`), and no bound
   executable schema validates the required compound table.

Required correction:

1. Define scientific replicate identity and generate the exact frozen counts
   without duplicate scientific samples.
2. Validate exact set names, sizes, supports, family shifts, and balance.
3. Reject any shifted factor that leaves generated scientific content
   unchanged.
4. Implement same-bin handoff using sub-bin event time mapped to one decision
   bin.
5. Bind and validate the compound-OOD table and commitment before model work.

### P1: R6 pairing improves, but inference and terminal decisions are bypassable

`paired_crossed_bootstrap` correctly enforces a complete arm-seed-video
Cartesian product and common paired seed/video draws
(`opentad/evaluations/prefix_route_r6_v2.py:454-493`). It also includes all
registered contrasts times eligible metrics in the multiplicity denominator
(`:493-587`). These are real improvements over V1.

The production parameters remain caller-controlled:
`resamples`, `seed`, and `family_wise_alpha` are public overrides
(`:433-440`). A probe with one bootstrap resample and family-wise alpha 0.9 was
accepted and reported as the inference method. No sealed protocol entry point
enforces 10,000 resamples and alpha 0.05.

`terminal_route_decision` consumes only selected entries from an arbitrary
`inference["intervals"]` dictionary (`:604-672`). It does not verify:

- method, resamples, seed, alpha, or multiplicity;
- the complete registered contrast/metric schema;
- linkage of per-video cells to emissions/ground truth;
- linkage of mOnlineAP rows to the bound evaluator;
- source-derived stress membership;
- negative-control results.

A hand-built favorable interval dictionary returned
`PASS_B4_ROUTE_SURVIVES` with `route_claim_allowed=true`. The protocol
registers `KILL_BENCHMARK_NOT_IDENTIFIABLE`, but the terminal function has no
negative-control input and no path to that status, contrary to
`PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md:203-211`.

Required correction:

1. Expose one sealed production inference entry point with fixed parameters.
2. Bind emissions, ground truth, R0 stress membership, per-video cells, and
   dataset mOnlineAP rows in one recomputed evidence envelope.
3. Validate the complete inference schema before terminal decisions.
4. Require source-derived negative-control decisions and implement a reachable
   benchmark-identifiability kill.
5. Add tests proving arbitrary intervals and parameter weakening cannot PASS.

### P2: Source bytes reproduce, but tests preserve false-PASS paths

The fixed-source mechanism is reproducible:

- local `HEAD` and tree match the frozen commit/tree;
- Protocol V2 is canonical, 18,127 bytes, and matches its expected SHA-256;
- the manifest is canonical, 2,427 bytes, and matches its expected SHA-256;
- all 20 entries match the SHA-256 of exact Git blob bytes;
- checkout bytes equal commit bytes for every entry;
- every bound text file is LF, has no CR, and ends with LF;
- `.gitattributes:1-8` freezes LF for Python, Markdown, JSON, and JSONL;
- rebuilding the manifest outside the repository produced byte-identical
  output;
- both the author worktree and detached audit checkout remained clean.

The test suite is not scientifically sufficient. Its positive population, R1,
and fairness fixtures are the same caller-created payloads that enable false
PASS. It lacks tests for authoritative source parsing, raw-video decoding,
baseline-cache token identity, measurement provenance, exact R5 set contracts,
no-effect shifts, locked R6 parameters, and negative controls in the terminal
decision.

There is also a Windows execution defect. `_root` resolves an 8.3 path to its
long form while `_lexical_absolute` retains the short form, then `relative_to`
rejects the contained file
(`opentad/utils/evidence_bundle.py:63-98`). A default temporary path under
`C:\Users\SKYWAL~1\...` was rejected as escaping its resolved
`C:\Users\skywalker\...` root. This fails closed but makes normal temporary
evidence construction nonportable on the review host.

Required correction:

1. Add regression tests for every false accept and internal R5 failure above.
2. Make positive R1/fairness fixtures derive evidence through controlled
   instrumentation rather than asserting it.
3. Canonicalize root and candidate paths consistently while retaining
   symlink/junction rejection.
4. Test Windows 8.3 aliases and normal temporary directories.

## V1 closure matrix

| V1 finding | Status | Reason |
|---|---|---|
| P0-1 independent review self-forgeable | CLOSED | Source and tests show one authorization path that rereads fixed protocol/manifest/repository inputs, rejects caller dictionaries, and blocks unsigned collection. This academic-only continuation did not process identity/signature artifacts. |
| P0-2 population/R0/R1 statuses asserted | OPEN | R0 report recomputation is improved, but arbitrary population provenance and fabricated R1 evidence still receive scientific PASS states. |
| P1-1 protocol semantics shallow | CLOSED | Exact nested values and policy lock reject all eight relocked forbidden weakenings. |
| P1-2 R0 not uniquely executable | OPEN | Most boundary rules are executable, but subset membership is unconstrained and the power model is not aligned or justified for the paired downstream estimand. |
| P1-3 annotation exposure uncontrolled | OPEN | Protocol/ledger disclose exposure, but the canonical R0 report understates it. |
| P1-4 B0-B4 fairness and D1/D2 isolation | OPEN | B2 is not uniquely executable; fairness inputs are assertions; downstream isolation can be forged through R6 intervals. |
| P1-5 semantic controls/R5 nonconstructive | OPEN | Semantic derangement is improved, but R5 required sets, no-op detection, same-bin geometry, and compound OOD remain invalid/incomplete. |
| P1-6 R6 incomplete | OPEN | Pairing and field mapping improve, but inference parameters, evidence linkage, controls, and terminal validation remain bypassable. |
| P2-1 source binding/tests incomplete | OPEN | Commit/blob/LF/manifest reproducibility is closed, but adversarial coverage and Windows temporary-path execution remain open. |

## Direct answers

### Incorrect scientific states accepted

Yes:

- arbitrary 211/213 provenance obtains `EXPLAINED_MISMATCH`;
- fabricated R1 obtains `PASS_R1_EXISTING_CACHE_CERTIFIED`;
- training-subset R0 obtains `PASS_R0_COMPLETE`;
- asserted fairness obtains `PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED`;
- one unbalanced R5 sequence passes the set audit;
- arbitrary favorable R6 intervals obtain `PASS_B4_ROUTE_SURVIVES`;
- one-resample, alpha-0.9 R6 inference is accepted.

### Nonunique estimands or statistical errors

Yes:

- R0 can target the wrong subset;
- R0 power assumptions are not matched to the paired downstream estimand;
- B2 admits multiple materially different implementations;
- R5 factor labels can create apparent shifts without scientific-content
  changes;
- R6 production parameters are not locked and terminal input is not validated.

### Unfair baseline

Still possible. B2 is not uniquely instantiated, and fairness is derived from
caller-asserted rows rather than model/resource evidence.

### Exposure statement

Incorrect in the R0 artifact:
`DESIGN_EXPOSED_ROUTE_SELECTION_ONLY` is weaker than the frozen
`DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK`.

### Terminal-token future check

Literal empty/no-op input identity is rejected, so that narrow V1 defect is
closed. Scientific execution is still open because arbitrary unequal hashes
and token bytes can be submitted without recomputation.

### Protocol executability

Not complete. The required R5 TRAIN set fails its own audit, compound OOD lacks
an executable path, R6 terminal control logic is absent, and normal Windows
temporary evidence paths can fail.

### Source and commit reproducibility

The frozen commit, tree, protocol, manifest, 20 Git blobs, checkout bytes, LF
policy, and rebuilt manifest are reproducible. No source-substitution or
line-ending mismatch was found. This does not cure the scientific validator
defects.

## Tests and commands

All Python runs set:

```text
CUDA_VISIBLE_DEVICES=-1
NVIDIA_VISIBLE_DEVICES=none
ROCR_VISIBLE_DEVICES=-1
PYTHONDONTWRITEBYTECODE=1
```

Commands:

```text
python tools/validate_prefix_route_protocol_v2.py validate-protocol
python tools/validate_prefix_route_protocol_v2.py authorize-collection
python -m pytest tests/test_prefix_route_protocol_v2.py -q -rs
python -m pytest <10 focused protocol/cache/evaluator test files> -q -rs
python tools/validate_prefix_route_protocol_v2.py build-manifest --output <outside-repository-path>
git ls-files --eol <manifest and all entries>
git diff --check
git diff --cached --check
git status --short --untracked-files=all
python - <temporary scientific negative-contract probes>
python - <default-Windows-temp containment probe>
```

Results:

- Protocol validator:
  `PROTOCOL_V2_VALID_REVIEW_REQUIRED`, expected protocol and manifest hashes,
  20 entries, GPU hours 0.
- Unsigned authorization: exit code 2,
  `BLOCKED_PENDING_SIGNED_INDEPENDENT_PROTOCOL_REVIEW`.
- V2 tests: 27 passed, 0 failed, 0 skipped, 1 warning.
- Focused regressions: 101 passed, 0 failed, 6 skipped, 1 warning.
- Six skips were Torch import failures in isolated subprocesses.
- Manifest rebuild: byte-identical, 2,427 bytes, exact expected SHA-256.
- Git/EOL checks: clean; all 20 bound files LF/no-CR/final-LF.
- Scientific probes: the false accepts and internal failures listed above were
  reproduced.
- GPU use: none.

## Remaining risk

- Real R0/R1 sources were intentionally not accessed, so their actual format,
  membership, raw-video decodability, cache geometry, and token behavior remain
  unknown.
- Six Torch-dependent subprocess paths were skipped in this environment.
- No Linux or alternate-filesystem run was performed.
- Compound OOD remains unrevealed and also lacks a complete executable
  validation contract.
- Passing unit tests currently establishes code self-consistency, not valid
  scientific evidence provenance.

## Next step

Prepare a new fixed protocol commit that:

1. derives population and R1 evidence from authoritative/executed sources;
2. enforces R0 subset and exact exposure;
3. binds an executable B2 and measurement-derived fairness;
4. repairs and fully validates every R5 set, including compound OOD;
5. seals R6 parameters, evidence linkage, negative controls, and terminal
   schema;
6. adds adversarial tests for every reproduced false accept;
7. repairs Windows path canonicalization.

Then repeat the same outcome-blind, zero-GPU, fixed-commit review. No R0/R1
collection, model implementation, B0-B4 construction, training, profile,
checkpoint/model-output access, or GPU work is authorized now.

REVISE_PROTOCOL_BEFORE_COLLECTION
