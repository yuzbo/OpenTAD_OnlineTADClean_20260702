# Independent Review of Prefix-Route Evidence Protocol V1

## Review Identity

- Reviewer: Codex independent sub-agent
- Reviewer task/agent ID: `019f6f63-496d-75a0-a77b-91425a8e7ea1`
- Reviewed at: `2026-07-17T09:37:40.410Z`
- Frozen commit: `5e858a161254395488428609b441b64f3c08be76`
- Parent commit: `ef4ba13c8d4fa05296b1cf3fa9259bab6fe1058c`
- Protocol SHA-256: `e49e32e21e5da10342694637fbba7acdd7fdb7dac5cd6f1c248355a24fd018dc`
- Review mode: sole independent, outcome-blind, read-only, zero GPU

## Verdict

`REVISE_PROTOCOL_BEFORE_COLLECTION`

This verdict blocks R0/R1 collection, profile, formal training, and scientific
claims. It does not reject the research route; it rejects Protocol V1 as an
executable evidence contract.

## Findings

### P0-1: Independent review can be self-forged

The certificate contract is only declarative
(`configs/causaltad/protocols/prefix_route_identifiability_v1.json:57`).
Validation merely compares `reviewer_id` with caller-supplied `author_id`,
checks hash syntax, and optionally hashes a supplied artifact
(`opentad/utils/prefix_route_protocol.py:685`,
`tools/validate_prefix_route_protocol.py:111`). There is no trusted identity,
signature, artifact-content/verdict parsing, or exact frozen-tree binding.
Worse, `collection_authorization()` trusts an unvalidated dictionary
(`opentad/utils/prefix_route_protocol.py:1038`); a fabricated
`{"certificate":{"verdict":"PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION"}}`
authorized collection in the reviewer probe.

Required fix:

- require platform-attested reviewer identity or a verifiable signature;
- bind the signed artifact to reviewer, task, verdict, frozen commit, tree, and
  manifest;
- make every authorization path reload and validate the certificate and review
  artifact.

### P0-2: Population, R0, and R1 PASS certificates need not contain verifiable evidence

Population validation checks declared counts and caller-provided set
differences but never reads or hashes the historical/canonical ID sources
(`prefix_route_identifiability_v1.json:90`,
`prefix_route_protocol.py:769`). Thus the claimed source-derived 211/213
reconciliation is not independently established; arbitrary IDs, paths, and
hashes passed.

R0 accepted `PASS_R0_COMPLETE` with empty source hashes, aggregates,
eligibility, and environment (`prefix_route_protocol.py:851`). R1 accepted
boolean-only audits, repeated dummy hashes, and disjoint noncanonical
raw/feature key sets (`prefix_route_identifiability_v1.json:306`,
`prefix_route_protocol.py:892`).

Required fix:

- require contained evidence files with exact paths and SHA-256 values;
- re-read the evidence files;
- derive ID sets, R0 aggregates, R1 support rows, perturbation outcomes, and
  cache-key equality;
- compute status instead of trusting a submitted status.

### P1-1: Protocol validation is structurally, not semantically, fail-closed

The main checks enforce top-level shape (`prefix_route_protocol.py:635`);
individual controls receive sparse checks (`prefix_route_protocol.py:548`).
The reviewer applied in-memory mutations that:

- added `FORMAL_TRAINING`;
- allowed future ground truth;
- allowed early stopping;
- deleted R0 aggregates;
- deleted R5 levels;
- deleted B4 survival;
- made the R1 mutation a no-op;
- made semantic derangement the identity.

All eight weakened protocols passed `validate_protocol`.

Required fix:

- use a strict nested schema plus semantic invariants for every authorization,
  gate, factor, control, and terminal action;
- add one mutation test per forbidden relaxation.

### P1-2: R0 is not uniquely executable and has no bound collector

The first-bin `d_(j-1)` is undefined while the schedule requires a supplied
previous frame (`prefix_route_identifiability_v1.json:127`,
`opentad/utils/prefix_instance_schedule.py:47`). Sorting and ties, clipping,
invalid intervals, and duplicate annotations are unspecified. The same-bin
pair rule conflates handoff, overlap, order, and possible double counting.
Positive-video and bootstrap denominators, confidence-interval algorithm, zero
cases, and the cluster-size vector are not fixed
(`prefix_route_identifiability_v1.json:146`).

The bound analyzer only reports legacy duration, concurrency, and count
summaries (`tools/analyze_ontad_instances.py:48`).

Required fix:

- freeze legal-interval preprocessing, first-bin and tie rules;
- distinguish overlap and handoff estimands;
- freeze exact denominators and percentile bootstrap;
- bind a collector with synthetic edge-case tests.

### P1-3: Reporting-label leakage is not controlled as an untouched scientific test

Reporting annotations drive route eligibility and failure actions, with
author-visible aggregates before any model contract is frozen
(`prefix_route_identifiability_v1.json:178`,
`research-wiki/query_pack.md:56`). This is outcome-blind with respect to model
predictions, but not annotation-unseen. The prior-exposure ledger lacks a
concrete append-only path, hash, and signing procedure.

Required fix:

- derive eligibility on development data and reserve an untouched reporting
  set, or formally designate the reporting set as design-exposed and provision
  a new prospective test set;
- freeze a signed append-only exposure ledger.

### P1-4: B0-B4 fairness and D1/D2 isolation are insufficient

Matching covers training MACs and parameters but not inference MACs,
live-state bytes, peak memory, or latency
(`prefix_route_identifiability_v1.json:465`). B2 is named a temporal-MOTR
reconstruction but receives a special pre-decision free-capacity restriction
(`prefix_route_identifiability_v1.json:424` and `:813`) that is not shown
equivalent to newborn/object plus propagated-query tracking designs.

B4 survival need not establish A4/D1 or joint A1+A2/D2 necessity
(`prefix_route_identifiability_v1.json:616` and `:884`).

Required fix:

- include a faithful canonical comparator;
- include full training, inference, state, memory, and latency resource views;
- require the corresponding D1 deletion or joint D2 factorial deletion for
  route survival.

### P1-5: Semantic controls and R5 are not constructively specified

Derangement omits split scope, byte serialization, feasibility/fallback for
imbalanced labels, and the promised class-AP margin
(`prefix_route_identifiability_v1.json:576`). R5 has factor names but no
sequence size, event placement, valid-cross constraints, probabilities,
observation equation, generator algorithm, or per-family seeds. The generator
is only a future commitment (`prefix_route_identifiability_v1.json:662`).

Required fix:

- bind executable generator and control code now;
- freeze exact distributions, feasibility rules, canonical serialization,
  seeds, expected cell counts, metrics, and terminal failure actions.

### P1-6: R6 metrics, inference, and terminal decisions are incomplete

The formal output uses `decision_frame` and `commit_frame`
(`prefix_route_identifiability_v1.json:13`), while bound evaluators require
`source_frame` and `emit_frame` (`opentad/evaluations/full_petal_metrics.py:280`).
Required per-video false-emission, same-bin, and direct-complete estimands are
absent or differ from implemented ratios
(`prefix_route_identifiability_v1.json:848`,
`opentad/evaluations/full_petal_metrics.py:885`,
`opentad/evaluations/online_budgeted_map.py:585`).

Sampling a seed independently inside each resampled video breaks model-seed
clustering; multiplicity omits contrasts; an inconclusive B4 confidence
interval has no terminal action (`prefix_route_identifiability_v1.json:872`).

Required fix:

- bind exact metric implementations and field mapping;
- preserve paired seed runs across videos or use valid crossed-cluster
  inference;
- adjust across all metrics and contrasts;
- define an indeterminate terminal result.

### P2-1: Commit/source binding and tests leave meaningful gaps

`prefix_route_protocol.py` imports validation primitives from unbound
`evidence_bundle.py` (`prefix_route_protocol.py:12`,
`prefix_route_identifiability_v1.json:927`). Git binding verifies the protocol
blob at a named commit, not the exact frozen commit/tree
(`prefix_route_protocol.py:1001`). Tests primarily use dummy hashes and
asserted statuses, so they validate bypasses rather than reject them
(`tests/test_prefix_route_protocol.py:39`).

This contradicts the fail-closed claim in `research-wiki/source_map.md:232`.
DR-047 and T40 correctly retain the collection block; DR-048 and T41 overstate
executability.

Required fix:

- bind every validator dependency and the exact tree manifest;
- require `protocol_commit` to equal the frozen commit;
- replace declaration fixtures with real temporary evidence artifacts and
  adversarial rejection tests.

## Commands and Verification

- `validate-protocol`: exit `0`,
  `PROTOCOL_VALID_REVIEW_REQUIRED`.
- `authorize-collection` without certificate: exit `2`,
  `BLOCKED_PENDING_INDEPENDENT_PROTOCOL_REVIEW`.
- `pytest tests/test_prefix_route_protocol.py -q`: `16 passed`.
- Focused nine-file zero-GPU suite: `96 passed, 4 skipped`.
- `pytest tests/test_p0_review_hardening.py -q -rs`:
  `5 passed, 2 skipped`.
- Six skips were Torch-runtime tests; `import torch` failed loading `c10.dll`
  with Windows error 1114.
- No GPU was used.
- Adversarial probes were in-memory only.
- All eight weakened-protocol mutations and all five forged-evidence cases
  described above were accepted.
- The worktree remained clean.
- No model outputs, checkpoints, R0/R1 statistics, or generated evidence were
  accessed.

## Verified Bindings

- Local HEAD:
  `5e858a161254395488428609b441b64f3c08be76`
- Remote branch:
  `5e858a161254395488428609b441b64f3c08be76`
- Protocol JSON SHA-256:
  `e49e32e21e5da10342694637fbba7acdd7fdb7dac5cd6f1c248355a24fd018dc`
- Protocol Markdown SHA-256:
  `e9d867ba9eb7953f9e0b0a7ef8a07f07f6b4972f87be1cff5060c360a78cc4b9`
- External attachment and archived review SHA-256:
  `f9064827b05991fd63156770b2a1d6d6d40c3e53a73f4113b31f312b17f82bd0`
- Review absorption SHA-256:
  `73f6c034c05f9fbab37fd8650ecd0152b52fafec55487842ab93e4c4d532d54b`
- All ten V1 `source_bindings.code_sources` hashes matched frozen-commit bytes.

## Machine-Readable Verdict

REVISE_PROTOCOL_BEFORE_COLLECTION
