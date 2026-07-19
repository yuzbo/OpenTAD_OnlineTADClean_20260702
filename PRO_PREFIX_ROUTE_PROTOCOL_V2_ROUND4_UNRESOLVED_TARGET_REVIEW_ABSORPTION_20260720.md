# Prefix Route V2 Round-4 Unresolved-Target Review Absorption

Date: 2026-07-20

## Source Certificate

- Attachment:
  `C:\Users\skywalker\.codex\attachments\f5c2b241-e040-4375-8693-be594d48f0b5\pasted-text.txt`
- Byte-identical archive:
  `PRO_PREFIX_ROUTE_PROTOCOL_V2_ROUND4_UNRESOLVED_TARGET_REVIEW_20260720.md`
- Size: 49,740 bytes
- Text lines: 1,257
- SHA-256:
  `272B3650F6656FA160328F0D549C0C117BD788F7037A4B020A6FB50636B2B802`
- Repository:
  `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`
- Context branch: `codex/full-petal-implementation`
- Visible branch head and rejected predecessor:
  `48e78e43a77822ee0e51cace9ad758ca935c1450`
- Caller-supplied target literal: `TARGET_COMMIT`

The archive contains two concatenated reviews. The first records that the
formal target is unresolved, then performs an auxiliary source audit of the
visible branch head. The second applies a strict stop-chain and treats only
the immutable-target failure as formally admissible. Both return `REVISE`,
but their code-level evidence has different status and must not be merged into
one signed fixed-commit review.

## Independent Verification

The project independently confirmed:

- the public branch still points to rejected commit `48e78e43`;
- the request did not replace `TARGET_COMMIT` with a 40-hex object ID;
- `derive_per_video_cell` accepts inline emissions and writes
  `immutable=True`;
- formal R6 OnlineAP uses `require_ledger=False`;
- optimizer events contain digest strings but no referenced per-event input,
  model-before/after, or optimizer-before/after artifact bytes;
- `_runtime_budget` checks all event rows structurally but gives only the first
  event to the live measurement path;
- `validate_fairness_audit_record` creates adapters with
  `model=None, optimizer=None` and validates serialized rows;
- controls bind source/constructed hash strings without re-executing the
  frozen transformation;
- `LEDGER_ONLY` declares `learning=False`, while generic run provenance
  requires nonempty positive optimizer updates;
- B2 risk set, assignment, transition, and emission utilities are not bound by
  one replayable per-decision transcript;
- population inventory enforces ID uniqueness but not duplicate artifact
  identity or an independent official per-video manifest;
- source binding is a manually listed set rather than a rebuilt transitive
  import closure.

The dedicated local suite reports `46 passed, 1 warning`. This confirms that
existing tests encode the current contract; it does not close the scientific
false-PASS paths.

An independent in-process attack constructed a complete favorable inference
mapping without raw cells or bootstrap execution and obtained:

```text
status=PASS_B4_ROUTE_SURVIVES
route_claim_allowed=true
```

This directly confirms the terminal-helper finding on commit `48e78e43`.

## Agreement Decision

The outcome is accepted, but the attachment is not treated as one internally
uniform fixed-commit certificate.

| Review statement | Decision | Reason |
|---|---|---|
| Current verdict is `REVISE_PROTOCOL` | Accept | No valid successor target exists, and the visible rejected commit still has evidence-chain bypasses. |
| No collection, model work, profile, training, or GPU is authorized | Accept | This matches the frozen protocol boundary. |
| `TARGET_COMMIT` cannot be silently replaced by a moving branch head | Accept | A signed review must bind an author-declared immutable object. |
| P0/P1 code findings describe `48e78e43` | Accept as auxiliary diagnosis | They are independently reproduced and align with the exact-commit Gate-A review. |
| Those code findings belong to this unresolved-target attestation | Reject | The second block correctly says code-level closure is formally inadmissible after the target identity gate fails. |
| Every prescribed implementation mechanism is uniquely required | Qualify | The required property is source-bound, replayable, fail-closed evidence. Exact storage and replay mechanics may differ if they provide equivalent or stronger falsifiability. |

Therefore the answer to "fully agree" is **no at the level of every sentence
and evidence label**, but **yes on the terminal decision, all authorization
blocks, and the substantive defect classes**.

## Implementation Qualifications

1. A private helper being callable is not itself a failure. The failure is
   that a caller-created mapping can produce an authorizable route PASS.
   Terminal authority must require source-bound raw cells or a verifier-created
   capability.
2. Duplicate video bytes should fail by default, but legitimate aliases must
   be explicitly registered rather than made impossible by a universal
   uniqueness rule.
3. Import closure should combine deterministic static discovery, clean-process
   runtime import tracing, and explicit dynamic-registration declarations.
4. Optimizer evidence must bind real bytes and the complete event chain.
   Canonical artifacts, content-addressed deltas, or equivalent replayable
   storage are acceptable; self-authored digest strings are not.
5. Positive tests may use compact generated artifacts, but they must be real
   and replayable. Arbitrary hashes, fake videos, or hand-written intervals
   cannot be scientific PASS fixtures.

## Frozen Decision

1. Preserve the raw review without changing authorization.
2. Keep the exact-commit Gate-A decision in DR-054 as the active engineering
   repair authority.
3. Do not request another review before a real successor commit exists.
4. Repair formal entry, emission/optimizer/fairness/control provenance, B2
   transcript binding, terminal authority, population identity, import
   closure, and hostile tests at zero GPU.
5. Commit and push one immutable successor with an explicit 40-hex SHA.
6. Return that SHA, tree, expected hashes, and relation contract to the same
   independent reviewer.
7. Even a future PASS may initially authorize only
   `READ_ONLY_SOURCE_IDENTITY_REGISTRATION`.

ChronoTransport remains blocked behind Prefix Gate A.
