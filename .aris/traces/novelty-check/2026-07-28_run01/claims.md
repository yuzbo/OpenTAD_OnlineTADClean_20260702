# EventMATR D1 novelty claims under review

Scope: standard closed-set, fully supervised, strict-causal Online Temporal
Action Localization (On-TAL).  This trace does not claim experimental
superiority.

## C1 — first-birth assignment and risk

Stable pre-birth temporal permutation-aware assignment coupled to
event-normalized interval-censored first-birth risk.

## C2 — duration-free end risk

Right-censored instance end hazard whose model boundary never consumes true
video duration, complete-video timing, or offline EOS.

## C3 — explicit action-instance ownership

Post-birth identity lock, owner-conditioned end, explicit cancellation and
reacquisition, including simultaneous same-class action instances.

## C4 — event-level exposure-bias closure

One chronological ragged event unroll shared by training and inference, with
mixed oracle/predicted tracks so false births, cancellations, and rebirths
become actual downstream training states.

## C5 — causal ledger

Strict causal input boundary plus immutable event ledger with positive length,
no duplicate emission, and no silent capacity eviction.  This is evaluated as
a correctness/reproducibility contribution, not a standalone learning
algorithm.

## Composite claim

The defensible candidate is C1+C2+C3+C4 as a unified event-process
reformulation of MATR for strict-causal On-TAL.  It does not claim invention of
set matching, survival hazard, identity-preserving queries, or scheduled
sampling in isolation.
