---
type: experiment
node_id: exp:ontad-rgb-dynamic-event-memory-design-20260722
title: "Raw-RGB Dynamic Event Memory Design Freeze"
status: design-written-awaiting-user-review
outcome: pending
updated: 2026-07-22
---

# Raw-RGB Dynamic Event Memory Design Freeze — 2026-07-22

## Purpose

Record the transition from the technically valid but low-performing fixed-slot
feature baseline to a raw-RGB, slot-free dynamic-event method built from faithful
official On-TAD and event-memory donors.

## Reproducible Starting Point

- branch: `codex/ontad-rgb-event-memory`;
- clean design workspace:
  `E:/DeskTop/TAD/OpenTAD_OnlineTADClean_20260702/_codex_worktrees/ontad-rgb-event-memory-clean`;
- starting commit: `36081a5` from `codex/ontad-science-fixed-rematch`;
- full specification:
  `docs/superpowers/specs/2026-07-22-raw-rgb-dynamic-event-memory-ontal-design.md`.

## Frozen Design Decisions

1. The mandatory final input is original RGB, not cached features.
2. Starts are detected immediately from causal evidence and create dynamic event
   records; class beliefs may refine while the event remains provisional/active.
3. Ends are conditioned on the owning event trajectory, preventing unconstrained
   start/end pairing.
4. Active events have no manually chosen semantic slot count. Learned
   hierarchical visual memory adapts its effective span to sample content and
   action duration.
5. A/B/C/D official repositories remain complete and read-only. Writable donor
   replicas and all A+B family fusion trees are isolated from them.
6. Every port records exact upstream SHA, source path, license, diff hash, and
   parity evidence. Official data, target, loss, assignment, decoder, metric,
   optimizer, and schedule paths must be reviewed before changing a donor.
   Guarded official runs write all outputs outside the source tree and fail if
   pre/post HEAD, tree hash, status, or file inventory changes.
7. Thresholds are calibration policies rather than a universal 0.5 rule.
8. Structured On-TAD output is mandatory; OnVLLM language output is optional.

## Planned Parallel Lanes

- acquire and fidelity-test A/B/C/D read-only references;
- build writable single-donor adapters and the fusion matrix;
- replace the raw-RGB visual stub and verify causal gradients/latency;
- run overlap, association, long-action, future-perturbation, and memory tests;
- launch calibration and multi-seed/report jobs through scheduler dependencies
  after their registered parents pass.

## Dependency-Aware Parallel Experiment Graph

```mermaid
flowchart LR
  subgraph R["Read-only official references"]
    A0["A ActionSwitch"]
    B0["B MATR"]
    C0["C HAT/OAT"]
    D0["D HEM 2025"]
  end

  subgraph F["Fidelity and writable replicas"]
    A1["A parity receipt"]
    B1["B parity receipt"]
    C1["C parity receipt"]
    D1["D parity receipt"]
  end

  subgraph M["Matched feature fusion array"]
    AB["AB"]
    AD["AD"]
    BD["BD"]
    BC["BC"]
    ABD["ABD primary mechanism"]
    ABCD["ABCD optional"]
  end

  subgraph V["Raw-RGB lane"]
    RGB0["Real causal visual encoder"]
    RGB1["RGB prefix/gradient/latency smoke"]
    RGB2["Raw-RGB adapter training"]
    RGB3["Raw-RGB ABD joint training"]
  end

  subgraph Q["Independent contract tests"]
    Q1["Overlap and association"]
    Q2["Future perturbation"]
    Q3["Adaptive memory and long actions"]
    Q4["Ledger and immutable output"]
  end

  subgraph E["Automatic evidence chain"]
    CAL["Calibration-only selection"]
    S1["Single-seed scientific gate"]
    MS["Multi-seed confirmation"]
    REP["Report-once main results"]
  end

  A0 --> A1 --> AB
  A1 --> AD
  B0 --> B1 --> AB
  B1 --> BD
  B1 --> BC
  C0 --> C1 --> BC
  D0 --> D1 --> AD
  D1 --> BD
  AB --> ABD
  AD --> ABD
  BD --> ABD
  ABD --> ABCD
  BC --> ABCD
  RGB0 --> RGB1 --> RGB2
  ABD --> RGB3
  RGB2 --> RGB3
  Q1 --> S1
  Q2 --> S1
  Q3 --> S1
  Q4 --> S1
  RGB3 --> CAL --> S1 --> MS --> REP
```

The graph permits A/B/C/D acquisition, parity work, raw-RGB infrastructure, and
contract tests to progress concurrently. Only expensive joint raw-RGB ABD
training waits for both a working RGB lane and the feature-level ABD mechanism
gate. ABCD is optional and cannot delay the primary ABD path.

No official repository has been modified, and no fusion implementation or new
training run is claimed at this design checkpoint.
