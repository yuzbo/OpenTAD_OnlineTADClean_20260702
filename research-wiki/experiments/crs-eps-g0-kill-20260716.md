---
type: experiment
node_id: exp:crs-eps-g0-kill-20260716
title: "CRS-EPS preregistered G0 replay-fidelity gate"
date: 2026-07-16
stage: completed
outcome: negative
implementation_commit: "70df86ea3d38d70c658ae0ee9e04245d57b834d4"
---

# CRS-EPS G0 Replay-Fidelity Gate

## Scope

This is a scientific falsification result for the current CRS-EPS
`dynamic_birth` training surrogate. It is not an On-TAD effectiveness result,
not a raw-video end-to-end result, and not evidence that the Full PETAL
infrastructure is ineffective.

## Immutable anchor

- repository: `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`
- branch: `codex/full-petal-implementation`
- implementation commit: `70df86ea3d38d70c658ae0ee9e04245d57b834d4`
- target platform: N16R4 Linux, CPU-only G0 audit
- config: `configs/causaltad/thumos_pes_q2_crs_eps_fixed.py`
- config SHA-256: `5348152130a9a6603a707f1d4e7a938da6331e06a96fc24bcabb75f4cb1b971d`

## Pre-outcome evidence chain

- local and target-Linux B0: `588/588` tests passed;
- same locked reviewer: `PASS / PROFILE=ALLOW / FORMAL=BLOCK / NEXT_GATE=G0`;
- selection artifact SHA-256:
  `457838e8c3cb6aa05b5d80a9f9240d872a8a5e8d2b38058d4a37bf1b797fea4f`;
- margin artifact SHA-256:
  `5160adab25fb6538990e91c611f094eb59f3b01b1f268158e7e81fb913004329`;
- exact checkpoint SHA-256:
  `68169922bc34b77ee9ac210b4bc90137276972e9b492745b5927c69652ff8e5f`;
- checkpoint generation: deterministic model initialization, seed `705`,
  `1,126,424` parameters, explicitly not model-quality evidence;
- persisted manifest identity:
  `96278dcef7835ceb31e96e499a20b0806f2d882167cd1fbb039534ee9a95b0e5`.

The raw evidence bundle remains outside the repository, as required by
`RTK.md`. Its signed terminal `audit.json` SHA-256 is
`a08183a8dba4ec5267b5a75eed3501882180a22f28d6b76ce2c7b4e65511009b`.
The repository records hashes and conclusions, not checkpoints, feature dumps,
or generated evidence archives.

## Frozen margins

| Metric | Frozen requirement |
|---|---:|
| minimum gradient cosine | `0.90` |
| minimum gradient sign agreement | `0.90` |
| maximum relative loss error | `0.10` |
| minimum continuous runtime-state cosine | `0.90` |
| discrete runtime state | exact equality |
| maximum mean dynamic replay ratio | `0.80` |
| maximum video-start fallback fraction | `0.25` |
| minimum dynamic minus fixed gradient cosine | `0.02` |
| minimum dynamic minus reset gradient cosine | `0.05` |

These margins, samples, checkpoint bytes, and method were frozen before the
terminal audit. They must not be weakened in response to this outcome.

## Terminal result

The signed gate returned `KILL` over four selected samples and twelve candidate
trace rows, with fourteen violations across three samples.

| Aggregate metric | Result |
|---|---:|
| maximum dynamic relative loss error | `0.8021221151` |
| minimum dynamic gradient cosine | `0.2830554455` |
| minimum dynamic gradient sign agreement | `0.5445116162` |
| minimum dynamic continuous-state cosine | `0.1027812195` |
| mean dynamic replay ratio | `0.7586404310` |
| video-start fallback fraction | `0.25` |
| mean dynamic minus fixed gradient cosine | `0.1141427186` |
| mean dynamic minus reset gradient cosine | `0.5005255434` |

The video-start fallback sample matched the full control. The other three
samples failed absolute fidelity. In two of those samples, `dynamic_birth` and
`fixed_192` produced exactly the same reported fidelity metrics. One sample
showed `dynamic_birth` improving over `fixed_192`, but it still failed every
relevant absolute fidelity margin.

## Decision

1. Do not launch fixed-step profile or formal training for this route.
2. Treat the present CRS-EPS `dynamic_birth` surrogate as scientifically
   rejected under its frozen contract.
3. Do not interpret HH/IPW probability correction as proof that omitted state
   or gradient paths are unbiased.
4. Use the exposed four G0 samples only for development diagnosis from now on.
5. Any replacement method requires a new commit, a new outcome-blind holdout
   G0 preregistration, B0, independent review, and a new terminal G0.

## Open causal question

The current evidence does not by itself distinguish a structural incompatibility
of event-centric detached replay from a localized implementation defect. The
next Pro review must derive the estimand, inspect the exact replay/state/gradient
path, and decide between `STRUCTURAL_KILL`, `IMPLEMENTATION_FIX_CANDIDATE`, and
`INSUFFICIENT_EVIDENCE` before any new training route is implemented.
