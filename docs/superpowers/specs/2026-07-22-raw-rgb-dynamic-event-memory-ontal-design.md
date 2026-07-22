# Raw-RGB Dynamic Event Memory On-TAD / OnVLLM Design

Status: written for user review
Date: 2026-07-22
Branch: `codex/ontad-rgb-event-memory`
Starting point: `36081a5` from `codex/ontad-science-fixed-rematch`

## 1. Outcome and scope

The mandatory final deliverable is a standard, fully supervised, strictly causal
Online Temporal Action Localization model whose runtime input is an original RGB
video stream. At time `t`, it may consume only frames at or before `t`. It must:

1. detect an action start as soon as causal visual evidence makes the transition
   observable;
2. publish a provisional start time and provisional class distribution;
3. maintain each ongoing action as an independent dynamic event;
4. update its class, boundary belief, and memory using only newly arrived evidence;
5. close the same event when its end is detected; and
6. emit one immutable, positive-length final interval with bounded video-time and
   wall-clock delay.

Feature-level experiments are an architecture-screening and attribution stage,
not the final model and not the paper's final claim. A result that works only on
cached SigLIP2, TSN, or SlowFast features does not satisfy the project objective.

The primary output is closed-set On-TAD. A secondary OnVLLM head may consume the
same event states and produce open-vocabulary event descriptions, but it must not
replace or weaken the structured interval output. The shared causal event engine
is therefore designed once and exposed through two output adapters:

- mandatory `StructuredOnTADHead`: class, start, end, confidence, event ID;
- optional `StreamingLanguageEventHead`: text conditioned on the same event ID
  and causal event state.

## 2. Explicit non-goals

- No fixed semantic slot count, fixed switch count, or manually chosen maximum
  number of simultaneous action instances.
- No free end-to-any-past-start matching as the primary association mechanism.
- No universal `0.5` start/end decision rule. An all-0.5 configuration is an
  ablation only.
- No report-split threshold or checkpoint tuning.
- No whole upstream repository copied into this repository.
- No claim that cached-feature success is raw-RGB end-to-end success.
- No change to the already frozen interpretation of the old FIXED/REMATCH runs;
  they remain a technically valid but operationally failed slot-based baseline.

## 3. Upstream methods and provenance

Four official implementations are mandatory donors and baselines:

| Symbol | Method | Official repository | Role |
|---|---|---|---|
| A | ActionSwitch | <https://github.com/musicalOffering/ActionSwitch-release> | immediate state-transition detection, overlap protocol, conservativeness loss, Hungarian F1 |
| B | MATR | <https://github.com/skhcjh231/MATR_codebase> | strong class/boundary decoder, selective past memory, current-end/past-start baseline |
| C | HAT/OAT | <https://github.com/sakibreza/ECCV24-HAT> | short/long context fusion, anchor baseline, OSN/online suppression baseline |
| D | Hierarchical Event Memory | <https://github.com/minghangz/OnVTG> | multiscale event construction, redundant-event merging, dynamic per-scale memory allocation, latency analysis |

OnPoint is a non-equivalent point-supervised task, but its actionness-calibrated
attention and anticipation losses are registered as optional module donors after
the four mandatory routes are stable.

The repositories are cloned concurrently into an external reference directory,
never inside the clean route repository. They are maintained as complete,
detached, read-only reference baselines rather than edited vendor trees. Each
baseline is checked before and after every use for its exact commit SHA, clean
working tree, and Git tree hash. No experiment is allowed to create a commit,
apply a patch, change tracked configuration, or push from a reference baseline.

The external layout has three deliberately separate layers:

```text
ONTAD_EXTERNAL_ROOT/
  official-readonly/          # complete untouched A/B/C/D repositories
    actionswitch/<sha>/
    matr/<sha>/
    hat/<sha>/
    onvtg/<sha>/
  method-workspaces/          # writable replicas for compatibility experiments
    A/<experiment-id>/
    B/<experiment-id>/
    C/<experiment-id>/
    D/<experiment-id>/
  fusion-workspaces/          # writable A+B, A+D, B+D, ... integration trees
    AB/<experiment-id>/
    AD/<experiment-id>/
    BD/<experiment-id>/
    BC/<experiment-id>/
    ABD/<experiment-id>/
    ABCD/<experiment-id>/
```

The baseline launcher executes an allow-listed read-only command from each
official tree while placing checkpoints, logs, caches, compiled extensions, and
temporary files in an external run directory. Where the platform supports it,
the official tree is mounted or permissioned read-only; independently of that
mechanism, `reference_guard.json` records the pre/post HEAD, tree hash, tracked
status, untracked status, and file inventory. Any change fails the job. This
prevents an otherwise harmless evaluation script from silently writing into or
mutating the gold reference.

Writable replicas may be cloned or copied from a frozen official baseline, but
they never replace it. An A+B experiment is implemented only in its own `AB`
workspace and later ported into this clean route through explicit adapters. A
failed fusion can therefore be deleted or rebuilt without changing either A or
B, and the untouched official implementations remain available for line-by-line
comparison and fidelity evaluation.

Acquisition creates `upstream_manifest.json` containing repository URL, exact
commit SHA, acquisition date, license path and hash, files consulted, and hashes
of every source file that is later ported. Every writable workspace adds a
`derivation_manifest.json` that maps each changed or ported file to its official
source file and commit, states why the change is necessary, and records the
resulting diff hash. The manifest also distinguishes exact reuse, interface-only
adaptation, scientific modification, and newly written glue code.

Before modifying a donor module, the implementation review must cover the whole
official path that determines its behavior: data preprocessing and coordinates,
model construction, state/memory updates, target generation, assignment,
losses, decoding/post-processing, evaluation, optimizer, and schedule. A paper-
only or isolated-file reconstruction is not accepted when official code exists.
A module may enter the route only through a small compatibility adapter and this
traceability record. This preserves scientific attribution while avoiding four
incompatible training stacks inside OpenTAD.

At each fusion milestone, the untouched parents are rerun through their guarded
smoke/evaluation entry points. This detects environment or shared-dependency
changes that could make A+B appear better merely because A or B was accidentally
degraded. Parent results, adapter-parity results, and fusion results are stored as
three distinct receipts.

Before fusion, each untouched official baseline must pass a fidelity check using
its published feature type, checkpoint when available, decoding rule, metric,
and training/evaluation entry point. The corresponding writable replica must
then pass an adapter-parity test against that untouched baseline on frozen input:
tensor shapes, deterministic intermediate states, decoded intervals, and metrics
are compared at registered checkpoints. The reproduction tolerance is
preregistered from deterministic reruns and metric rounding; a failed
reproduction is reported as a baseline-integration failure and is not silently
replaced by a locally simplified look-alike.

## 4. Model architecture

### 4.1 Raw-RGB causal visual encoder

The final path uses the existing `FrameWindowDataset` and
`OnlineVideoMAEAdapter` interface, but the contract-only stub is replaced by a
real streaming visual encoder. The primary implementation is a causal VideoMAE-
style visual tower with cached temporal state and a left-only temporal adapter.
At decision time `t`, every token carries its source-frame timestamp and the
encoder asserts `source_frame <= t`.

Three training modes share the same runtime API:

1. `feature_compatibility`: consumes official or SigLIP2 cached features solely
   for fast donor reproduction and mechanism attribution;
2. `raw_rgb_adapter`: consumes RGB frames, freezes most visual-tower weights,
   and trains causal adapters plus the event model;
3. `raw_rgb_joint`: consumes RGB frames and jointly updates the registered final
   visual blocks/adapters and event model.

Only modes 2 and 3 qualify as raw-RGB results. Formal raw-RGB receipts must show
that no feature cache was loaded and that the intended visual parameters received
finite, non-zero gradients and optimizer updates.

### 4.2 Start detector and dynamic event birth

ActionSwitch supplies the immediate state-transition principle but not its fixed
switch representation. A dense causal start-hazard head predicts per-class start
evidence and an abstention/background score from the current causal prefix.
Accepted starts create ragged `EventRecord` objects rather than occupying a
preallocated semantic slot.

Each `EventRecord` contains:

- immutable event ID and detected start frame;
- start feature and causal context snapshot;
- provisional class distribution;
- recurrent event token and confidence trajectory;
- lifecycle state: provisional, active, ending, cancelled, or finalized;
- links to retained multiscale memory nodes; and
- a ledger of every provisional update.

Start acceptance is selected on a training-only calibration split using a
preregistered event-level utility that balances start recall, false starts, and
start delay. It is not fixed to probability 0.5. The threshold or sequential
stopping parameters and their hashes are frozen before report evaluation.

### 4.3 Start-conditioned continuation and end

Every active event independently consumes the new causal feature and its retrieved
history, then predicts continue, end, or cancel. The end decision is conditioned
on that event's own start token and trajectory. Therefore an end cannot freely
attach to an unrelated past start.

Training uses one-to-one instance assignment, same-class hard negatives, explicit
overlap examples, and a continuity/conservativeness loss derived from
ActionSwitch. A duplicate-birth head may suppress a new provisional event only
when it is causally judged to duplicate an already active event; it may never
delete or rewrite a finalized event.

The MATR class and boundary decoders are retained as the strong localization
donor. MATR's end-to-memory start search is implemented as a registered baseline
arm, while the primary arm replaces it with the start-conditioned lifecycle.

### 4.4 Hierarchical adaptive memory

Memory has two different responsibilities and they are not conflated:

- `ActiveEventMemory`: losslessly retains the compact state of every currently
  active event until it ends or is explicitly cancelled. It has no learned or
  manually fixed semantic instance capacity.
- `VisualHistoryMemory`: retains recent fine-grained RGB features and compresses
  older history into progressively coarser event nodes.

The HEM segment-tree/event hierarchy is the initial implementation of visual
history. HEM's fixed total `K`, dataset-frequency allocation, cosine threshold,
and FIFO fallback are reproduced as a baseline. The primary model replaces them
with sample-conditioned retention and merge decisions:

- a retention head predicts the value of each memory node for every active event;
- a merge head decides whether adjacent compatible nodes should be compressed;
- a differentiable sparsity/budget penalty learns the effective context span;
- active-event anchors cannot be pruned;
- phase-one experiments retain the full prefix physically while using continuous
  learned weights, so a bad retention decision cannot invalidate correctness;
- after validation, hard-concrete/straight-through gates materialize physical
  pruning and compression for latency and memory experiments.

A physical guard remains necessary for finite hardware, but it is not a model
slot count and cannot silently truncate history. If the guard is reached before
validated compression can preserve the contract, the formal run is invalid and
reports the guard hit.

### 4.5 Long/short context and optional OnVLLM adapter

HAT contributes its short-current/long-history refinement as a donor arm. The
full model lets the current RGB prefix query hierarchical event memory, rather
than always compressing a manually fixed history window into a fixed token count.

For the optional OnVLLM extension, finalized and active event tokens are projected
into a language decoder as structured event prompts. The language decoder may
describe ongoing events, but the authoritative temporal boundaries remain those
of the structured On-TAD head. Language generation must use the same causal
prefix and event ledger; it cannot read future frames or revise finalized spans.

## 5. Fusion matrix

The implementation explicitly tests donor combinations instead of jumping from
four reproductions to one opaque full model:

| Arm | Fusion | Question answered |
|---|---|---|
| A | faithful ActionSwitch | How much do immediate state changes and conservative transitions solve? |
| B | faithful MATR | What is the strong end-centric memory baseline? |
| C | faithful HAT/OAT | What do fixed long/short history and online suppression provide? |
| D | closed-set HEM adaptation | What do hierarchical events provide before lifecycle changes? |
| AB | ActionSwitch + MATR | Can immediate starts coexist with the stronger MATR localizer? |
| AD | ActionSwitch + HEM | Can immediate starts be maintained without fixed switches? |
| BD | MATR + HEM | Does hierarchical history improve end-centric localization? |
| BC | MATR + HAT | Does HAT history refinement improve the MATR donor under matched inputs? |
| ABD | dynamic event primary | Immediate birth + strong localization + hierarchical adaptive memory |
| ABCD | full history variant | Adds HAT refinement to ABD; retained only if it gives an isolated gain |

OnPoint attention/anticipation components are later ablations on ABD, not a fifth
training stack and not a change from full supervision.

## 6. Data flow and output semantics

For each newly arrived RGB frame or causal micro-chunk:

1. the visual encoder emits timestamped causal features;
2. the start head may create zero or more provisional event records;
3. hierarchical history incorporates or merges the new visual evidence;
4. every active event retrieves relevant recent and long-term evidence;
5. every event independently predicts class update, continuation, end, or cancel;
6. provisional messages are written to a mutable lifecycle stream;
7. an accepted end produces one immutable final interval and closes that event.

False starts may be cancelled only while provisional. A missed start remains a
miss; the primary model may not invent an unseen start at the end. EOS is an
observable current signal but does not automatically close every active event.
Every final interval must satisfy `start < end <= source_frame <= emit_frame`,
have one unique event ID, be emitted at most once, and remain immutable.

## 7. Training and experiment DAG

The work is deployed as one dependency-aware DAG rather than manually waiting
for each stage:

### Lane 1: upstream acquisition and fidelity

Clone A/B/C/D concurrently, freeze SHAs, inspect licenses, run official unit or
checkpoint evaluations, and write fidelity receipts.

### Lane 2: OpenTAD adapters and feature-level fusion

Implement four isolated adapters behind a shared `StreamingEventModel` interface.
Run A/B/C/D and AB/AD/BD/BC/ABD/ABCD as a Slurm array on official features and
on the project's frozen SigLIP2 features. Checkpoints are evaluated by calibration
workers as soon as they appear. Epoch count is not assumed in advance; official
schedules are used for fidelity and matched optimizer-update budgets are used for
fusion comparisons.

### Lane 3: raw-RGB path in parallel

While feature arms train, replace the VideoMAE stub, run raw-RGB prefix/gradient/
latency smoke tests, and prepare frozen-backbone adapter jobs. Expensive joint RGB
training is submitted in the same DAG with a dependency on the feature-level ABD
mechanism gate. Thus raw-RGB work starts immediately without spending the full
joint-training budget on a broken event mechanism.

### Lane 4: robustness and OnVLLM

CPU/small-GPU tests for same-class overlap, different-class overlap, simultaneous
ends, very long actions, false starts, adjacent actions, and future perturbations
run concurrently. The optional language adapter starts after the event-record API
is stable and does not block the mandatory raw-RGB On-TAD result.

### Lane 5: automatic confirmation

The best preregistered models launch multi-seed confirmation through scheduler
dependencies. Report evaluation runs once after calibration artifacts, model SHA,
data split hash, and decision policy hash are frozen.

## 8. Datasets and comparison policy

- THUMOS14 is the primary standard On-TAD dataset.
- MUSES is the long-context external validation dataset.
- MultiTHUMOS and/or FineAction provide dense-overlap and repeated/same-class
  stress tests.
- Official feature types are used for fidelity claims.
- All fusion comparisons use identical inputs, splits, update budgets, and output
  contracts.
- Raw-RGB results are reported separately and cannot be compared numerically to
  cached-feature results without an explicit input/backbone label.

## 9. Metrics and scientific gates

Every arm reports:

- mAP at the dataset-standard tIoUs and their average;
- event-level precision, recall, and Hungarian F1;
- prediction/GT ratio and duplicate-finalization rate;
- start delay, end delay, and wall-clock latency;
- false-start cancellation and missed-start rates;
- wrong start/end association rate, stratified by same/different class overlap;
- active-event count and learned effective-memory-length distribution;
- peak visual memory, GPU memory, FLOPs/FPS, and compression ratio;
- all causal, positive-length, uniqueness, sequence, and immutability violations.

Technical validity requires zero causal and final-ledger violations. Scientific
selection first requires fidelity baselines to be credible, then compares each
fusion against its direct parents under matched inputs. The full model must improve
the accuracy-delay-memory Pareto frontier; a gain obtained only by emitting fewer
instances is rejected. Raw-RGB promotion additionally requires real-frame input,
non-zero registered backbone updates, prefix invariance, and an end-to-end latency
receipt.

## 10. Tests and failure classification

Required tests include:

- official-output parser and metric parity for A/B/C/D;
- full-prefix versus stepwise/chunked causal equivalence;
- future-frame perturbation invariance of all already emitted messages;
- ragged batches with more live events than any old slot configuration;
- same-class overlapping events with crossed end order;
- no event can finalize twice or attach to another event's start;
- learned memory span changes with action duration and sample content;
- memory compression never removes an active start anchor;
- raw-RGB input path cannot fall back to cached predictions/features;
- intended RGB parameters receive finite gradients and optimizer updates;
- provisional cancellation never mutates a finalized detection;
- deterministic calibration and immutable report receipts.

Failures are classified as upstream-fidelity, adapter-interface, causal-contract,
optimization, association, calibration, raw-RGB integration, resource, or model-
performance failures. A failure in one class is not rewritten as another and all
usable artifacts are retained with hashes outside the source repository.

## 11. Repository boundaries and deliverables

The route repository contains only compatibility adapters, the fused model,
configs, tests, launch/check helpers, concise provenance manifests, and research
notes. External clones, datasets, feature dumps, checkpoints, logs, and generated
result archives remain outside the repository.

Implementation is divided into four bounded work packages sharing stable APIs:

1. `WP-A`: upstream acquisition, fidelity adapters, and receipts;
2. `WP-B`: dynamic event lifecycle and hierarchical adaptive memory;
3. `WP-C`: real raw-RGB causal visual encoder and joint training;
4. `WP-D`: experiment DAG, metrics, multi-seed confirmation, and optional OnVLLM
   output adapter.

The paper's main method is the raw-RGB `ABD` dynamic-event model. Feature-only
results explain why the components work; they are not presented as the final
system.
