# Strict Review Prompt: Online TAD + Adaptive Frame Selection

Use this prompt with a high-tier code/research reviewer. The reviewer must use the public GitHub repository and must not rely on copied snippets alone.

Repository:

https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702

Review target:

- Branch: `codex/online-tad-clean-20260702`
- Scope: current Online/Causal TAD implementation, SigLIP/SigLIP2 raw-frame route, MATR-style dense head integration, emission ledger evaluation, full-dataset P1 training route, and the next step toward end-to-end adaptive frame selection + AdaTAD.

Required review style:

1. Perform a line-by-line code review of the relevant implementation. Do not give a high-level-only answer.
2. Every finding must include file path, line number or narrow line range, severity, and a concrete fix.
3. Distinguish confirmed bugs from risks, missing tests, and design weaknesses.
4. Verify causal/online correctness explicitly. Check whether any future frames, future labels, overlapping-window memory, cached raw predictions, GT shortcuts, or offline post-processing leak into online decisions.
5. Verify that the current model can be trained end to end from raw frames for the advertised route, including dataset import, frame loading, processor behavior, backbone/projection/head gradient flow, optimizer parameter groups, loss wiring, and evaluation call path.
6. Judge whether the current experiments are elegant enough for a paper-quality route. Be strict about pilot subsets, full validation semantics, emission counts, latency metrics, FPS/stride/seconds roundtrip, and whether claims match actual evidence.
7. Propose the next implementation plan for an end-to-end adaptive frame selection + AdaTAD model without causing performance collapse.

Files that must be inspected at minimum:

- `configs/causaltad/thumos_siglip2_matr_ontad_p0.py`
- `configs/causaltad/thumos_siglip2_matr_ontad_p1.py`
- `configs/causaltad/thumos_siglip2_matr_ontad_p1_fix.py`
- `configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py`
- `configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py`
- `configs/causaltad/thumos_videomae_adapter_matr_ontad.py`
- `opentad/datasets/raw_frame.py`
- `opentad/datasets/transforms/end_to_end.py`
- `opentad/models/backbones/online_siglip_adapter.py`
- `opentad/models/backbones/online_videomae_adapter.py`
- `opentad/models/projections/causal_temporalmaxer_proj.py`
- `opentad/models/projections/causal_proj.py`
- `opentad/models/dense_heads/matr_head.py`
- `opentad/cores/train_engine.py`
- `opentad/cores/test_engine.py`
- `opentad/evaluations/mAP.py`
- `opentad/utils/online_protocol.py`
- `tools/remote/submit_siglip_ontad_n16r4.sh`
- all tests under `tests/` that mention SigLIP, raw frames, online protocol, mAP, emission ledger, or causal projections.

Questions to answer:

1. Is the current implementation actually causal and online, or only a windowed offline approximation? Give a precise verdict.
2. Is the current P1 full-dataset/full-validation training job valid as an end-to-end raw-frame online TAD experiment? If not, list the blockers.
3. Are there implementation bugs that can silently inflate or destroy mAP, especially around timestamps, window starts, feature-grid mapping, label assignment, NMS, and evaluation filtering?
4. Are the optimizer groups correct? Check frozen versus trainable backbone parts, projection, adapter, and detection head.
5. Are emission ledger and latency statistics faithful to online deployment? If not, specify the exact failure mode.
6. Are the current tests sufficient? Add the missing minimum tests with names and expected assertions.
7. Is the experimental route elegant? If it is only engineering glue, say so and explain what would make it a paper-worthy contribution.
8. How should we implement adaptive frame selection + AdaTAD end to end while avoiding performance collapse?

Required AdaTAD / adaptive frame selection proposal:

- Define the selector input, output, and budget constraint.
- Specify whether selection is hard, soft, or hybrid, and how gradients flow.
- Specify how selected frames are converted into temporal tokens and aligned back to seconds.
- Specify how the detector head consumes irregular or budgeted temporal tokens.
- Specify how to avoid collapse to selecting only easy/background/static frames.
- Specify the training schedule: warm-up, frozen encoder phase, selector phase, detector phase, and joint fine-tuning phase.
- Specify auxiliary losses or regularizers: budget loss, entropy/diversity, temporal coverage, actionness distillation, boundary preservation, and latency penalty.
- Specify online constraints: no future frames, bounded memory, per-video state keying, emission protocol, and maximum decision latency.
- Specify ablations needed to support the claim.

Expected output format:

1. Verdict: one paragraph, direct and strict.
2. Blocking bugs: bullet list with file/line references.
3. Causality audit: pass/fail table.
4. Training validity audit: pass/fail table.
5. Experiment elegance audit: what is solid, what is weak, and what claims are not yet supported.
6. Line-by-line notes: file-grouped findings with severity and fix.
7. Concrete implementation plan for adaptive frame selection + AdaTAD.
8. Minimal code sketch for the selector, token packing/alignment, loss terms, and detector interface.
9. Test plan and expected commands.
10. Final recommendation: continue, revise, or pivot.
