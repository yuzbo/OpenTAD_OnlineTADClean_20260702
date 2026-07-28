# Novelty search log — 2026-07-28

Primary-source searches were run with multiple formulations over On-TAL/OAD,
online tracking, survival/event-time modelling, and recent 2025–2026 streaming
video work.  The following URLs are the evidence-bearing results; empty or
off-topic search returns are omitted.

## On-TAL and online boundary work

- MATR, ECCV 2024: https://arxiv.org/abs/2408.02957
  - Defines On-TAL as prefix-only and immutable after emission.
  - Detects an end from the current segment and scans memory for its start.
  - Its own limitation states that multiple memory instances can be matched to
    incorrect starts.
- ActionSwitch, ECCV 2024: https://arxiv.org/abs/2407.12987
  - Already claims class-agnostic overlapping and same-class On-TAL.
  - Therefore overlap/same-class support alone is not a D1 novelty claim.
- HAT, ECCV 2024: https://arxiv.org/abs/2408.06437
  - Long-term history, instance anchors, and imbalance-aware loss are prior art.
- CMeRT, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Pang_Context-Enhanced_Memory-Refined_Transformer_for_Online_Action_Detection_CVPR_2025_paper.html
  - Explicitly diagnoses OAD training/inference discrepancy and non-causal
    leakage from anticipated-future interactions.
- Online Generic Event Boundary Detection, 2025:
  https://arxiv.org/abs/2510.06855
  - Prefix-only online boundaries are prior art, but its taxonomy-free boundary
    task is not supervised action-instance lifecycle localization.
- OZ-TAL, 2026: https://arxiv.org/abs/2605.09976
  - Recent zero-shot/open-world On-TAL; different supervision and task.
- OnPoint, 2026: https://arxiv.org/abs/2607.00289
  - Recent point-supervised On-TAL; different supervision and does not by
    itself answer D1's fully supervised event-lifecycle claim.

## Identity and chronological unroll

- TrackFormer, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Meinhardt_TrackFormer_Multi-Object_Tracking_With_Transformers_CVPR_2022_paper.html
  - Identity-preserving propagated track queries and new/static queries are
    established.
- MeMOTR, ICCV 2023: https://arxiv.org/abs/2307.15700
  - Long-term memory explicitly stabilizes and separates identity embeddings.
- 3DMOTFormer, ICCV 2023:
  https://openaccess.thecvf.com/content/ICCV2023/html/Ding_3DMOTFormer_Graph_Transformer_for_Online_3D_Multi-Object_Tracking_ICCV_2023_paper.html
  - Autoregressive recurrent online training to reduce train/inference
    distribution mismatch is prior art.
- Eliminating Exposure Bias and Metric Mismatch in MOT, CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Maksai_Eliminating_Exposure_Bias_and_Metric_Mismatch_in_Multiple_Object_Tracking_CVPR_2019_paper.html
  - Training on the model's own tracking mistakes is prior art.

## Censoring and event time

- Neural Conditional Event Time Models, MLHC 2020:
  https://proceedings.mlr.press/v126/engelhard20a.html
  - Neural maximum-likelihood learning from right-censored event times is
    established.
- Learning Hawkes Processes from Short Doubly-Censored Event Sequences, ICML
  2017: https://proceedings.mlr.press/v70/xu17b.html
  - Censored event sequences are not a new statistical primitive.
- Survival Seq2Seq, MLHC 2022:
  https://proceedings.mlr.press/v182/pourjafari22a.html
  - Sequential neural time-to-event prediction with censored data is prior art.

## Search formulations

Queries included at least three variants per claim, including:

- `online temporal action localization stable temporal assignment interval censored birth hazard`
- `online action detection temporal assignment query identity`
- `online temporal action interval-censored`
- `right-censored action duration prediction video hazard neural`
- `temporal point process video event boundary detection online`
- `online temporal action localization same class overlap identity tracking query`
- `MOTR track query identity lock birth death reacquisition`
- `MeMOTR stable identity query memory`
- `oracle predicted tracks training inference mismatch transformer`
- `scheduled sampling oracle predicted trajectories online tracking`
- `differentiable unroll online video tracking query propagation`
- `online action future information leakage causal protocol`
- `streaming temporal action localization offline future context leakage`
- `2026 online temporal action localization`
- `2026 online action detection transformer streaming video`

## Search conclusion

No searched primary source was found that jointly instantiates, for standard
fully supervised strict-causal On-TAL, interval-censored first birth,
right-censored instance end, post-birth action identity with
cancel/reacquisition, and a shared ragged chronological train/infer unroll.
This is a bounded search result, not proof of global novelty.
