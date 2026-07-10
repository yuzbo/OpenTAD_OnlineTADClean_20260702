# CausalTAD

> [Harnessing Temporal Causality for Advanced Temporal Action Detection](https://arxiv.org/abs/2407.17792)
> Shuming Liu, Lin Sui, Chen-Lin Zhang, Fangzhou Mu, Chen Zhao, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

As a fundamental task in long-form video understanding, temporal action detection (TAD) aims to capture inherent temporal relations in untrimmed videos and identify candidate actions with precise boundaries. Over the years, various networks, including convolutions, graphs, and transformers, have been explored for effective temporal modeling for TAD. However, these modules typically treat past and future information equally, overlooking the crucial fact that changes in action boundaries are essentially causal events.
Inspired by this insight, we propose leveraging the temporal causality of actions to enhance TAD representation by restricting the model's access to only past or future context. We introduce CausalTAD, which combines causal attention and causal Mamba to achieve state-of-the-art performance on multiple benchmarks. Notably, with CausalTAD, we ranked 1st in the Action Recognition, Action Detection, and Audio-Based Interaction Detection tracks at the EPIC-Kitchens Challenge 2024, as well as 1st in the Moment Queries track at the Ego4D Challenge 2024.

## Results and Models

**ActivityNet-1.3**

| Features | Classifier | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :--------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSP    |    CUHK    |  55.62  |  38.51   |   9.40   |  37.46   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1s9mzuOqc-KSoBg6xyEZgb-X5SzR5Y6i-/view?usp=sharing)   \| [log](https://drive.google.com/file/d/12hej5jRg_FX9v-ShT2epAvG83lxBv8NR/view?usp=sharing) |


**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  84.43  |  80.75  |  73.57  |  62.70  |  47.33  |  69.75   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1P7O8RJp-gX_gBY2RQgYk9CZwHbdY2ef4/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1ImdvPnX56npu-ZqHFvm_Fxf7bY9x6Z-1/view?usp=sharing) |


**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  28.13  |  26.79  |  25.16  |  22.63  |  18.70  |  24.28   | [config](epic_slowfast_noun.py) | [model](https://drive.google.com/file/d/186JsNtmsSYOe_HFIG6UvhOfAEt_xpeg5/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1k5lU-ArJ1h5Vnvz05azlMdJ0MpgBKv03/view?usp=sharing) |
|  Verb  | SlowFast |  29.62  |  28.69  |  27.16  |  25.24  |  21.44  |  26.43   | [config](epic_slowfast_verb.py) | [model](https://drive.google.com/file/d/1icQcNStbjRvdiu71JpgU_h5WMgymbNNS/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1muFQYyB1__3NZrODwtu5fHAGkisspJa0/view?usp=sharing) |

**Ego4D-MQ**

|   Features   | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :----------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo1 |  37.68  |  35.28  |  32.23  |  29.49  |  26.29  |  32.19   | [config](ego4d_internvideo1.py) | [model](https://drive.google.com/file/d/1SC3XFSSwguJG8_8DhdYi8doB6W6Ayfne/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1BLTbyw_lSnWtjHZY1tZO_laF_Chgye_h/view?usp=sharing) |
| InternVideo2 |  39.01  |  36.05  |  33.06  |  30.45  |  26.70  |  33.05   | [config](ego4d_internvideo2.py) | [model](https://drive.google.com/file/d/1U2k9RLHNiCDSlppAPUl5GADYmfKtQlZ0/view?usp=sharing)   \| [log](https://drive.google.com/file/d/14D-q6N7RiCgmRexFPiozjpa0BiaQGnlI/view?usp=sharing) |

For our solution to Ego4D Challenge 2024 and EPIC-Kitchens Challenge 2024, please refer to [here](egovis_challenge_2024/README.md), including detailed challenge config and ensemble strategy.

## Experimental Raw-Frame Online TAD Validation Candidates

- [thumos_videomae_adapter_matr_ontad.py](thumos_videomae_adapter_matr_ontad.py): raw-frame VideoMAE adapter + MATR-style head validation candidate. It explicitly declares `input_format="raw_frames"`, enables strict causal projection, uses a causal adapter optimizer group while the base backbone is frozen, and keeps raw-prediction shortcuts disabled. The config marks its stub backbone as contract-only until a causal/streaming VideoMAE with masked attention and cache state is wired in; formal training claims require disabling the stub, passing single-rank emission-ledger evaluation, and running on remote Slurm.

- `thumos_siglip2_matr_ontad_p0.py`, `thumos_siglip2_matr_ontad_p1.py`, and `thumos_siglip2_motion_matr_ontad_p2.py` are raw-frame SigLIP/SigLIP2 validation candidates. Their streaming-safe ledger evaluator is single-rank only for now; DDP evaluation must wait for a video-contiguous sampler or centralized online state machine.

- `thumos_siglip2_adaptive_matr_ontad_final.py` is retained under its legacy filename as a controlled fixed-stride chunk-end baseline. Its runtime selector policy is `fixed_causal_stride2`, its decision cadence is one check per non-overlapping window, and `MATRHead(memory_size=0)` is recorded as a baseline head rather than a MATR reproduction. It must not be used to support adaptive-selection or strict rolling On-TAD claims.

- [thumos_pceh_ontad.py](thumos_pceh_ontad.py) is the strict packet-based PCEH-OnTAD candidate. It uses 8-frame chronological packets, an incremental bounded feature cache, independent class/start/ongoing/end/completion/emission hazards, immutable read-provenance ledgers, and `OnlineAPBudgeted` at 0.5/1/2/4 seconds. [thumos_pceh_chunk_end_baseline.py](thumos_pceh_chunk_end_baseline.py) and [thumos_pceh_rolling_fixed_stride2.py](thumos_pceh_rolling_fixed_stride2.py) are controlled schedule/acquisition baselines. All three remain `formal_training_ready=False` until remote path validation, optimizer coverage, causal compliance, and a single-rank smoke run pass.

- [thumos_pceh_endpoint_only.py](thumos_pceh_endpoint_only.py) is the fair rolling endpoint-only control: it preserves PCEH input, cadence, cache, visual compute, optimizer, and evaluator while removing completion/emission decisions and losses. The two `*_pilot.py` configs run a three-epoch single-seed decision pilot with the same scheduler and are not formal-result configs.

Captured packet, ledger, future-perturbation, and chunk-invariance artifacts can be checked without loading a model:

```shell
python tools/check_causal_compliance.py packets packets.json
python tools/check_causal_compliance.py ledger pceh_emission_ledger.json
python tools/check_causal_compliance.py future-perturbation reference.json perturbed.json --cut 120
python tools/check_causal_compliance.py chunk-invariance chunk1.json chunk8.json
```

On N16R4, first submit the gated smoke and inspect its artifacts. Pilot submission requires the passed smoke directory explicitly:

```shell
bash tools/remote/submit_pceh_n16r4.sh smoke configs/causaltad/thumos_pceh_ontad.py
bash tools/remote/check_pceh_n16r4.sh /absolute/smoke/run/dir JOB_ID
ALLOW_PILOT=1 SMOKE_RUN_DIR=/absolute/smoke/run/dir bash tools/remote/submit_pceh_n16r4.sh pilot configs/causaltad/thumos_pceh_pilot.py
```


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CausalTAD on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/thumos_i3d.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CausalTAD on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/thumos_i3d.py --checkpoint exps/thumos/causal_i3d/gpu1_id0/checkpoint/epoch_38.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@article{liu2024harnessing,
  title={Harnessing Temporal Causality for Advanced Temporal Action Detection},
  author={Liu, Shuming and Sui, Lin and Zhang, Chen-Lin and Mu, Fangzhou and Zhao, Chen and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2407.17792},
  year={2024}
}
```
