## Verdict

**结论：当前分支不能被称为“已可论文 claim 的端到端 raw-frame online TAD 训练实现”。**
它更准确地说是：一个已经开始具备 **streaming-safe emission protocol / raw-frame SigLIP frozen encoder / causal projection / MATR-style head glue** 的工程原型；其中 P0 的 SigLIP/SigLIP2 路线在静态代码层面基本遵守“无 feature cache / 无 raw prediction cache / 无 video-level NMS / 单 rank emission ledger”的在线评估约束，但 **P1 full-60 配置链存在阻塞级继承错误**，VideoMAE 路线仍是 **contract-only stub**，训练监督仍有 **未来 endpoint label shortcut 风险**，评估是 **窗口级批处理 emission**，不是低延迟连续流式 agent。仓库 README/status 也明确说当前没有特征缓存、raw-prediction 缓存、checkpoint/result 文件，raw-frame routes 仍处于 causal-audit validation，single-rank emission ledger 且 **not paper-ready**。

我做的是公开 GitHub 静态源码审查；没有实际跑 60 epoch 训练或 mAP。由于很多 raw 文件被压缩成 1–5 个超长行，下面的 line number 采用 GitHub raw/source 的实际行号，并在必要处补充“逻辑段”。

---

## Blocking bugs

### 1. `p1_fix.py` / `p1_full_60.py` 的 config base chain 是坏的

**Type:** confirmed bug
**Severity:** blocker
**Files / lines:**
`configs/causaltad/thumos_siglip2_matr_ontad_p1_pilot.py:L1`
`configs/causaltad/thumos_siglip2_matr_ontad_p1_fix.py:L0-L1`
`configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py:L0-L1`

`p1_fix.py` 以 `p1_pilot.py` 为 base，而 `p1_pilot.py` 本身没有 `_base_` 指向 P0/P1，因此它没有继承 P0 的 `model`、raw-frame dataset、pipeline、inference、streaming-safe post-processing 等核心定义。`p1_full_60.py` 又以 `p1_fix.py` 为 base，所以 full-60 配置链同样断裂。`p1_full_60.py` 的注释声称 full THUMOS raw-frame online TAD training/eval 60 epochs，但同一行配置仍标 `formal_training_ready=False`。

**修复方案：**

```python
# configs/causaltad/thumos_siglip2_matr_ontad_p1_pilot.py
_base_ = "./thumos_siglip2_matr_ontad_p1.py"
```

或：

```python
# configs/causaltad/thumos_siglip2_matr_ontad_p1_fix.py
_base_ = "./thumos_siglip2_matr_ontad_p1.py"
```

然后给所有 P1/P1-fix/P1-full 配置加 config-load/build 测试，至少断言：

```python
assert cfg.model.backbone.type == "OnlineVideoMAEAdapter"
assert cfg.model.backbone.raw_frame_encoder.type == "OnlineSigLIPFrameEncoder"
assert cfg.inference.load_from_raw_predictions is False
assert cfg.inference.save_raw_prediction is False
assert cfg.post_processing.streaming is True
assert cfg.post_processing.streaming_safe_emission is True
assert cfg.post_processing.sliding_window is False
```

---

### 2. `p1_full_60.py` 目前不能支撑“full-dataset/full-validation online TAD 训练”结论

**Type:** confirmed bug + claim blocker
**Severity:** blocker
**Files / lines:**
`configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py:L0-L1`

因为 base chain 断裂，full-60 并没有可靠继承 P0/P1 的模型、raw-frame pipeline、streaming-safe inference contract。即使修复继承，当前 full-60 也仍然显式标记 `formal_training_ready=False`。因此它不能被称为已经完成的 full-dataset/full-validation formal experiment。

**修复方案：**

1. 修复 `_base_`。
2. 显式保留 streaming-safe 关键项，不依赖 merge 语义：

```python
post_processing = dict(
    streaming=True,
    streaming_safe_emission=True,
    sliding_window=False,
    max_latency=0.0,
    save_emission_ledger=True,
    save_latency_summary=True,
    nms=dict(...),
)
```

3. 增加 CI 测试：加载 full-60 config，build dataset/model/evaluator，跑 2 个 tiny video 的 train/eval smoke。
4. 只有通过 full dataset 训练、streaming ledger validation、no-future perturb、mAP evaluator 过滤一致性后，才能把 `formal_training_ready` 改成 `True`。

---

### 3. 训练监督存在 future endpoint label shortcut 风险

**Type:** risk / likely design bug for strict online claim
**Severity:** high
**Files / lines:**
`opentad/datasets/raw_frame.py:get_gt / __getitem__ logical segment`
`opentad/models/dense_heads/matr_head.py:L2`

`FrameWindowDataset` 把完整 GT segment 转成 window-local grid target；MATR head 的训练 loss 仍按完整 segment 回归 start/end。推理时 P0/P1 的 head 会把 proposal end clamp 到当前 point / current grid，但训练时 target 仍可能让当前 prefix 内的 token 学到未来动作结束位置。这不是 feature 未来帧泄漏，但属于 **future label shortcut**：训练目标知道当前时刻之后尚未发生的 endpoint。

**修复方案：**

为 online training 增加 censored-target 模式：

```python
# 对每个 training point t，只允许监督 observed prefix 内的信息
observed_end = min(gt_end, current_grid + max_future_offset)
target_end = observed_end
end_is_censored = gt_end > current_grid + max_future_offset

# 对被 censor 的样本：
# 1. 不回归真实 end；
# 2. 或只训练 actionness / inside-action / lower-bound duration；
# 3. end/boundary loss 只在 endpoint 已 observed 时启用。
```

并在 head loss 中区分：

```python
if online_censored_training:
    loss_reg_end = loss_reg_end * endpoint_observed_mask
    loss_actionness = loss_actionness * valid_prefix_mask
```

---

### 4. VideoMAE route 不是实际 VideoMAE online TAD route

**Type:** confirmed design limitation / claim blocker
**Severity:** high
**Files / lines:**
`configs/causaltad/thumos_videomae_adapter_matr_ontad.py:L0-L1`
`opentad/models/backbones/online_videomae_adapter.py:L849-L936`

该配置明确写着 backbone 是 contract-only stub，`formal_training_ready=False`；adapter 代码也要求如果使用 stub 必须显式 `stub_backbone_contract_only=True`。因此当前 VideoMAE adapter + MATR route 只能证明接口 contract，不证明 VideoMAE online visual encoder、真实梯度、真实性能或 causal attention 正确。

**修复方案：**

1. 接入真实 VideoMAE causal/streaming backbone。
2. 去掉 `use_stub_backbone=True`。
3. 增加 no-future perturb test：扰动未来 raw frames，不允许过去 tokens/head outputs 变化。
4. 增加 gradient test：确认 adapter/projection/head 有梯度；若解冻 VideoMAE，确认被解冻 block 有梯度。
5. 只有上述通过后才允许 claim “VideoMAE adapter route”。

---

### 5. `train_engine.py` 对 `model.module` 的假设会破坏单卡/非 DDP 路径

**Type:** confirmed bug / robustness bug
**Severity:** medium-high
**File / line:**
`opentad/cores/train_engine.py:L0`

训练 engine 中存在直接访问 `model.module` 的路径，例如记录 backbone LR 时先访问 `model.module.backbone`。如果模型没有被 DDP/DataParallel 包裹，`model.module` 会触发 attribute error。

**修复方案：**

```python
target_model = model.module if hasattr(model, "module") else model

if hasattr(target_model, "backbone"):
    backbone = target_model.backbone
    ...
```

同时加单卡非 DDP train step smoke test。

---

## Causality audit table

| Component                      |                                                          Current status | Verdict                          | Evidence / issue                                                                                                                                                  |
| ------------------------------ | ----------------------------------------------------------------------: | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SigLIP/SigLIP2 raw-frame route | Per-frame encoder, frozen vision tower, no feature/raw-pred cache in P0 | Mostly causal at feature level   | P0 config uses raw video loading, `strict_online=True`, `memory_size=0`, `load_from_raw_predictions=False`, `save_raw_prediction=False`, streaming-safe postproc. |
| SigLIP encoder                 |                  Enforces one frame per token path; frozen no-grad path | Good for P0                      | Adapter enforces shape and `frame_stride == 1`; frozen encoder runs under no-grad.                                                                                |
| CausalTemporalMaxer projection |                                        Left-padded causal conv/max-pool | Good, but needs whole-model test | Projection code uses left padding and causal downsample/max-pool.                                                                                                 |
| MATR head inference            |                           End clamp to current grid; P0 memory disabled | Mostly causal                    | `clamp_end_to_current=True`, `memory_size=0` in P0/P1. Head clamps end during inference.                                                                          |
| MATR head training             |                                            Uses full GT endpoint target | High risk                        | Future endpoint label shortcut; must add censored online target mode.                                                                                             |
| Stream memory                  |                     P0/P1 disabled; VideoMAE config uses bounded memory | P0 OK; VideoMAE needs tests      | MATR resets memory on non-contiguous/gap/overlap, but VideoMAE route is stub and overlap-heavy.                                                                   |
| Window overlap                 |             P0/P1 `window_overlap_ratio=0.0`; VideoMAE val/test overlap | P0 OK; VideoMAE risk             | Overlap + memory requires strict reset / prefix-stability tests.                                                                                                  |
| Emission protocol              |                                   Streaming-safe path avoids global NMS | Good protocol skeleton           | Single-stage detector calls `OnlineEmitter`; test engine disables sliding window and video-level NMS under streaming-safe mode.                                   |
| Max latency                    |         Config says `max_latency=0`, but emission happens at window end | Design weakness                  | Early-window detections are emitted only when the whole window batch is processed. This is no-future, but not low-latency continuous streaming.                   |
| Offline NMS leakage            |   Disabled only if `streaming_safe_emission=True` survives config merge | P0 OK; P1 full currently broken  | `test_engine` bypasses video-level NMS when streaming-safe; full-60 must explicitly retain this.                                                                  |
| Raw prediction cache           |                                                   P0 disables load/save | OK for P0                        | P0 config explicitly disables raw predictions.                                                                                                                    |
| GT shortcut                    |                                         No obvious GT in inference path | Mostly OK                        | Risk is training label censoring, not eval GT shortcut.                                                                                                           |
| mAP filtering                  |                                    Allowed-videos filters both GT/preds | OK, limited tests                | Evaluator supports `allowed_videos`; tests cover basic filter contract.                                                                                           |

---

## Training validity audit table

| Question                                                      | Answer                                                                       | Severity | Required fix before claim                                                                    |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------: | -------------------------------------------------------------------------------------------- |
| Can current `p1_full_60.py` be run as formal full-dataset P1? | **No.** Config inheritance is broken and flag remains false.                 |  blocker | Fix base chain, add config-build smoke, run full train/eval.                                 |
| Is it end-to-end raw-frame?                                   | **Partially.** It starts from raw frames, but SigLIP vision tower is frozen. |   medium | Say “raw-frame input with frozen visual encoder,” not full visual E2E.                       |
| Is VideoMAE route end-to-end?                                 | **No.** Stub only.                                                           |     high | Replace stub with real causal VideoMAE and tests.                                            |
| Are optimizer parameter groups proven correct?                | Not fully.                                                                   |   medium | Add tests checking frozen params absent / trainable params present / LR groups match config. |
| Is gradient flow proven?                                      | Tiny SigLIP smoke exists, but not full P1 config chain.                      |   medium | Add full config tiny train step.                                                             |
| Is training label assignment online-safe?                     | Not yet.                                                                     |     high | Censored endpoint training / emit-only supervision.                                          |
| Is validation streaming-safe?                                 | P0 skeleton yes; full-60 not proven because config broken.                   |     high | Validate ledger and assert no video-level NMS.                                               |
| Is mAP口径完整验证？                                                 | Basic allowed-videos test exists; streaming ledger-to-mAP test missing.      |   medium | Add end-to-end ledger rows → evaluator test.                                                 |
| Can full-60 support paper mAP claim?                          | **No.**                                                                      |  blocker | Need fixed config, full run, ablations, latency/mAP ledger, causal tests.                    |

---

## Experiment elegance audit

当前实现的优点是：P0 的目标很干净，避免了 feature cache/raw prediction cache；SigLIP frozen encoder + causal projection + memory_size=0 + streaming-safe emission ledger，是一个合适的 **P0 causal-audit baseline**。测试目录也已经有 online protocol、mAP contract、SigLIP runtime smoke、causal projection、VideoMAE contract 等相关测试文件。

但它还不优雅，主要原因：

1. **P1 full-60 配置链断裂**，这不是实验细节，而是 formal training blocker。
2. **VideoMAE route 是 stub**，不能作为真实模型路线。
3. **online emission 是窗口级批处理 emission**，不是 per-token/per-frame continuous online agent。它可以 no-future，但早期动作需要等窗口结束才 emission，实际最大决策延迟不等于 `max_latency=0`。
4. **训练目标仍像 offline TAD 一样知道完整 GT endpoint**，这会削弱 online/causal claim。
5. **adaptive frame selection + AdaTAD 尚未实现**，当前最多是 raw-frame frozen encoder + MATR-style detector glue，不是“模型主动决定看哪些帧”。

因此，当前可以 claim 的最强表述应限制为：

> “We implemented a raw-frame, frozen-SigLIP, windowed causal TAD prototype with streaming-safe emission ledger and no raw-prediction cache under P0 audit.”

还不能 claim：

> “End-to-end raw-frame online TAD training,”
> “VideoMAE online TAD,”
> “Low-latency streaming detector,”
> “Adaptive frame selection + AdaTAD,”
> “Paper-ready full THUMOS mAP.”

---

## Line-by-line findings

| File / lines                                                        | Type                             |    Severity | Finding                                                                                                            | Concrete fix                                                            |
| ------------------------------------------------------------------- | -------------------------------- | ----------: | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `configs/causaltad/thumos_siglip2_matr_ontad_p0.py:L1`              | design weakness                  |      medium | P0 是 frozen SigLIP/SigLIP2 raw-frame route，不是 visual encoder full finetune。                                        | 文档和实验表中写成 “raw-frame input, frozen visual encoder”。                     |
| `configs/causaltad/thumos_siglip2_matr_ontad_p0.py:L1`              | risk                             |      medium | `max_latency=0.0` 但 detector 一次处理整个 window，emission 发生在 window end；早期 segment latency 会很大。                         | ledger 中报告 p50/p95/max latency；不要把它叫 zero-latency online。               |
| `configs/causaltad/thumos_siglip2_matr_ontad_p0.py:L1`              | confirmed good                   |        info | 明确禁用 `load_from_raw_predictions` / `save_raw_prediction`，且 `streaming_safe_emission=True`。                         | 保持；在 P1/full configs 显式继承并测试。                                           |
| `configs/causaltad/thumos_siglip2_matr_ontad_p1.py:L0-L1`           | risk                             |      medium | P1 自己标 `formal_training_ready=False`，说明它不是正式训练配置。                                                                  | 通过 full config smoke、ledger、mAP 后再改 True。                               |
| `configs/causaltad/thumos_siglip2_matr_ontad_p1_pilot.py:L1`        | confirmed bug                    |     blocker | 没有 `_base_`，所以不是 P0/P1 的子配置。                                                                                       | 加 `_base_ = "./thumos_siglip2_matr_ontad_p1.py"`。                       |
| `configs/causaltad/thumos_siglip2_matr_ontad_p1_fix.py:L0`          | confirmed bug                    |     blocker | `_base_="./thumos_siglip2_matr_ontad_p1_pilot.py"` 继承坏配置。                                                          | 改成继承 P1，或修复 pilot base。                                                 |
| `configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py:L0`      | confirmed bug                    |     blocker | 继承 `p1_fix.py`，因此 full-60 同样断链。                                                                                    | 修复 base chain；加 config load/build test。                                 |
| `configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py:L1`      | claim blocker                    |        high | 注释说 full training，但 `formal_training_ready=False`。                                                                 | 不允许 paper claim；通过全套 gate 后再改。                                          |
| `configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py:L1`      | risk                             |        high | `post_processing` 未显式写 `streaming_safe_emission=True`；依赖 merge 语义不稳。                                               | 在 full config 显式写全 streaming-safe 字段。                                   |
| `configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py`          | missing test                     |      medium | motion branch 路线需要 no-future 和 gradient 测试。                                                                        | 加扰动未来帧测试、motion branch 梯度测试。                                            |
| `configs/causaltad/thumos_videomae_adapter_matr_ontad.py:L0-L1`     | claim blocker                    |        high | 配置明确是 contract-only stub，不是真 VideoMAE route。                                                                       | 接入真实 causal VideoMAE；去掉 stub flag。                                      |
| `configs/causaltad/thumos_videomae_adapter_matr_ontad.py:L1`        | causality risk                   |      medium | val/test `window_overlap_ratio=0.5` 且 head memory_size=256；需要证明 overlap reset 和 prefix stability。                  | 加 overlapping-window memory reset + no duplicate emission test。         |
| `opentad/datasets/raw_frame.py:get_gt/__getitem__`                  | risk / design bug                |        high | 训练 label 使用完整 GT segment endpoint，online prefix 训练会知道未来 endpoint。                                                  | 加 `online_censored_training`，未观测 endpoint 不回归真实 end。                    |
| `opentad/datasets/raw_frame.py:window meta segment`                 | missing test                     |      medium | `window_start_frame`、`window_end_frame`、`feature_start_idx`、`feature_end_idx` 需要 roundtrip 测试覆盖 video/rawdir 两种输入。 | 加 frame→grid→seconds→frame exact/±1 测试。                                 |
| `opentad/datasets/transforms/end_to_end.py:L3462-L3474`             | missing test                     |      medium | `LoadFrames(method="sliding_window")` 的 frame indices / valid mask 需要和 P0 snippet_stride 对齐测试。                     | 构造 start=0,stride=8,window=192 的 test，断言 indices。                       |
| `opentad/datasets/transforms/end_to_end.py:L2523-L2537`             | risk                             |  low-medium | `LoadSnippetFrames` 的 `clip_len=1` 分支有空 index 风险；P0 当前不用它。                                                         | 修成 `np.arange(clip_len) - clip_len // 2` 或 clip_len=1 special case。     |
| `opentad/datasets/transforms/loading.py:L1230-L1277`                | risk                             |      medium | `LoadRawFrames` 默认 `start_index=1`，video/cv2 通常应 0-based；P0 显式设 0，其他配置可能踩坑。                                        | video format 默认 0；或强制 config 必须显式设 start_index。                         |
| `opentad/datasets/transforms/loading.py:L1313-L1378`                | missing test                     |      medium | cv2 seek 使用 `frame_idx + start_index`，需要防 off-by-one。                                                              | 合成视频测试：读取第 0/1/8 帧像素编码。                                                 |
| `opentad/models/backbones/online_siglip_adapter.py:L1406-L1428`     | confirmed good                   |        info | raw-frame shape 约束清楚，`num_clips=1`；避免多 clip 混合。                                                                    | 保持；加 P1 full config runtime test。                                       |
| `opentad/models/backbones/online_siglip_adapter.py:L1162-L1218`     | confirmed good                   |        info | 拒绝 `frame_stride != 1`，避免 adapter 内部再跨帧采样。                                                                         | 保持。                                                                     |
| `opentad/models/backbones/online_siglip_adapter.py:L1634-L1693`     | design limitation                |      medium | frozen encoder no-grad 正确，但这意味着不是 full visual finetuning。                                                          | claim 限定；如需 E2E，解冻最后 N blocks 并加 no-future/grad tests。                  |
| `opentad/models/backbones/online_videomae_adapter.py:L849-L936`     | claim blocker                    |        high | `strict_online` 对真实 backbone 多为声明式检查；stub route 不证明 causal VideoMAE。                                               | 真实 backbone 必须通过数值 no-future perturb。                                   |
| `opentad/models/backbones/online_videomae_adapter.py:L1063-L1107`   | missing test                     |      medium | forward 支持 freeze/no-freeze，但缺真实 VideoMAE 梯度与 mask 测试。                                                             | 加 trainable/frozen param group + forward/backward tests。                |
| `opentad/models/projections/causal_temporalmaxer_proj.py:L539-L562` | confirmed good                   |        info | causal conv 使用 left padding。                                                                                       | 保持；做全模型 no-future test。                                                 |
| `opentad/models/projections/causal_temporalmaxer_proj.py:L688-L761` | confirmed good / missing test    | info-medium | causal maxpool/downsample 逻辑合理，但需要和 head/source_grid 对齐测试。                                                         | 加 FPN level source time alignment test。                                 |
| `opentad/models/projections/causal_proj.py`                         | risk                             |      medium | strict causal 依赖外部 block/kernel 配置；VideoMAE route 又是 stub。                                                         | 对真实 strict path 做 perturbation test。                                    |
| `opentad/models/dense_heads/matr_head.py:L2`                        | risk                             |        high | training loss 仍可能回归未来 endpoint；inference clamp 不能解决训练 shortcut。                                                    | online censored target/loss。                                            |
| `opentad/models/dense_heads/matr_head.py:L2-L4`                     | confirmed good                   |        info | P0/P1 `memory_size=0` 时无跨窗口 memory；inference 可以 clamp end 到 current。                                               | 保持；测试 memory_size>0 路线。                                                 |
| `opentad/models/detectors/single_stage.py:L1-L3`                    | confirmed good / design weakness |      medium | streaming-safe path绕开普通 NMS，调用 OnlineEmitter；但 emission 粒度是 window end。                                            | 增加 per-token emission 或明确 latency contract。                             |
| `opentad/utils/online_protocol.py:L0`                               | design weakness                  |      medium | `candidate_source_grid` 忽略 `flattened_index`，对多 FPN level 的 source 时间可能不够严谨。                                       | 从 head 返回 point center / level stride；source_grid 用真实 point time。       |
| `opentad/utils/online_protocol.py:L0`                               | risk                             |      medium | `last_emitted_grid` 只随 emitted proposals 前进；被低分过滤的旧 grid 可能后续在 overlap/memory 路线重现。                                | 加 `seen_source_grid_watermark`，或要求 non-overlap + memory disabled。       |
| `opentad/cores/test_engine.py:L0-L2`                                | confirmed good / risk            |      medium | streaming-safe 时验证 world size、禁用 sliding window/NMS、保存 ledger；但 full config 必须保留 flag。                             | 加 full config integration test。                                         |
| `opentad/cores/test_engine.py:gather_ddp_results logical segment`   | risk                             |  low-medium | `world_size=0` 默认值不安全，若非分布式路径误入会失败。                                                                                | `if world_size <= 1 or not dist.is_initialized(): return result_dict`。  |
| `opentad/evaluations/mAP.py:L3-L8`                                  | confirmed good / risk            |  low-medium | allowed-videos 同时过滤 GT/preds；unknown label 被映射到额外类，AP 均值只算 known classes。                                          | 对 unknown label policy 加测试和文档。                                          |
| `opentad/cores/train_engine.py:L0`                                  | confirmed bug                    | medium-high | 单卡非 DDP 访问 `model.module` 风险。                                                                                      | 使用 `target_model = model.module if hasattr(model,"module") else model`。 |
| `tools/remote/submit_siglip_ontad_n16r4.sh`                         | confirmed good / limitation      | info-medium | 强制单 GPU与 streaming-safe 单 rank一致；但只是提交脚本，不证明 full-60 合法。                                                           | 脚本前置 `python tools/check_config_contract.py CONFIG`。                    |
| `tests/test_siglip_ontad_runtime_smoke.py`                          | missing test                     |      medium | tiny no-future test 主要覆盖 encoder；tiny detector 用的是 `TemporalMaxerProj`，不是 P0 的 `CausalTemporalMaxerProj`。          | 改成加载 P0/P1 config 的 full chain no-future smoke。                         |
| `tests/test_videomae_matr_ontad_contracts.py`                       | confirmed good / claim blocker   |   info-high | 测试明确承认 VideoMAE 是 stub contract。                                                                                   | 接真实 VideoMAE 后增加 runtime tests。                                         |
| `tests/test_map_evaluator_contracts.py`                             | missing test                     |      medium | 只覆盖 allowed_videos；缺 streaming ledger → mAP 端到端测试。                                                                 | 用 synthetic ledger rows 跑 evaluator，断言无 global NMS。                     |

---

## Adaptive frame selection + AdaTAD implementation plan

### 1. Selector 输入

每个视频维护 per-video causal state。selector 在时刻 `t` 只能看到：

* 当前及过去 raw frames 的低成本特征：缩略图 RGB、frame diff、motion magnitude、blur/edge、scene-change score。
* 过去已选帧的 frozen SigLIP/VideoMAE token。
* 过去 detector 输出的 actionness、uncertainty、boundary risk，但只能来自已经 emitted 或已经 observed 的 prefix。
* 时间信息：absolute frame index、seconds、距上次选择的 gap、当前预算剩余、最大允许延迟。

禁止输入：

* 未来帧。
* 全视频 feature cache。
* raw prediction cache。
* GT segment。
* 离线 teacher prediction at test time。

---

### 2. Selector 输出与预算约束

输出一个 causal selection packet：

```python
SelectionPacket(
    selected_abs_frames: LongTensor[B, K],
    selected_mask: BoolTensor[B, K],
    keep_logits: Tensor[B, T_prefix],
    keep_probs: Tensor[B, T_prefix],
    budget_used: Tensor[B],
    max_gap: Tensor[B],
    decision_frame: Tensor[B],
)
```

硬约束：

* `sum(selected_mask) <= K_max`
* `max_gap <= G_max`
* `decision_frame - source_frame <= latency_max`
* per-video state bounded memory：最多保留 `M` 个 selected tokens。
* train/test 同 budget；不能 train dense、test sparse 而不声明。

---

### 3. Hard / soft / hybrid selection 与梯度路径

推荐 hybrid：

1. **Warm-up soft mask**：所有 cheap features 可见，visual encoder 可以只跑 dense teacher 或 frozen cached teacher；selector 学 actionness/boundary/coverage。
2. **Hard top-k / Bernoulli STE**：训练时用 Gumbel-TopK 或 straight-through Bernoulli，forward 只选 K 帧，backward 走 soft probs。
3. **Inference hard only**：只允许根据 prefix selector state 选择帧。

梯度路径：

```text
L_det -> detector -> selected tokens -> visual adapter
                         ↑
              STE / Gumbel selector logits
```

如果视觉 encoder frozen，梯度只到 selector、adapter、projection、head。若后期解冻 encoder，只解冻最后 N blocks，低 LR，并重新跑 no-future perturb test。

---

### 4. Selected frames 如何打包成 temporal tokens 并映射回 seconds

不要把 irregular frames 强行伪装成 uniform grid。每个 token 必须携带真实时间：

```python
token = {
    "feat": selected_feature,             # [B, K, C]
    "abs_frame": selected_abs_frames,     # [B, K]
    "time_sec": selected_abs_frames / fps,
    "delta_prev_sec": diff(time_sec),
    "observed_mask": selected_mask,
    "budget_id": route/budget metadata,
}
```

proposal decode 使用 seconds：

```python
center_t = token_time_sec[k]
start_sec = center_t - rel_left_sec
end_sec = min(center_t + rel_right_sec, current_observed_time_sec)
```

如果为了复用 AdaTAD dense head，需要一个 bridge：

* `selected_tokens -> irregular temporal token sequence`
* `token_time_sec -> point generator`
* `cell_width_sec -> local support width`
* `observed_mask / gap_mask -> head attention mask`

---

### 5. Detection head 如何消费 irregular/budgeted temporal tokens

三种可行路线，按推荐优先级：

**A. Native irregular head，最干净。**
实现 `IrregularMATRHead` 或改 AdaTAD point generator，使每个 point 的 center 是真实 `time_sec[k]`，duration regression 直接输出 seconds offset。

**B. Irregular-to-dense causal scatter bridge，工程风险较低。**
把 selected tokens scatter/interpolate 到 fixed causal grid，额外提供 `observed_mask` 和 `gap_distance`。缺点是会重新引入伪 dense time。

**C. AdaTAD adapter route。**
保留 AdaTAD detector，但输入不再是 uniform feature map，而是：

```python
features: [B, C, K]
masks: [B, K]
token_times_sec: [B, K]
cell_widths_sec: [B, K]
```

并修改 prior generator / decode / NMS，使 proposal seconds 来自真实 token time，而不是 `idx * stride / fps`。

---

### 6. 避免 selector collapse

必须同时有这些约束，否则 selector 很容易 collapse 到背景、静态清晰帧、长动作中段或 easy samples：

* **budget loss**：控制平均/硬预算。
* **entropy target**：防止过早全 0 / 全 1。
* **diversity / repulsion**：防止 K 帧挤在同一小段。
* **temporal coverage / max-gap loss**：保证持续可观测。
* **actionness distillation**：从 dense/fixed teacher 的 prefix-safe actionness 蒸馏。
* **boundary preservation loss**：对 start/end 附近的帧提高选择权重。
* **latency penalty**：动作结束后超过 `Δ` 秒才 emit 要惩罚。
* **background quota / hard negative mining**：防止只看动作区域而 FP 爆炸。
* **motion/static anti-collapse**：静态高置信背景不应被反复选择。

---

### 7. Training schedule

推荐最小可控流程：

1. **Stage 0：修复当前 P1。**
   先把 frozen SigLIP + MATR P1 full config 修到可训练、可评估、ledger 合法。

2. **Stage 1：dense/fixed teacher。**
   训练或固定一个严格 streaming-safe teacher，只用于训练蒸馏，不进 test path。

3. **Stage 2：selector warm-up。**
   冻结 detector/encoder，训练 selector 预测 teacher actionness/boundary + coverage/budget。

4. **Stage 3：selected-token detector training。**
   selector frozen 或半 frozen，训练 AdaTAD/MATR irregular head 消费 selected tokens。

5. **Stage 4：hybrid joint fine-tuning。**
   selector + detector 联合，Gumbel/STE hard selection，teacher distillation 降权。

6. **Stage 5：strict inference。**
   test 时禁用 teacher/cache/prediction cache，只保留 selector state + selected frames + detector。

---

## Minimal code sketch

```python
@dataclass
class SelectionPacket:
    selected_abs_frames: torch.LongTensor
    selected_mask: torch.BoolTensor
    keep_logits: torch.Tensor
    keep_probs: torch.Tensor
    budget_loss: torch.Tensor
    coverage_loss: torch.Tensor
    entropy_loss: torch.Tensor
    state: dict


class OnlineFrameSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim, budget_k, max_gap_frames):
        super().__init__()
        self.rnn = nn.GRUCell(in_dim, hidden_dim)
        self.keep_head = nn.Linear(hidden_dim, 1)
        self.budget_k = budget_k
        self.max_gap_frames = max_gap_frames

    def forward_step(self, cheap_feat_t, frame_idx_t, state, train=True):
        h = self.rnn(cheap_feat_t, state["h"])
        logit = self.keep_head(h).squeeze(-1)
        prob = torch.sigmoid(logit)

        if train:
            # straight-through Bernoulli
            hard = (prob > torch.rand_like(prob)).float()
            keep = hard.detach() - prob.detach() + prob
        else:
            keep = self._causal_budget_decision(prob, frame_idx_t, state)

        state["h"] = h
        state["budget_used"] = state["budget_used"] + keep.detach()
        state["last_selected_frame"] = torch.where(
            keep.bool(), frame_idx_t, state["last_selected_frame"]
        )
        return keep, logit, prob, state

    def budget_regularizer(self, keep_probs):
        expected = keep_probs.sum(dim=1)
        return torch.relu(expected - self.budget_k).pow(2).mean()
```

```python
def pack_selected_frames(frames, selected_abs_frames, fps, encoder):
    """
    frames: raw video access object or tensor prefix only
    selected_abs_frames: [B, K]
    """
    selected_rgb = gather_raw_frames(frames, selected_abs_frames)
    feats = encoder(selected_rgb)  # frozen or partially trainable

    token_times_sec = selected_abs_frames.float() / fps[:, None]
    delta_sec = torch.diff(
        token_times_sec,
        dim=1,
        prepend=token_times_sec[:, :1],
    )

    return {
        "feats": feats,
        "token_times_sec": token_times_sec,
        "delta_sec": delta_sec,
        "mask": selected_abs_frames >= 0,
    }
```

```python
class IrregularAdaTADHead(nn.Module):
    def forward(self, feats, token_times_sec, mask, current_time_sec):
        x = self.causal_temporal_block(feats, mask, token_times_sec)

        cls_logits = self.cls_head(x)
        left_sec = F.softplus(self.left_reg(x))
        right_sec = F.softplus(self.right_reg(x))

        center = token_times_sec
        start = center - left_sec.squeeze(-1)
        end = torch.minimum(
            center + right_sec.squeeze(-1),
            current_time_sec[:, None],
        )

        proposals = torch.stack([start, end], dim=-1)
        return cls_logits, proposals
```

```python
def assert_no_future_selector(selector, frames_prefix, frames_future):
    out_a = run_until_t(selector, frames_prefix, future=None)
    out_b = run_until_t(selector, frames_prefix, future=frames_future)
    assert_close(out_a.keep_logits, out_b.keep_logits)
    assert_close(out_a.selected_abs_frames, out_b.selected_abs_frames)
```

---

## Test plan

### Must-pass before continuing P1 full-60

1. **Config inheritance test**

   * Load P0/P1/P1-fix/P1-full.
   * Assert model/dataset/pipeline/inference/postproc keys exist.
   * Assert P1-full inherits raw-frame SigLIP route.

2. **Raw-frame index test**

   * Synthetic video with frame id encoded in pixels.
   * Assert `LoadFrames + LoadRawFrames` reads exact `[0,8,16,...]` for P0.

3. **Frame-grid-seconds roundtrip**

   * Test `window_start_frame`, `snippet_stride`, `fps`, `offset_frames`.
   * `grid -> seconds -> frame` must be exact or documented ±1.

4. **Whole-model no-future perturb**

   * Run P1 tiny model on sequence A.
   * Replace future frames after token `t`.
   * Assert logits/proposals/emission candidates up to `t` unchanged.

5. **Training target censor test**

   * Construct action segment ending outside current prefix.
   * Assert online mode does not regress future endpoint.

6. **Streaming mAP test**

   * Synthetic ledger rows → evaluator.
   * Assert no video-level NMS, no future rows, allowed-videos filters both GT/preds.

7. **Optimizer group test**

   * Frozen SigLIP params absent or lr=0.
   * Adapter/projection/head trainable.
   * If motion branch enabled, motion branch grads nonzero.

8. **Single-rank guard test**

   * streaming-safe world size >1 must fail.
   * world size 1 must pass.

9. **Emission latency test**

   * One proposal ending early in window.
   * Assert ledger latency reflects `window_end_frame - end_frame`.
   * This will expose true window-level latency.

### Must-pass before adaptive selection + AdaTAD claim

1. Selector no-future perturb.
2. Selector hard budget invariant.
3. Max-gap invariant.
4. Selected frame packing time-order invariant.
5. Irregular head proposal seconds decode test.
6. Selected-token detector overfit on 2 synthetic videos.
7. Dense teacher unavailable at test assertion.
8. No raw prediction cache assertion.
9. Ablation smoke:

   * uniform K
   * random K
   * motion-only selector
   * learned selector without boundary loss
   * learned selector with full losses

---

## Final recommendation

**Recommendation: HOLD for paper claims, CONTINUE for engineering after fixing blockers.**

Immediate priority order:

1. Fix `p1_pilot` / `p1_fix` / `p1_full_60` config inheritance.
2. Add config-load/build smoke tests for all causal TAD configs.
3. Add online-censored training target mode; otherwise strict online training claim is weak.
4. Add whole-model no-future perturb test for actual P1 chain.
5. Make latency contract honest: current route is **windowed streaming-safe**, not true low-latency continuous On-TAD.
6. Treat VideoMAE route as stub until real causal VideoMAE is wired.
7. Only after P1 fixed baseline is valid, implement adaptive selector + irregular AdaTAD head. Current branch is not yet adaptive selection, not yet AdaTAD, and not yet paper-ready On-TAD.
