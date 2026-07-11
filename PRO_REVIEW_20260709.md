我能直接审查的是公开 GitHub 分支 `codex/online-tad-clean-20260702`；我无法读取你机器上的 `E:\DeskTop\TAD\OpenTAD_OnlineTADClean_20260702` 本地未提交改动。因此，下面结论对**当前公开分支**有效；若本地有未 push 的关键 patch，需要重新审查。

总体裁决先放前面：**HOLD，不是 GO，也不是彻底 NO-GO。** 当前 repo 已经有一个较认真、可审计的 **raw-frame / selected-only / single-rank streaming-safe Online TAD validation skeleton**，但还不能支撑 CCF-A full paper 的主张。最危险的虚高 claim 是：**adaptive selected-frame Online TAD、paper-ready Online TAD、full online-censored training、learned selector、DDP-safe streaming evaluation**。项目自己的 README 和吸收文档也基本承认这一点：当前 SigLIP/SigLIP2 与 VideoMAE-adapter routes 只是 validation candidates；single-rank streaming-safe emission-ledger evaluation 之外不能宣称 DDP-auditable streaming eval、paper-ready Online TAD 或 visual-tower finetune；验证/测试也明确禁止 GT、teacher cache、raw prediction shortcut 和旧 cache。([GitHub][1])

---

## 阶段 1：任务定义审查

### 1. 当前任务到底是什么？

当前最准确的定义是：

> **windowed / raw-frame / selected-only / single-rank streaming-safe Online Temporal Action Detection skeleton**，而不是完整 paper-level On-TAD，也不是单纯 OAD，也不是 offline TAD。

它不是 OAD，因为输出目标不是仅 frame-level class 或是否发生动作；它有 start/end segment、score、label、emission frame、source frame、latency ledger、OnlineMAP wrapper。它也不是严格意义上的 full continuous On-TAL/On-TAD，因为当前 README 与配置都把它定位为 validation candidate，并承认 streaming-safe emission-ledger evaluation 目前是 single-rank only，DDP 需要 video-contiguous sampler 或 centralized online state。([GitHub][2])

最安全命名应是：

```text
Selected-only causal-stride raw-frame Online TAD validation skeleton
with single-rank emission-ledger evaluation.
```

不应命名为：

```text
Budgeted Adaptive Online TAD
Learned adaptive selected-frame On-TAD
Paper-ready streaming Online TAL
```

### 2. 时刻 t 允许访问什么，禁止访问什么？

允许访问：当前及过去 raw frames / selected frames、当前窗口内不超过时刻 t 的 tokens、causal projection state、过去 emission state、过去已输出 proposal、当前 prefix metadata。`RESEARCH_DISCUSSION_PROMPT_20260707.md` 也把任务定义为：时刻 t 只可用当前/过去 frame、token、memory 和 prediction state，输出 class/start/end/confidence/emission time，并评估 no-future、ledger、latency、state、compute budget。([GitHub][3])

禁止访问：future frames、full-video feature cache、raw prediction cache、future-label shortcut、teacher cache、GT 决策、offline video-level NMS。RTK 与 README 都明确验证/测试不能使用 GT、teacher cache、raw prediction shortcut 或 hidden cache；`inference.load_from_raw_predictions=False` 也在 P0 config 中显式关闭。([GitHub][4])

### 3. 输出是什么？

当前输出不是单一 frame class。它实际输出：

```text
segment = [start_sec, end_sec]
label
score
emit_frame
source_grid / source_frame
start_frame / end_frame
latency_sec
stream_key / stream_id / frame_policy
```

`SingleStageDetector` 在 streaming-safe postprocess 中把 candidates 送入 `OnlineEmitter`，再写出 `emit_frame`、`source_grid`、`source_frame`、`start_frame`、`end_frame`、`latency_sec` 等 ledger 字段。([GitHub][5]) `OnlineMAP` 则把 emission ledger 转成 online detection dataframe，并统计 latency、future-end/source violations 等。([GitHub][6])

### 4. 当前 repo 的安全 claim 边界

可安全写：

```text
We implement a selected-only raw-frame causal-stride Online TAD skeleton,
with strict raw-prediction-cache disabled, single-rank emission-ledger auditing,
and an OnlineMAP wrapper for emitted-row evaluation.
```

不可安全写：

```text
We solve adaptive Online TAD.
We propose a learned adaptive selector.
We achieve paper-ready online-censored training.
We provide DDP-safe streaming Online TAD evaluation.
We prove full no-future correctness only from ledger metrics.
```

吸收文档已经列出允许 claim：selected-only causal-stride skeleton、ledger-aware emitted-row mAP wrapper、irregular dense-grid metadata path under validation、single-rank streaming-safe emission ledger candidate；不允许 claim：paper-ready Online TAD、true adaptive selected-frame、learned budgeted selector、paper-level OnlineMAP、full online-censored training。([GitHub][7])

### 5. 目前使用过度的术语

**adaptive**：final config 里 `frame_policy="adaptive_selected_frames"`，但实际 selector 是 `policy="causal_stride", keep_ratio=0.5, stride=2`。这是固定策略，不是 adaptive。([GitHub][8])

**final**：文件名 `thumos_siglip2_adaptive_matr_ontad_final.py` 与 comment “Final target-code route” 过强，虽然注释也写明不是 paper-ready，但标题仍会误导审稿人。([GitHub][8])

**online-censored**：P1 config 继承设置 `online_censored_training=True`，但代码只对 regression end endpoint 进行权重屏蔽；classification/actionness 等仍有 full-segment target 风险，不能叫 full online-censored training。([GitHub][9])

**causal**：P0 config 中 `strict_causal=True`，这很好；但 `CausalTemporalMaxerProj` 类本身默认 `strict_causal=False`，非 strict 时存在 forward/backward 双向路径，因此 causal claim 必须绑定到具体 config 与 tests。([GitHub][10])

**end-to-end**：当前 SigLIP2 visual tower 是 frozen，P2 只训练 motion_branch / adapter / projection / neck / rpn_head，不是完整视觉塔端到端训练。([GitHub][11])

### 6. 最容易被 CCF-A 审稿人质疑的任务定义

最容易被打穿的是：

```text
“Adaptive Online TAD from raw frames”
```

因为当前 final route 同时存在三层不稳：selector 是 causal stride 固定采样；online-censored training 不完整；single-rank ledger 只能审计输出行，不能单独证明整个 pipeline 无 future source。([GitHub][7])

---

## 阶段 2：当前实现事实审查

### 1. OnlineSigLIPFrameEncoder 是否真正只编码 selected frames？

**结论：配置级别是 yes，类级别不是无条件 yes。**

`OnlineSigLIPFrameEncoder` 支持 `encode_policy="dense"` 或 `"selected_only"`；只有在 `selected_only` 且存在 `frame_selector` 时才先选帧再编码。代码在 selected-only path 中调用 `_select_frames_before_vision`，然后只对 `selected.frames` reshape 后送入 `_encode_pixels_chunked`，再 scatter 成 selected token features。([GitHub][12])

P0 hardening test 也有 runtime test：8 帧输入、stride-2 selector、selected-only encoder，fake encoder 实际只看到 4 帧。([GitHub][13])

但问题是：encoder metadata 里仍写 `frame_policy="adaptive_selected_frames"`，这在固定 stride 情况下是命名污染。([GitHub][12])

### 2. CausalFrameSelector 是固定策略、motion heuristic，还是 learned adaptive selector？

**结论：当前不是 learned adaptive selector。**

final config 明确是：

```python
frame_selector=dict(
    type="CausalFrameSelector",
    policy="causal_stride",
    keep_ratio=0.5,
    stride=2,
    max_gap=2,
    always_first=True,
)
```

这就是固定 causal stride。([GitHub][8]) P2 motion branch 是 trainable causal motion feature branch，但那不是 selector；P2 config 只说 trainable causal motion branch under validation。([GitHub][11]) 吸收文档也明确指出：`policy="causal_stride"` 是固定 selection，不是 content-adaptive / learned。([GitHub][7])

### 3. MATRHead 的 start/end/actionness/emit 分支是否构成新方法？

**结论：目前只能算在线 head 工程化，不足以单独构成机制级新方法。**

`MATRHead` 添加了 start_head、end_head、actionness_head、emit_head，并用 BCE loss 训练 start/end/actionness/emit 分支；forward_test 时可用 actionness/emit scores 调制 proposal score。([GitHub][14]) 这更像 ActionFormer/MATR-style head 的在线扩展，而不是已经证明的 emission hazard learning。要成为贡献，必须把 emit 定义成 bounded-delay hazard，配合在线-censored target、latency-sensitive loss、false-early penalty、missed-emission penalty，并做 latency/AP trade-off 实验。

### 4. online_censored_training 当前是否完整？

**结论：不完整。可以说“regression endpoint partially censored”，不能说 full online-censored training。**

积极事实：`AnchorFreeHead` 有 `_online_censored_regression_weights`，用 `target_segments[:,1] <= point_centers + online_censored_max_future_offset` 来决定 regression loss 权重；reg loss 会乘这个权重。([GitHub][15]) P1 config 也设置 `online_censored_training=True`，注释目标是“不在 endpoint observable 前 regress or emit future endpoint”。([GitHub][9])

但缺口很大：`MATRHead._build_online_branch_targets` 中 actionness 仍按 full segment interior 标 1，start/end/actionness/emit target 构建仍直接遍历完整 GT segment；只对 end/emit mask 做 endpoint observed 条件。([GitHub][14]) classification target 仍由 `prepare_targets(points, gt_segments, gt_labels)` 基于完整 GT 生成，至少从可见代码看没有 prefix-censored classification/actionness target。([GitHub][15])

### 5. irregular-time decoding 是否真的原生处理 sparse selected tokens？

**结论：只是受限的 irregular metadata path，不是 paper-level native irregular sparse decoding。**

`MATRHead` 的 irregular path 对 batch 有强限制：如果 selected positions 在 batch 内不一致，会 fail-close，当前 adaptive irregular MATRHead 要求 batch_size=1 或 identical selected positions。([GitHub][14]) `_build_irregular_points` 把 selected-axis coordinates 映射回 dense-grid points，返回 `"dense_grid"` proposal axis；这不是完整 per-sample sparse temporal geometry head。([GitHub][14])

另一个风险：`irregular_time_decode.py` 用 `duration` 或 `times[-1]` 作为右端 sentinel；若 duration 是整视频 duration 而当前只有 prefix/window，可在 decode 上引入未来长度语义。([GitHub][16])

### 6. OnlineEmitter / OnlineMAP 是否足以证明 no-future correctness？

**结论：不足以单独证明，只能作为 emitted-row audit。**

`OnlineEmitter` 会跳过 `source_frame > now_frame`、`end_frame > now_frame`、超过 latency 的 candidate，并写出 latency。([GitHub][17]) `OnlineMAP` 会要求 ledger，按 allowed video / latency 过滤后统计 num emissions、future_end/source violations，并转成 mAP dataframe。([GitHub][6])

但这只能证明**输出行**没有显式 future end/source 字段，不证明 candidate generation、feature extraction、proposal scoring、NMS 排序没有未来信息。尤其是如果上游用了 full-video feature、teacher logits、raw-prediction cache 或 offline NMS，ledger 仍可能看起来干净。

还有一个具体 P0 风险：`OnlineEmitter.step` 的 `state.last_emit_frame = now_frame` 只在 `if emitted:` 分支后更新；如果某个 step 没有 emission，后续更小的 now_frame 可能不会被 `now_frame < state.last_emit_frame` 捕获。当前 test 只检查有 emission 时 `state.last_emit_frame == 10`，没有覆盖 no-emission monotonic state update。([GitHub][18])

### 7. DDP、多视频、多窗口 state 是否存在协议风险？

**结论：存在，而且当前文档承认 single-rank only。**

README / configs README 明确写：streaming-safe ledger 目前 single-rank only，因为 DDP 会 split per-video state；formal DDP 需要 video-contiguous sampler 或 centralized online state。([GitHub][2]) `SingleStageDetector` 用 `self._online_states` dict 管理 online state，按 stream_key 取 state；这对 single-process 可以工作，但不是 DDP global state 证明。([GitHub][5])

### 8. 当前 final config 的名字和注释是否夸大？

**结论：是，尽管注释中有防守性 disclaimer。**

`thumos_siglip2_adaptive_matr_ontad_final.py` 写着 “Final target-code route: Budgeted Adaptive Online TAD with causal emission”，`frame_policy="adaptive_selected_frames"`，但 selector 是 fixed `causal_stride`。([GitHub][8]) 这会被审稿人理解成 claim inflation。更合适的文件名：

```text
thumos_siglip2_causal_stride_matr_ontad_selectedonly_p3.py
```

---

## 阶段 3：问题澄清清单

在提出最终方法前，必须确认这些问题：

1. 首要目标排序是什么：offline mAP、OnlineAP、latency、compute budget、strict no-future correctness，哪一个是硬优先？
2. 主数据集只做 THUMOS14，还是必须扩展 ActivityNet / Ego4D / EPIC-Kitchens？
3. 主输入形态是什么：pre-extracted features、raw frames、frozen SigLIP/SigLIP2、VideoMAE adapter、还是 trainable visual tower？
4. 是否允许 offline teacher 仅训练期蒸馏？
5. 测试期是否绝对禁止 teacher、prediction cache、future context、full-video NMS？
6. compute budget 用 encoded raw frames、tokens、FLOPs、FPS 还是 wall-clock memory 作为主预算？
7. 预算是单卡 smoke、3-seed full run，还是完整 ablation grid？
8. 论文定位是 technical report、workshop、CCF-A full paper，还是开源 online TAD baseline？
9. 是否接受主贡献是 online supervision / emission protocol，而不是 backbone 或 selector？
10. adaptive selector 是主贡献，还是只作为 supporting module？
11. 如果 learned selector 失败，是否允许降级为 strong online protocol baseline？
12. latency 是 bounded delay 检测，还是越早越好但允许更新？
13. 输出是否允许 revision？当前协议看起来不应允许任意修订已发 emission。
14. OnlineAP 是只评 emitted rows，还是要按每个 prefix 做 streaming AP？
15. 是否需要严格 per-class latency/AP breakdown？

---

## 阶段 4：创新性审查

### 1. 哪些只是已有模块组合？

当前路线的基础模块组合包括：frozen SigLIP/SigLIP2 raw-frame encoder、causal projection、FPN-like neck、ActionFormer/AnchorFree-style regression head、MATR-style start/end/actionness/emit branches、fixed stride selector、emission ledger、OnlineMAP wrapper。P0 config 明确是 frozen SigLIP2 recent-frame baseline，MATR streaming memory disabled，strict causal projection，raw-prediction disabled。([GitHub][10])

### 2. 哪些是工程 hardening，不是论文贡献？

P0 hardening 包括：all-invalid selector fail-fast、selected-only encoder runtime assert、OnlineMAP filtering-before-stats、irregular metadata B=1 fail-fast。这些是必要工程质量，但不是主创新。([GitHub][13])

### 3. 哪些可能成为机制级创新？

最可能成为机制级创新的是：

```text
online-censored boundary / endpoint / emission hazard supervision
```

也就是从训练目标层面改变 TAD：在 prefix t 下，只监督当前可观察事实；end regression、emit hazard、boundary evidence、actionness 都严格 obey observed-prefix semantics。这能正面解决 online TAD 的核心科学问题：不能在动作结束前训练模型预测完整未来 endpoint。

第二可能是：

```text
budgeted causal evidence selection + native irregular-time decoding
```

但当前 fixed stride 距离 learned adaptive selection 还远，风险更高。

### 4. 最接近已有工作的 novelty gap

当前最接近的是 CausalTAD / causal projection + anchor-free TAD head + online emission wrapper。configs README 记录的是 CausalTAD 论文结果与 claims，而本 repo 的 raw-frame SigLIP2/VideoMAE routes 被明确标为 experimental candidates。([GitHub][2]) 若没有新监督机制或强实验证据，你的方法会被审稿人归类为：把已有 causal TAD baseline 接上 frozen visual encoder 与 ledger。

### 5. 审稿人会怎么写 Weakness？

典型 Weakness 会是：

1. “Adaptive” claim is unsupported; the selector is a fixed causal stride.
2. The proposed online-censored training is incomplete; classification/actionness still appear to use full GT segment labels.
3. OnlineMAP only audits emitted rows, not the entire computation graph.
4. The irregular-time decoder is batch-size-1 / shared-grid constrained.
5. The method is evaluated only on THUMOS14, risking dataset narrowness.
6. Many contributions are engineering safeguards rather than algorithmic novelty.
7. No full mAP / latency / compute / multi-seed evidence is shown.
8. DDP and multi-video state are explicitly not solved.

### 6. 如果没有显著结果，哪些 claim 必须删除？

必须删除：

```text
adaptive selected-frame Online TAD
learned selector
paper-ready Online TAD
end-to-end visual online TAD
full online-censored training
native irregular sparse TAD head
DDP-safe streaming evaluation
SOTA or competitive mAP claim
compute-preserving boundary quality
```

### 7. 是否存在最小但强的主贡献？

**存在，但当前代码还没有完成。**

最小强贡献应是：

> A strict online-censored localization and emission-hazard training protocol for windowed Online TAD, with auditable no-future emission ledger and bounded-delay metrics.

这个贡献比“adaptive selector”更小、更清楚、更可测，也更符合当前代码已有基础。

---

## 阶段 5：Claim Map

| Claim                                                                            | 当前代码支持情况                                                                                                                     | 需要新增机制                                                                                             | 必须实验                                                | 成功证据                                                                      | 失败判据                                                | 可写入论文等级                                                  |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------- |
| Strict streaming-safe online TAD protocol                                        | 部分支持：raw prediction disabled、strict causal config、emission ledger、OnlineMAP；但 DDP 不支持，ledger 不能证明全图 no-future。([GitHub][10]) | no-emission monotonic state test、future-perturbation tests、DDP 禁用 fail-fast、candidate-source trace | P0 protocol gate + perturb tests                    | future_end/source violations=0；future perturb 不改变 prefix outputs；no cache | 任一 future perturb 改变 prefix prediction；DDP state 混乱 | Workshop / strong baseline，可作为 CCF-A supporting protocol |
| Online-censored localization reduces future endpoint leakage                     | 目前只 partial：regression end 有 endpoint-observed weight；actionness/cls 不完整。([GitHub][15])                                      | prefix-censored cls/actionness/start/end/emit target builder                                       | full GT endpoint vs censored target ablation        | late/early emission 更合理；future-label leakage probe 降低；OnlineAP/latency 不崩 | mAP 大跌且 latency 无改善；leakage probe 不变                | 可成为主贡献，但需补机制                                             |
| Causal boundary / emission hazard improves bounded-delay detection               | 有 emit branch，但只是 BCE 分支；hazard 定义不充分。([GitHub][14])                                                                         | hazard loss、false-early penalty、missed-end penalty、bounded-delay target                            | with/without emit branch, latency curve             | OnlineAP ↑或同等，p90 latency ↓，false early 不升                                | emit branch 只调 score，early false positives 上升       | 可成为主贡献核心                                                 |
| Budgeted frame/token selection reduces compute while preserving boundary quality | selected-only encoding 在 final config 支持；但 selector fixed stride，不是 adaptive。([GitHub][12])                                  | learned/motion/budget policy，selector collapse guard，budget telemetry                              | dense vs uniform/random/motion/learned budget curve | encoded frames/FLOPs ↓，mAP/IoU@0.7 接近 dense                               | mAP 低于 uniform/random，selector collapse             | 只能作为 supporting claim，当前不能主打                             |
| Irregular-time decoding is necessary for sparse selected tokens                  | 有 irregular decode helper 与 MATR metadata path；但 batch-size / shared-axis 限制强。([GitHub][14])                                 | per-sample native sparse temporal grid，window-bounded sentinel                                     | with/without irregular decode                       | sparse budget 下 high-IoU AP 明显提升                                          | 与 uniform-axis 无差别或出错                               | supporting claim                                         |
| Motion branch improves online boundary evidence                                  | P2 有 trainable causal motion branch，但未证明。([GitHub][11])                                                                      | branch-specific boundary target / diagnostic                                                       | with/without motion branch                          | boundary recall、OnlineAP、latency 有稳定提升                                    | 无提升或过拟合                                             | P2 supporting only                                       |
| Offline-to-online distillation helps online student without inference leakage    | 当前未见完整实现                                                                                                                     | train-only teacher logits / soft endpoints / no inference teacher audit                            | distill vs no-distill + leakage audit               | OnlineAP 提升，测试期无 teacher/cache                                            | teacher leakage 或无提升                                | 可选 supporting，风险高                                        |

---

## 阶段 6：候选方法路线

### A. Online-censored boundary / emission hazard learning

**Thesis**：把 Online TAD 的核心从“离线 segment 标签训练 online 模型”改成“prefix-observed supervision”：只有当前 prefix 下可观察的事实才进入 localization / actionness / boundary / emission loss。

**新机制**：统一 `PrefixObservedTargetBuilder`，输出 `observed_actionness`, `start_visible`, `end_visible`, `emit_hazard`, `regression_weight`, `false_early_mask`, `late_emit_mask`。当前代码已有 partial regression endpoint weighting，但需要扩展到 classification/actionness/branch targets。([GitHub][15])

**解决 bottleneck**：训练期 future endpoint leakage。

**为什么不是换名**：它改变 target semantics，而不是换 backbone/head。

**需改文件**：

```text
opentad/models/dense_heads/anchor_free_head.py
opentad/models/dense_heads/matr_head.py
opentad/utils/online_protocol.py
opentad/evaluations/online_map.py
configs/causaltad/thumos_siglip2_matr_ontad_p1.py
tests/test_p0_review_hardening.py
tests/test_final_online_tad_contracts.py
```

**最小实验闭环**：full-GT training vs online-censored target；with/without emit hazard；same backbone, same selector, same postprocess。

**最大风险**：offline mAP 下降，审稿人认为只是 label masking。

**回滚 claim**：从“method”降级为“protocol-correct online supervision baseline”。

---

### B. Budgeted causal evidence selection + irregular-time decoding

**Thesis**：在 fixed compute budget 下，只编码 causal-selected evidence，并用 native irregular temporal geometry decode segment boundaries。

**新机制**：learned causal utility selector + per-sample sparse temporal grid + irregular decode loss/postprocess。

**解决 bottleneck**：raw-frame compute 与 sparse token geometry mismatch。

**为什么不是换名**：selector 必须是 content-dependent、learned、test-time teacher-free；decode 必须 per-sample native sparse，而不是 batch-shared metadata path。

**需改文件**：

```text
opentad/models/selectors/causal_frame_selector.py
opentad/models/backbones/online_siglip_adapter.py
opentad/models/dense_heads/matr_head.py
opentad/models/utils/irregular_time_decode.py
opentad/utils/online_protocol.py
configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py
```

**最小实验闭环**：dense upper / uniform causal stride / random / motion heuristic / learned selector at same encoded-frame budget。

**最大风险**：learned selector 不如 uniform stride；irregular decode 引入 geometry bugs。

**回滚 claim**：保留 selected-only compute skeleton，不宣称 adaptive selector。

---

### C. Offline teacher to online student distillation，仅训练期使用

**Thesis**：允许 offline teacher 只在训练期提供 soft targets，让 online student 学会 prefix-compatible evidence，但测试期严格 teacher-free。

**新机制**：teacher logits / soft segment quality / soft boundary hints 转成 prefix-censored distillation target；测试期 audit 禁止 teacher/cache。

**解决 bottleneck**：online prefix label 稀疏、hard label 噪声大。

**为什么不是换名**：关键是 distillation target 必须 prefix-censored，不能直接复制 offline future endpoint。

**需改文件**：

```text
tools 或 datasets 中 teacher export / load
opentad/models/dense_heads/matr_head.py
opentad/utils/online_protocol.py
configs/causaltad/*
tests/test_*_teacher_leakage.py
```

**最小实验闭环**：no-distill vs train-only distill；测试期 teacher/cache absence audit。

**最大风险**：被认为 teacher leakage 或离线模型变相 online。

**回滚 claim**：删除 teacher 主张，只作为 appendix ablation。

---

### D. Causal motion / boundary evidence branch，轻量辅助

**Thesis**：motion difference 是 online boundary evidence，可作为 frozen visual feature 的 causal补充。

**新机制**：causal motion branch + boundary auxiliary loss + boundary recall diagnostics。

**解决 bottleneck**：frozen SigLIP2 per-frame appearance 对动作边界不敏感。

**为什么不是换名**：必须证明 motion branch 提升 boundary recall / latency，而不仅是多一个 conv。

**需改文件**：

```text
opentad/models/backbones/online_siglip_adapter.py
opentad/models/dense_heads/matr_head.py
configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py
```

**最小实验闭环**：P1 no-motion vs P2 motion, same seed / same protocol / same budget。

**最大风险**：提升很小，审稿人认为 trivial。

**回滚 claim**：只作为 implementation ablation，不放 contribution list。

### 路线评分

| 路线                                                              | Problem fidelity | Method specificity | Contribution quality | Feasibility | Validation clarity | CCF-A readiness | Overclaim risk |
| --------------------------------------------------------------- | ---------------: | -----------------: | -------------------: | ----------: | -----------------: | --------------: | -------------: |
| A. Online-censored boundary / emission hazard                   |                9 |                  8 |                    8 |           6 |                  8 |               7 |              4 |
| B. Budgeted causal evidence selection + irregular-time decoding |                8 |                  7 |                    8 |           5 |                  7 |               6 |              7 |
| C. Offline teacher-to-online student distillation               |                7 |                  7 |                    6 |           6 |                  6 |               5 |              8 |
| D. Causal motion / boundary evidence branch                     |                6 |                  6 |                    4 |           8 |                  7 |               4 |              5 |

---

## 阶段 7：推荐主路线

我只推荐一个主路线：

```text
A. Online-censored boundary / emission hazard learning
```

### 1. 为什么它最可能形成 CCF-A 可辩护贡献？

因为它直接攻击 Online TAD 的核心定义漏洞：训练时不能用未来 endpoint 当作当前 prefix 的确定监督。当前代码已经有 partial regression endpoint censoring，可以自然扩展；审稿人也更容易接受“监督协议创新 + emission hazard + no-future evidence”作为清晰贡献，而不是把固定 stride 包装成 adaptive selector。([GitHub][15])

### 2. 为什么它比其他路线更小、更清楚、更可测？

它不依赖 learned selector 是否超过 uniform，不依赖 VideoMAE stub，不依赖 DDP，不需要引入 RL/LLM。对照实验清楚：full-GT training vs online-censored training；with/without emit hazard；with/without boundary target。

### 3. dominant contribution

```text
A prefix-observed online localization and emission-hazard supervision protocol
for windowed Online TAD.
```

### 4. optional supporting contribution

```text
Selected-only causal-stride compute skeleton + emitted-row OnlineMAP audit.
```

### 5. explicit non-contributions

```text
Not a new visual foundation model.
Not a learned adaptive selector yet.
Not DDP-safe streaming evaluation.
Not full continuous online deployment.
Not SOTA offline TAD.
Not teacher-at-inference.
```

### 6. 不应该加入的诱人模块

不要现在加入 VLM/LLM/RL controller、VideoMAE stub 主张、teacher-at-test、complex memory revision、learned selector 主贡献、multi-dataset sprawl。先把 online-censored supervision 做闭环，否则 contribution 会散。

---

## 阶段 8：实验闭环

### P0 protocol gate

必须先跑并通过：

```text
no-future row audit
future-frame perturbation test
raw prediction disabled
teacher/cache disabled
inference.load_from_raw_predictions=False
source_frame <= emit_frame
end_frame <= emit_frame
single-rank streaming state
no-emission monotonic state update
latency semantics test
selected-only encoded-frame count test
DDP fail-fast for streaming eval
```

现有 tests 覆盖了 selected-only encoded count、all-invalid selector fail-fast、OnlineMAP latency filtering、emitter latency semantics，但缺 no-emission monotonic state case。([GitHub][13])

### P1 current baseline

当前 baseline 应命名为：

```text
P1 selected-only causal-stride SigLIP2-MATR Online TAD skeleton
```

不要叫 adaptive。跑：dense frozen encoder upper bound、causal stride 50%、random 50%、recent-window baseline。

### P2 main mechanism

加入：

```text
PrefixObservedTargetBuilder
online-censored cls/actionness/reg/end/emit target
hazard-style emission target
false-early penalty
late/missed-emission penalty
```

### P3 supporting module

只有 P2 通过后再加 motion branch或 irregular decode。P2 失败前不要加 learned selector，否则无法归因。

### P4 full ablation

拆：

```text
full GT endpoint training
online-censored regression only
online-censored actionness only
emit hazard only
boundary only
all online-censored targets
with/without actionness score modulation
with/without emit score modulation
```

### P5 stress tests

必须包含：

```text
long background
overlapping actions
late emission
sparse budget
window boundary
very short action
very long action
dense action burst
action starts before window
action ends after window
```

### P6 multi-seed

至少 3 seeds，报告 mean ± std。只给单 seed 不足以支撑 CCF-A。

### P7 failure analysis

按类别统计：

```text
false early emission
missed endpoint
late emission
boundary drift
source future rejection
end future rejection
selector collapse
background false positives
overlapping action confusion
window-boundary truncation error
```

---

## 阶段 9：Baseline / Ablation 要求

必须至少包含：

1. Offline / feature-level CausalTAD baseline。
2. Fixed uniform causal stride。
3. Recent-window online baseline。
4. Motion heuristic selector。
5. Dense frozen encoder upper bound。
6. Random selected-frame budget baseline。
7. MATR-style head without proposed mechanism。
8. Full GT endpoint training vs online-censored training。
9. With / without boundary branch。
10. With / without emit branch。
11. With / without irregular-time decode。
12. With / without motion branch。
13. Teacher distillation vs no distillation，如启用 teacher。
14. Same budget, same postprocess, same visual encoder 的公平对照。
15. Raw-prediction-cache forbidden negative control。
16. Future-frame perturbation negative control。
17. Offline video-level NMS negative control。
18. DDP disabled/fail-fast protocol test。

---

## 阶段 10：指标

必须报告：

1. mAP at standard IoU thresholds。
2. Online AP / emitted-row AP。
3. emission latency mean / p50 / p90 / p95。
4. no-future violations。
5. future source violations。
6. false early emission rate。
7. boundary recall after endpoint observed。
8. compute budget：encoded frames、FLOPs、FPS、GPU memory。
9. selector collapse rate。
10. per-class AP。
11. robustness under long background。
12. wall-clock training time and inference throughput。
13. high-IoU AP，尤其 @0.7。
14. emitted proposal count per video。
15. rejected candidate count：future-end、future-source、latency timeout。
16. prefix perturbation consistency rate。

---

## 阶段 11：审稿风险

### 1. Novelty weak risk

“该方法主要是 frozen visual encoder + causal projection + MATR-like head + ledger，缺少明确算法创新。”

### 2. Protocol leakage risk

“OnlineMAP 只审计 emission rows，不能证明 backbone/projection/head/postprocess 没有访问 future context。”

### 3. Label leakage risk

“online_censored_training 只部分屏蔽 endpoint regression；classification/actionness 仍可能用完整 GT segment。”

### 4. Baseline unfairness risk

“缺少同预算 random/uniform/recent-window/motion/dense upper bound，无法归因。”

### 5. Ablation insufficiency risk

“emit/actionness/boundary/motion/irregular decode 混在一起，不能证明哪个模块有效。”

### 6. Metric mismatch risk

“offline mAP 不能证明 online latency；OnlineAP 也不能替代 no-future correctness。”

### 7. Overclaiming risk

“adaptive/final/paper-ready 命名与代码实际 fixed causal stride 不一致。”

### 8. Engineering-only contribution risk

“P0 hardening 是必要工程质量，但不是 CCF-A 级主贡献。”

### 9. Dataset narrowness risk

“只在 THUMOS14 上验证，Online TAD 泛化不足。”

### 10. Reproducibility risk

“formal_training_ready=False，single-rank only，DDP state 未解决，full run/multi-seed evidence 缺失。”

---

## 阶段 12：最终输出

### 1. 当前 paper-worthiness

```text
HOLD
```

不是 GO：因为 adaptive selector、full online-censored training、paper-level OnlineMAP、DDP-safe streaming eval 都不成立。

不是 NO-GO：因为 raw-frame selected-only skeleton、strict causal config、emission ledger、OnlineMAP wrapper、P0 hardening tests 已有基础，可以继续发展为强 protocol paper。([GitHub][13])

### 2. 最需要用户回答的 10 个问题

1. 主目标是 OnlineAP/latency 还是 offline mAP？
2. 是否接受主贡献是 online-censored supervision，而不是 adaptive selector？
3. 测试期是否绝对禁止 teacher/cache/future/offline NMS？
4. 是否只做 THUMOS14，还是要扩展第二数据集？
5. 预算单位是 encoded frames、FLOPs 还是 wall-clock？
6. 是否必须 raw-frame，还是 feature-level online baseline 可作为主线？
7. 是否允许 full-GT training 作为负控？
8. 是否要支持 DDP streaming eval，还是论文明确 single-rank audit？
9. 是否允许 fixed causal stride 作为主系统，而 adaptive selector 只做 future work？
10. 失败时是否接受降级为 “strict online protocol baseline”？

### 3. 最推荐的主贡献方向

```text
Prefix-observed online-censored localization + bounded-delay emission hazard learning.
```

### 4. 最小实现闭环

```text
PrefixObservedTargetBuilder
-> censored cls/actionness/reg/start/end/emit targets
-> hazard loss + false-early/late penalties
-> emitted-row OnlineMAP + no-future ledger
-> full-GT vs censored training ablation
-> latency/AP/boundary drift analysis
```

### 5. 第一批必须跑的实验

```text
P0 protocol gate
P1 causal-stride selected-only baseline
P1 dense frozen upper bound
P1 random/uniform/recent-window baselines
P2 full-GT vs online-censored training
P2 with/without emit hazard
P2 with/without actionness censoring
P2 future-frame perturbation negative control
P2 3 seeds
```

### 6. 实验前绝对不能写的 claim

```text
adaptive selected-frame Online TAD
learned budgeted selector
paper-ready Online TAD
full online-censored training
DDP-safe streaming evaluation
native irregular sparse TAD head
compute reduction while preserving boundary quality
teacher-free distillation benefit
SOTA / competitive mAP
```

### 7. 如果结果失败，如何重构定位

如果 online-censored hazard 不能提升 OnlineAP/latency，论文不要硬写方法成功。应降级为：

```text
A rigorous negative-result and protocol baseline for streaming-safe Online TAD:
showing where offline TAD supervision leaks future endpoints,
which metrics expose the leakage,
and why fixed causal selected-only raw-frame baselines are insufficient.
```

这仍可能是 workshop / benchmark paper / strong open-source baseline，但不是 CCF-A full paper 的算法主贡献。

[1]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702 "GitHub - yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[2]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/configs/causaltad/README.md "raw.githubusercontent.com"
[3]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/RESEARCH_DISCUSSION_PROMPT_20260707.md "OpenTAD_OnlineTADClean_20260702/RESEARCH_DISCUSSION_PROMPT_20260707.md at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[4]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/RTK.md "raw.githubusercontent.com"
[5]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/detectors/single_stage.py "OpenTAD_OnlineTADClean_20260702/opentad/models/detectors/single_stage.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[6]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/evaluations/online_map.py "OpenTAD_OnlineTADClean_20260702/opentad/evaluations/online_map.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/PRO_REVIEW_ABSORPTION_20260708.md "OpenTAD_OnlineTADClean_20260702/PRO_REVIEW_ABSORPTION_20260708.md at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[8]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py "OpenTAD_OnlineTADClean_20260702/configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[9]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_matr_ontad_p1.py "OpenTAD_OnlineTADClean_20260702/configs/causaltad/thumos_siglip2_matr_ontad_p1.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_matr_ontad_p0.py "OpenTAD_OnlineTADClean_20260702/configs/causaltad/thumos_siglip2_matr_ontad_p0.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py "OpenTAD_OnlineTADClean_20260702/configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[12]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/backbones/online_siglip_adapter.py "OpenTAD_OnlineTADClean_20260702/opentad/models/backbones/online_siglip_adapter.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[13]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/tests/test_p0_review_hardening.py "OpenTAD_OnlineTADClean_20260702/tests/test_p0_review_hardening.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[14]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/dense_heads/matr_head.py "OpenTAD_OnlineTADClean_20260702/opentad/models/dense_heads/matr_head.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[15]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/dense_heads/anchor_free_head.py "OpenTAD_OnlineTADClean_20260702/opentad/models/dense_heads/anchor_free_head.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[16]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/utils/irregular_time_decode.py "OpenTAD_OnlineTADClean_20260702/opentad/models/utils/irregular_time_decode.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[17]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/utils/online_protocol.py "OpenTAD_OnlineTADClean_20260702/opentad/utils/online_protocol.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[18]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/opentad/utils/online_protocol.py "raw.githubusercontent.com"
