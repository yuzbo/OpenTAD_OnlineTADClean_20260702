## Verdict：HOLD

当前 `7caf2b9` **不是 FAIL**，因为它确实把几个关键骨架接上了：`CausalFrameSelector` 注册进 selector，`OnlineSigLIPFrameEncoder` 在 selected path 中用 `selected.frames.reshape(...)` 构造进入视觉塔的 batch，`MATRHead` 开始传递 `proposal_axis`，`SingleStageDetector` 能把 `metas` 传到 backbone/head，并且 `OnlineMAP` 已经要求 emission ledger 字段。目标 commit 可见，GitHub commit 页显示 `7caf2b9 implement adaptive online tad target route`，完整 SHA 为 `7caf2b972958328753d36f0f0079fce2893468c8`，并且本轮确实是 14 个文件变更。([GitHub][1])

但这版**不能放行远端真实训练、不能宣称论文级 Online TAD、也不能宣称真正 adaptive selected-frame online TAD 已完成**。最严重问题是：`OnlineEmitter` 的 latency gate 语义反了；`OnlineMAP` 仍然是普通 mAP 加 ledger side stats；adaptive batch>1 与 `MATRHead` 的 shared irregular axis 合同冲突；final config 本身仍写着 `formal_training_ready=False`，而且 selector policy 是 `causal_stride`，不是内容自适应策略。([GitHub][2])

我没有在容器里成功 `git clone`/跑 pytest，原因是容器 DNS 无法解析 GitHub；下面审查基于 GitHub 网页/raw 文件逐文件读取，不把未运行过的 build/forward/backward 当成已验证事实。

---

## 总体结论，对应你的 G 问题

1. **当前 commit 是否可以称为“端到端 raw-frame online TAD 训练实现”？**
   **不能。** 它是一个 raw-frame online TAD **目标骨架 / validation candidate**。训练路径把 `metas` 传入 rpn head，base config 的 `trainable_scope` 覆盖 motion branch、adapter、projection、neck、rpn_head；但 final config 明确 `formal_training_ready=False`，且没有真正 build/forward/backward 行为测试覆盖。([GitHub][3])

2. **当前 commit 是否可以称为“真正 adaptive selected-frame online TAD”？**
   **不能。** 它可以较谨慎地称为“selected-only causal stride selected-frame skeleton”。final config 的 `frame_selector.policy="causal_stride"`、`stride=2`、`max_tokens=384`，这是固定步长预算选择，不是内容自适应或 learned adaptive selection。([GitHub][2])

3. **当前 commit 是否可以称为“论文级 Online TAD metric/evaluator”？**
   **不能。** `OnlineMAP` 继承普通 `mAP`，只在导入 prediction 时附带 ledger 校验和 online side stats；AP 匹配本身没有把 emission time、latency budget、TP latency 绑定到 matching。([GitHub][4])

4. **当前 commit 是否可以开始远端真实训练？**
   **不建议。** 只允许本地 tiny/smoke、单卡、batch_size=1 的 shape/contract 验证。远端真实训练至少要先修 P0，尤其是 latency gate、batch>1 irregular axis、OnlineMAP 过滤一致性、selector all-invalid mask。

---

## A. selected-only 是否真实实现？

**局部成立，但防线不足。**

在 final config 下，`OnlineSigLIPFrameEncoder` 要求 `encode_policy="selected_only"` 且 frame selector 非空；forward 中先 `_select_frames_before_vision`，然后 selected path 使用 `selected.frames.reshape(...)` 构造 `pixels`，再调用 `_encode_pixels_chunked(pixels)`。这说明在该 path 中，未选帧不会进入 heavy vision encoder。([GitHub][5])

但是 `assert_selected_only` 目前只检查 `selected.frames.shape[0] > total_dense`，这个条件几乎永远不会触发，不能证明没有 dense encode 后 mask 的伪选择；而且 `_select_frames_before_vision` 只要 `frame_selector` 存在就会选择，即使 `encode_policy` 不是 selected_only，也没有严格 fail-closed。([GitHub][5])

---

## B. online / causal / no-future 是否成立？

**部分成立，但 evaluator 与 emission 逻辑有 P0 问题。**

`CausalMotionBranch` 的 delta 使用当前与过去帧，causal conv 左 padding 后再 depthwise conv，形式上不看未来。([GitHub][5]) `MATRHead._predict_levels` 通过 stream memory 拼接过去上下文，然后只返回当前窗口最后 `current_len` 的预测；从已读代码看，head 主体没有显式未来卷积。([GitHub][6])

但 `OnlineEmitter.step` 的 latency 逻辑把 `latency_frames` 当成“最小等待时间”而不是“最大允许延迟”；这会直接破坏 online AP/latency claim。([GitHub][7]) 同时 training target 中 `online_censored_training` 只抑制 end target，没有抑制 inside/actionness 的未来标签，严格 online 训练语义仍不干净。([GitHub][6])

---

## C. P4 irregular-time 是否正确？

**只完成了一个有价值但不完整的 half-step。**

`MATRHead._build_irregular_points` 会从 `irregular_selected_positions` 把 selected-axis point 映射回 dense-grid center，并返回 `proposal_axis="dense_grid"`，`SingleStageDetector.post_processing` 在 non-streaming path 中看到 dense_grid 后设置 `irregular_native_axis=True`，这确实是在避免 double irregular conversion。([GitHub][6])

但这套实现有三个风险：第一，`MATRHead._extract_shared_irregular_axis` 明确要求 batch 内所有样本 selected positions 完全一致，否则报错；这和 adaptive selection 的 batch>1 训练天然冲突。([GitHub][6]) 第二，non-streaming NMS 仍然发生在 `convert_to_seconds` 前；如果 fallback 到 selected_axis/token_times path，NMS 坐标轴会错。([GitHub][3]) 第三，`decode_irregular_segments_to_seconds` 使用整段视频 `duration` 作为右端 sentinel，滑窗内最后一个 token 后的 segment 可能被错误拉伸到整段视频结尾。([GitHub][8])

---

## D. OnlineMAP 是否论文级？

**不是。**

`OnlineMAP` 的 docstring 也更准确地说它是 “mAP over online-emitted detections with ledger-aware validity stats”。代码先用 `compute_online_detection_metrics(data["results"])` 统计 ledger，再按普通 mAP 所需字段构造 dataframe，`evaluate()` 只是调用父类 `super().evaluate()` 并把 `online` stats 附加进去。([GitHub][4])

更严重的是，`compute_online_detection_metrics` 在 `_import_prediction` 里发生在 `allowed_videos` / `blocked_videos` / `max_latency_sec` 过滤之前，因此 online side stats 和实际参与 AP 的 prediction set 可能不一致。([GitHub][4])

---

## E. 训练是否能在线端到端跑？

**不能确认；从代码合同看仍是 HOLD。**

正面证据是：`SingleStageDetector.forward_train` 会把 `metas` 传入 backbone，再根据 rpn_head signature 把 `metas` 传入 `forward_train`；base P2 config 的 trainable scope 覆盖 motion branch、adapter、projection、neck、rpn_head。([GitHub][3])

阻塞证据是：final config 自己标注 `formal_training_ready=False`；selector 是无参数 hard selection；batch>1 adaptive irregular axis 会在 `MATRHead` 报错；当前 tests 主要是字符串/配置存在性检查，不是 runtime build/forward/backward 行为测试。([GitHub][2])

---

# Findings

## P0-1：`OnlineEmitter.step` 的 latency gate 语义反了

**位置**
`opentad/utils/online_protocol.py::OnlineEmitter.step`
可搜索片段：

```python
eligible_end_frame = int(now_frame) - self.latency_frames
...
if end_frame > eligible_end_frame:
    continue
```

代码还只在 `emitted` 非空时更新 `state.last_emit_frame`。([GitHub][7])

**问题是什么**
`latency_frames` 如果表示最大允许延迟，应当允许 `end_frame <= now_frame` 且 `now_frame - end_frame <= latency_frames`。当前代码变成：只有 `end_frame <= now_frame - latency_frames` 才能 emit。也就是说 latency budget 越大，越要求 detection 更旧；刚结束、低延迟、最在线的 detection 反而被过滤。

**为什么影响 online/TAD/mAP**
这会直接扭曲 emission ledger、latency_sec、online AP under latency constraint。若 `max_latency=5s`，正确语义是“最多延迟 5 秒”，当前实现接近“至少延迟 5 秒”。这会让模型看起来 conservative、低召回，并且 online latency 指标不可解释。

**最小修复**

```python
# before loop, after monotonic check
now_frame = int(now_frame)

# inside loop
start_frame = self.grid_spec.grid_to_frame(cand.start_grid)
end_frame = self.grid_spec.grid_to_frame(cand.end_grid)

# no future
if end_frame > now_frame:
    continue

if cand.source_frame is not None and int(cand.source_frame) > now_frame:
    continue

latency_frames = now_frame - end_frame
if self.latency_frames >= 0 and latency_frames > self.latency_frames:
    continue

latency_sec = latency_frames / float(self.grid_spec.fps)
```

同时把 stream clock 更新改成每次 step 都更新，不只在有 emission 时更新：

```python
state.last_emit_frame = max(state.last_emit_frame, now_frame)
```

**需要补的测试**

* `latency_frames=5`，候选 end 分别为 `now`、`now-3`、`now-6`：前两个通过，第三个过滤。
* 连续两次 step，第一次无 emission，第二次 `now_frame` 倒退：必须抛 `ProtocolViolation`。
* `source_frame > now_frame` 必须过滤并计入 audit 或至少可观测。

---

## P0-2：`MATRHead` 的 irregular axis 只支持 batch 内完全一致，和 adaptive selection 冲突

**位置**
`opentad/models/dense_heads/matr_head.py::_extract_shared_irregular_axis`
可搜索片段：

```python
"MATRHead irregular selected-axis currently requires identical selected positions per batch; "
"use batch_size=1 for adaptive online routes"
```

`CausalFrameSelector.select` 是逐 sample 生成 `positions_per_sample`，理论上 batch 内不同视频会有不同 selected positions。([GitHub][6])

**问题是什么**
当前 head 用 `metas[0]` 的 selected positions 生成整批 points，然后要求其它 sample 的 selected positions 完全一致。对 `causal_stride` 且所有 mask 长度相同，可能暂时不爆；一旦 causal_motion/adaptive、不同 padding、不同 valid length 或 batch>1，就会直接 `ValueError`。

**为什么影响 online/TAD/mAP**
真正 adaptive selector 的核心是不同视频/窗口选不同 token。现在 head 的训练合同要求 batch 内共享一套 irregular axis，这等于把 adaptive batch training 禁掉。若强行 batch>1，要么 crash，要么如果以后有人放松检查，会产生坐标错配，导致 GT assignment、decode、NMS、mAP 全部不可信。

**最小修复**

短期 fail-closed：

```python
if self.training and metas is not None and any("irregular_selected_positions" in m for m in metas):
    if len(metas) != 1:
        raise ValueError("adaptive irregular MATRHead currently requires batch_size=1")
```

并在 final config 显式设置或注释：

```python
batch_size = 1  # until per-sample irregular points are implemented
```

长期正确修复：`points` 改为 per-sample list，即 `points[batch_idx][level]`，loss target assignment 和 `get_refined_proposals` 都按 sample 独立 decode。不要用 batch-shared `torch.cat(points, dim=0)` 作为唯一坐标轴。

**需要补的测试**

* `B=2`、两个 sample selected positions 不同，当前应 fail-fast，不能 silent wrong。
* `B=2`、positions 相同，应 forward_train/forward_test 通过。
* per-sample implementation 完成后，构造两个 sample 不同 sparse axis，验证 decoded proposals 分别落在各自 dense-grid 坐标。

---

## P0-3：`OnlineMAP` 不是论文级 online AP，且 online stats 与 AP 过滤集合不一致

**位置**
`opentad/evaluations/online_map.py::compute_online_detection_metrics`
`opentad/evaluations/online_map.py::OnlineMAP._import_prediction`
可搜索片段：

```python
self.online_metric_dict = compute_online_detection_metrics(data["results"], ...)
...
if video_id in self.blocked_videos: continue
if self.allowed_videos is not None and video_id not in self.allowed_videos: continue
...
if not self._validate_ledger_row(result): continue
```

([GitHub][4])

**问题是什么**
Online stats 先对全量 `data["results"]` 计算，然后才对 `blocked_videos`、`allowed_videos`、`max_latency_sec` 做 AP 输入过滤。结果是日志里的 `num_emissions`、latency summary、no-future 统计可能和实际 AP 使用的 rows 不一致。

更重要的是，AP 仍然是父类普通 mAP。`latency_sec`、`emit_frame` 只是 dataframe 的额外列，并没有进入 TP/FP matching 逻辑。([GitHub][4])

**为什么影响 online/TAD/mAP**
论文级 Online TAD metric 必须回答：检测是否在允许 emission time 内产生，TP 的 latency 是多少，超预算 TP 是否算 FP/ignored，AP 是否在 latency constraint 下计算。当前只能说“ordinary mAP over emitted rows + ledger audit”。

**最小修复**

先做一致性修复：先过滤 rows，再统计 online metrics，再构造 AP dataframe。

```python
filtered_results = {}
for video_id, rows in data["results"].items():
    if video_id in self.blocked_videos:
        continue
    if self.allowed_videos is not None and video_id not in self.allowed_videos:
        continue
    kept = []
    for row in rows:
        if self._validate_ledger_row(row):
            kept.append(row)
    filtered_results[video_id] = kept

self.online_metric_dict = compute_online_detection_metrics(
    filtered_results, require_no_future=self.require_no_future
)
```

论文级修复：新增至少三类指标：

```text
ordinary_emitted_mAP
latency_filtered_mAP@{0s,1s,2s,5s}
TP_latency_mean/p50/p90 at each tIoU
online_AP_with_deadline
```

**需要补的测试**

* 一个 prediction 文件里含 allowed video 和 blocked video，确认 online `num_emissions` 与 AP row 数一致。
* 一个超 `max_latency_sec` 的高分 TP，普通 mAP 与 latency-filtered mAP 应不同。
* 缺 `source_frame` / `emit_frame` 等 ledger 字段时必须 fail-fast。

---

## P0-4：selector all-invalid mask 会伪造 frame 0 token

**位置**
`opentad/models/selectors/causal_frame_selector.py::_valid_indices`
可搜索片段：

```python
if valid.numel() == 0:
    return torch.zeros(1, device=device, dtype=torch.long)
```

([GitHub][9])

**问题是什么**
当某个 sample 的 mask 全 false，selector 不返回 empty selection，而是返回 index 0，并且 `dense_lengths.append(int(valid.numel()))` 会记录为 1。随后 `packed_frames.append(frames[batch_idx, 0])`，等于把无效 padded sample 的第 0 帧送进视觉塔。([GitHub][9])

**为什么影响 online/TAD/mAP**
这会在空窗口、padding-only batch、异常 dataloader 输出时制造伪视觉证据，可能污染 selected_positions、dense_lengths、token_times、source_frame 和后续 decode。online setting 下这类伪 token 还可能被 emission ledger 当成真实 source evidence。

**最小修复**

选一种 fail-closed 语义：

```python
if valid.numel() == 0:
    raise ValueError("CausalFrameSelector got all-invalid mask; dataset/window contract is broken")
```

或者支持真正 empty selection，但 downstream 必须处理：

```python
return torch.empty(0, device=device, dtype=torch.long)
```

如果支持 empty，`select()` 里不能 `torch.stack([])`，应返回 zero-slot tensor 或在 batch collate 前过滤。

**需要补的测试**

* `B=1,T=8`、mask 全 false，必须抛错或返回 empty，不得 encode frame 0。
* `B=2`，一个正常，一个 all-invalid，必须 fail-fast，不能半静默。

---

## P1-1：final config 叫 adaptive，但实际是 fixed causal stride

**位置**
`configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py`
可搜索片段：

```python
frame_selector=dict(
    type="CausalFrameSelector",
    policy="causal_stride",
    keep_ratio=0.5,
    stride=2,
    max_tokens=384,
)
```

([GitHub][2])

**问题是什么**
`causal_stride` 是固定规则。它不会根据 motion、uncertainty、emission risk、boundary likelihood 或 detector utility 改变 selected positions。`CausalFrameSelector` 虽然有 `causal_motion` policy，但 final config 没用它；而且 `causal_motion` 也只是 raw pixel diff `.mean().item()` 的 hard heuristic，不是 learned adaptive selector。([GitHub][9])

**为什么影响 online/TAD/mAP**
如果论文 claim 是 “adaptive selected-frame online TAD”，现在证据只能支持“selected-only fixed stride before heavy encoder”。这与 P3 “adaptive selector 真正选帧/选 token”目标不一致。

**最小修复**

短期改名或降 claim：

```python
route_stage = "P3P4-selected-only-stride-target"
frame_policy = "causal_stride_selected_only"
```

真正 P3 则需要：

```python
policy="causal_motion"
# or learned_causal_selector with budget/logits/temperature/exported ledger
```

并记录 selector stats：

```python
meta["selector_policy"] = self.frame_selector.policy
meta["selected_positions"] = ...
meta["encoded_frames"] = ...
meta["dense_frames"] = ...
```

**需要补的测试**

* 两段视频内容不同但长度相同，adaptive policy 的 selected positions 应不同。
* fixed stride policy 的 selected positions 应完全可预测，并在 claim hygiene 里禁止叫 adaptive。

---

## P1-2：`assert_selected_only` 防线太弱

**位置**
`opentad/models/backbones/online_siglip_adapter.py::_select_frames_before_vision`
可搜索片段：

```python
if self.assert_selected_only and selected.frames.shape[0] > total_dense:
    raise RuntimeError(...)
```

([GitHub][5])

**问题是什么**
这个 assert 只检查 selected frame 数是否超过 dense frame 数；它不能证明 heavy encoder 只编码 selected frames，也不能防止 selector 退化为全选，更不能防止未来有人绕过 selected path。

**为什么影响 online/TAD/mAP**
伪选帧最危险的形式是 dense 全帧进视觉塔，然后再 mask/ledger 伪装 selected-only。当前 tests 只查源码字符串 `"selected.frames.reshape"`，没有 monkeypatch `_encode_pixels_chunked` 计数，因此不能防住这个风险。([GitHub][10])

**最小修复**

在 forward 中 encode 前后严格校验：

```python
if self.encode_policy == "selected_only":
    if selected is None:
        raise RuntimeError("selected_only requires selected path")
    expected = int(selected.selected_masks.sum().item())
    if pixels.shape[0] != expected:
        raise RuntimeError(f"selected_only encoded {pixels.shape[0]} frames, expected {expected}")
    if self.assert_selected_only and expected > total_dense:
        raise RuntimeError("selected slots exceed dense input")
```

如果 `keep_ratio < 1` 且 all-valid，可以加 warning/assert：

```python
if expected == total_dense and self.frame_selector.policy != "causal_motion":
    ...
```

**需要补的测试**

* monkeypatch `_encode_pixels_chunked`，记录 `pixels.shape[0]`，输入 `B=1,T=8,stride=2`，必须等于 4，不是 8。
* 故意让 `_select_frames_before_vision` 返回 None，在 selected_only 下必须抛错。
* `B=2` varying selection，encoded count 必须等于 selected mask sum。

---

## P1-3：post-processing 的坐标轴/NMS 只在 dense_grid path 下相对安全，selected_axis fallback 仍危险

**位置**
`opentad/models/detectors/single_stage.py::post_processing`
`opentad/models/utils/post_processing/utils.py::grid_to_seconds`
可搜索片段：

```python
segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
...
if rpn_meta ... proposal_axis == "dense_grid":
    seconds_meta["irregular_native_axis"] = True
segments = convert_to_seconds(segments, seconds_meta)
```

`grid_to_seconds` 会优先用 `token_times_sec` 解码 selected-axis，之后才看 `irregular_selected_positions`。([GitHub][3])

**问题是什么**
当前 final MATR dense-grid proposal path 可以绕开 double conversion，这是对的。但 fallback selected_axis path 中，NMS 在 convert_to_seconds 前执行；如果 selected tokens 是非均匀时间，selected-axis IoU 和 seconds IoU 不等价，NMS 可能保错 proposal。

**为什么影响 online/TAD/mAP**
高 tIoU mAP 对边界坐标极敏感。NMS 坐标轴错，会导致重复预测保留/删除错误，尤其会伤 @0.6/@0.7。

**最小修复**

给 post-processing 增加显式 axis policy：

```python
proposal_axis = rpn_meta.get("proposal_axis", "selected_axis") if rpn_meta else "selected_axis"

if proposal_axis != "dense_grid":
    segments_for_nms = convert_to_seconds(segments.clone(), metas[i])
else:
    segments_for_nms = segments

# either run NMS in seconds consistently, or decode all to dense_grid first
```

更推荐：所有 detector output 在进入 NMS 前统一转换到 seconds，NMS/AP 都在 seconds 上做；ledger 另存 source_grid/source_frame。

**需要补的测试**

* 构造 irregular token_times `[0, 1, 10]`，两个 proposal 在 selected-axis IoU 与 seconds IoU 排序不同，确认 NMS 使用 seconds 后结果正确。
* dense_grid proposal_axis 下不得再次 irregular decode。

---

## P1-4：`online_censored_training` 没有真正 censor 所有未来标签

**位置**
`opentad/models/dense_heads/matr_head.py::_build_online_branch_targets`
可搜索片段：

```python
if self.online_censored_training:
    endpoint_observed = segment[1] <= centers + self.max_future_offset + 1e-6
    end_mask = torch.logical_and(end_mask, endpoint_observed)

actionness_target[batch_idx, inside_mask, 0] = 1.0
```

([GitHub][6])

**问题是什么**
只 censor 了 end target；inside/actionness target 仍然对整个 GT segment 内的中心点置 1，包括那些在真实在线时尚未知道该 action 会持续到何处的时刻。主 detection loss `self.losses(...)` 也仍用完整 `gt_segments`。

**为什么影响 online/TAD/mAP**
这会让训练阶段看到完整未来 segment 的监督语义，而测试时靠 `clamp_end_to_current` 截断。模型学到的是 offline segment structure，而不是严格 online emission decision。

**最小修复**

至少将 online auxiliary targets 分成 observed/pending：

```python
observed_end = segment[1] <= centers + self.max_future_offset + 1e-6
inside_observed = inside_mask & observed_end
actionness_target[batch_idx, inside_observed, 0] = 1.0
emit_target[batch_idx, end_mask & observed_end, 0] = 1.0
```

更严格：为 online route 单独构造 censored gt segments：在每个 center/emit time，只允许使用 `gt_start <= now` 且 `gt_end <= now + allowed_future_offset` 的完成事件；未完成 action 可作为 pending/ignore，而不是 positive full-regression target。

**需要补的测试**

* GT segment `[10, 100]`，center=20，`max_future_offset=0`：end/actionness/emit 不应把该点当作完整已观测 positive。
* center=100 后才允许 end/emit positive。

---

## P2-1：`decode_irregular_segments_to_seconds` 的右端 sentinel 使用整段视频 duration，滑窗下会拉爆 segment

**位置**
`opentad/models/utils/irregular_time_decode.py::decode_irregular_segments_to_seconds`
`opentad/models/backbones/online_siglip_adapter.py::_write_selection_meta`
可搜索片段：

```python
end_time = float(duration) if duration is not None else times[-1]
fp = segments.new_tensor(times + [end_time])
```

`_write_selection_meta` 只写 token times，没有写 window right boundary。([GitHub][8])

**问题是什么**
`duration` 是视频总时长，不是当前滑窗末端。若 selected tokens 属于一个局部 window，coordinate `len(token_times)` 会被插值到整段视频末尾，导致 proposal 右边界过大。

**为什么影响 online/TAD/mAP**
一旦走 token_times fallback path，高 tIoU 会崩，因为短窗口内的 proposal 会被扩展成跨越大半个视频的 segment。

**最小修复**

meta 中增加：

```python
meta["token_right_boundary_sec"] = (
    window_end_frame + offset_frames
) / fps
```

decode 改成：

```python
right_time = meta.get("token_right_boundary_sec", None)
decode_irregular_segments_to_seconds(..., right_boundary_sec=right_time)
```

并要求 `right_boundary_sec >= token_times_sec[-1]`，否则 fail-fast。

**需要补的测试**

* `token_times_sec=[10,12,14]`，video duration=100，window_end_sec=16，segment `[2,3]` 应 decode 到 `[14,16]`，不是 `[14,100]`。

---

## P2-2：streaming path 没有显式携带 `proposal_axis`，目前靠 final MATR dense_grid 偶然安全

**位置**
`opentad/models/detectors/single_stage.py::_format_streaming_safe_results`
可搜索片段：

```python
explicit_source_frame = grid_spec.grid_to_frame(local_source_grid)
...
OnlineCandidate(start_grid=float(segment[0].item()), end_grid=float(segment[1].item()))
```

([GitHub][3])

**问题是什么**
streaming path 用 `GridSpec.grid_to_frame` 直接把 `segment` 和 `source_grid` 当 dense grid 转 frame。对于当前 `MATRHead` dense_grid proposal_axis，这基本合理；但如果未来 rpn_head 返回 selected_axis 或 token_times-native proposal，streaming path 不会知道，直接错解坐标。

**为什么影响 online/TAD/mAP**
source_frame、end_frame、emit_frame 是 online no-future audit 的核心字段。坐标轴错会导致 false no-future pass 或 false violation，并污染 latency。

**最小修复**

`post_processing` 调 `_format_streaming_safe_results` 时传入：

```python
proposal_axis = rpn_meta.get("proposal_axis", "selected_axis")
```

然后：

```python
if proposal_axis == "dense_grid":
    frame = grid_spec.grid_to_frame(...)
elif proposal_axis == "selected_axis":
    # use irregular_selected_positions/token_times_sec to map to dense/frame first
else:
    raise ValueError(...)
```

**需要补的测试**

* dense_grid path：source_grid=10, stride=4 -> source_frame 正确。
* selected_axis path：selected_positions=[0, 10, 20]，source_grid=1 -> source_frame 对应 dense 10，不是 selected index 1。

---

## P2-3：registration 文件基本正确，但只证明可 import，不证明运行合同

**位置**
`opentad/models/selectors/__init__.py`
`opentad/evaluations/__init__.py`
`opentad/models/__init__.py`

selector init 导出了 `CausalFrameSelector`，evaluation init 导出了 `OnlineMAP`，models init 加了 `from .selectors import *`。([GitHub][11])

**问题是什么**
注册是必要条件，不是行为正确性。当前 `tests/test_final_online_tad_contracts.py` 主要检查 config 字段和源码字符串，例如 `"selected.frames.reshape"`、`"proposal_axis"`、`"source_frame"` 是否存在。([GitHub][10])

**为什么影响 online/TAD/mAP**
字符串测试能防 claim drift，但不能发现 selected-only 伪路径、latency 反向、batch>1 crash、NMS 坐标轴错、forward/backward shape mismatch。

**最小修复**

新增 runtime tests，而不是替换 claim hygiene tests：

```text
test_selected_only_encoder_runtime_count
test_online_emitter_latency_budget_semantics
test_matr_irregular_batch_size_fail_closed
test_online_map_filtered_stats_match_ap_rows
test_dense_grid_no_double_irregular_decode
```

**需要补的测试**

见上。

---

# 可执行修复计划

## 第一步：先修 P0，不要跑远端长训

1. 修 `OnlineEmitter.step` latency 语义与 clock update。
2. 在 final config 或 dataloader 层强制 adaptive irregular route `batch_size=1`，并在 `MATRHead` 早期 fail-fast。
3. 修 `OnlineMAP._import_prediction` 的过滤顺序，让 online stats 与 AP rows 使用同一 filtered result set。
4. 修 selector all-invalid mask，至少先 fail-fast。

## 第二步：补 5 个必须过的本地行为测试

```text
pytest tests/test_online_emitter_latency.py -q
pytest tests/test_selected_only_runtime.py -q
pytest tests/test_matr_irregular_batch_contract.py -q
pytest tests/test_online_map_filter_consistency.py -q
pytest tests/test_irregular_time_decode_window_boundary.py -q
```

其中 selected-only runtime test 必须 monkeypatch `_encode_pixels_chunked`，不能只查源码字符串。

## 第三步：再做 tiny backend build/forward/backward smoke

最小 smoke 条件：

```text
backend="tiny"
B=1
T=8 or 16
policy="causal_stride"
streaming_safe_emission=True
rpn_head.online_censored_training=True
```

检查：

```text
loss finite
backward finite
encoded_frames < dense_frames
irregular_selected_positions exists
proposal_axis == "dense_grid"
ledger has emit_frame/source_frame/start_frame/end_frame/latency_sec
no future violations == 0
```

## 第四步：真实 adaptive 前先改 claim 或改 selector

两条路线二选一：

* **保守路线**：把 final route 改名为 `selected-only causal stride online TAD skeleton`，不叫 adaptive。
* **研究路线**：实现 learned/causal utility selector 或至少 causal_motion adaptive policy，并证明 selected positions 随内容变化。

---

# 关键 patch sketches

## Patch A：修 latency gate

```python
# opentad/utils/online_protocol.py

def step(self, video_name: str, now_frame: int, state: OnlineState, candidates):
    now_frame = int(now_frame)

    if video_name != state.video_name:
        raise ProtocolViolation(f"state video mismatch: {state.video_name} vs {video_name}")

    if now_frame < state.last_emit_frame:
        raise ProtocolViolation(
            f"non-monotonic online stream for {video_name}: "
            f"now_frame={now_frame}, last_emit_frame={state.last_emit_frame}"
        )

    detections = []
    for cand in candidates:
        if int(cand.source_grid) <= state.last_emitted_grid:
            continue
        if float(cand.score) < self.score_threshold:
            continue
        if cand.source_frame is not None and int(cand.source_frame) > now_frame:
            continue

        start_frame = self.grid_spec.grid_to_frame(cand.start_grid)
        end_frame = self.grid_spec.grid_to_frame(cand.end_grid)

        if end_frame > now_frame:
            continue

        latency_frames = now_frame - end_frame
        if self.latency_frames >= 0 and latency_frames > self.latency_frames:
            continue

        source_frame = (
            int(cand.source_frame)
            if cand.source_frame is not None
            else self.grid_spec.grid_to_frame(cand.source_grid)
        )
        if source_frame > now_frame:
            continue

        detections.append(
            OnlineDetection(
                video_name=video_name,
                label=cand.label,
                score=float(cand.score),
                start_frame=start_frame,
                end_frame=end_frame,
                emit_frame=now_frame,
                source_grid=int(cand.source_grid),
                source_frame=source_frame,
                latency_sec=float(latency_frames) / float(self.grid_spec.fps),
            )
        )

    emitted = []
    for det in prefix_nms(detections, self.nms_iou_threshold):
        if any(
            det.label == prev.label and _segment_iou(det, prev) > self.nms_iou_threshold
            for prev in state.emitted
        ):
            continue
        emitted.append(det)

    if emitted:
        state.last_emitted_grid = max(state.last_emitted_grid, max(det.source_grid for det in emitted))
        state.emitted.extend(emitted)

    state.last_emit_frame = now_frame
    return emitted
```

## Patch B：`OnlineMAP` 过滤一致性

```python
def _filter_prediction_results_for_online_map(self, results):
    filtered = {}
    for video_id, rows in results.items():
        if video_id in self.blocked_videos:
            continue
        if self.allowed_videos is not None and video_id not in self.allowed_videos:
            continue
        kept = []
        for row in rows:
            if self._validate_ledger_row(row):
                kept.append(row)
        filtered[video_id] = kept
    return filtered

def _import_prediction(self, prediction_filename):
    data = self._load_prediction_data(prediction_filename)
    if not all([field in list(data.keys()) for field in self.pred_fields]):
        raise IOError("Please input a valid prediction file.")

    filtered_results = self._filter_prediction_results_for_online_map(data["results"])
    self.online_metric_dict = compute_online_detection_metrics(
        filtered_results,
        require_no_future=self.require_no_future,
    )

    # build dataframe from filtered_results, not data["results"]
```

## Patch C：selector all-invalid fail-fast

```python
def _valid_indices(self, mask, seq_len, device):
    if mask is None:
        return torch.arange(seq_len, device=device, dtype=torch.long)
    valid = torch.nonzero(mask.bool(), as_tuple=False).flatten().to(device=device)
    if valid.numel() == 0:
        raise ValueError("CausalFrameSelector got all-invalid mask; refusing to encode fake frame 0")
    return valid
```

## Patch D：selected-only runtime assert

```python
def forward(self, frames, masks=None, metas=None):
    frames, masks = self._prepare_frames(frames, masks=masks)
    batch_size, seq_len, channels, height, width = frames.shape
    selected = self._select_frames_before_vision(frames, masks=masks, metas=metas)

    if self.encode_policy == "selected_only" and selected is None:
        raise RuntimeError("encode_policy='selected_only' requires selected frames at runtime")

    if selected is None:
        pixels = frames.reshape(batch_size * seq_len, channels, height, width)
        ...
    else:
        expected = int(selected.selected_masks.sum().item())
        pixels = selected.frames.reshape(selected.frames.shape[0], channels, height, width)
        if self.assert_selected_only and pixels.shape[0] != expected:
            raise RuntimeError(f"selected-only encoded {pixels.shape[0]} frames, expected {expected}")
```

---

# 最终放行建议

**当前状态：HOLD。**

允许：

```text
PASS_ALLOW_LOCAL_STATIC_AND_TINY_SMOKE_ONLY
```

禁止：

```text
NO_REMOTE_REAL_TRAINING
NO_PAPER_CLAIM
NO_ONLINE_MAP_CLAIM
NO_ADAPTIVE_SELECTOR_CLAIM
```

修完 P0 并补 runtime tests 后，可以进入：

```text
PASS_ALLOW_SINGLE_GPU_B1_TINY_OR_SHORT_THUMOS_SMOKE
```

只有当 `selected-only encoded frame count`、`batch irregular contract`、`latency semantics`、`OnlineMAP filtering consistency`、`dense_grid/irregular decode` 五类行为测试全部通过后，才建议开始远端短训。

[1]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/7caf2b9 "implement adaptive online tad target route · yuzbo/OpenTAD_OnlineTADClean_20260702@7caf2b9 · GitHub"
[2]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py "OpenTAD_OnlineTADClean_20260702/configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[3]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/opentad/models/detectors/single_stage.py "OpenTAD_OnlineTADClean_20260702/opentad/models/detectors/single_stage.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[4]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/opentad/evaluations/online_map.py "OpenTAD_OnlineTADClean_20260702/opentad/evaluations/online_map.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[5]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/opentad/models/backbones/online_siglip_adapter.py "OpenTAD_OnlineTADClean_20260702/opentad/models/backbones/online_siglip_adapter.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[6]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/opentad/models/dense_heads/matr_head.py "OpenTAD_OnlineTADClean_20260702/opentad/models/dense_heads/matr_head.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/opentad/utils/online_protocol.py "OpenTAD_OnlineTADClean_20260702/opentad/utils/online_protocol.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[8]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/opentad/models/utils/irregular_time_decode.py "OpenTAD_OnlineTADClean_20260702/opentad/models/utils/irregular_time_decode.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[9]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/opentad/models/selectors/causal_frame_selector.py "OpenTAD_OnlineTADClean_20260702/opentad/models/selectors/causal_frame_selector.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/7caf2b9/tests/test_final_online_tad_contracts.py "OpenTAD_OnlineTADClean_20260702/tests/test_final_online_tad_contracts.py at 7caf2b972958328753d36f0f0079fce2893468c8 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/selectors/__init__.py "OpenTAD_OnlineTADClean_20260702/opentad/models/selectors/__init__.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
