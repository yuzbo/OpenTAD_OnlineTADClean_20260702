## 访问结论：无法访问仓库，当前 commit 不能被实际代码审查

我无法访问 `yuzbo/OpenTAD_OnlineTADClean_20260702`。用户给出的 GitHub 仓库页返回 404；我又尝试读取指定 commit 下的 raw 配置文件，同样返回 404。这个现象可能是 private、仓库/分支/commit 名称不存在、或权限不足；我不能区分是哪一种。

我还按仓库名、commit、关键文件名在可用文件库里搜索了你可能已经上传过的审查包，没有命中。因此下面不是“对 253e6a9 当前源码的逐行结论”，而是严格审稿门槛下的 **NO-GO / INCOMPLETE** 判定、必须补交材料、以及你这条路线要变成可发表 Online/Causal TAD 系统所需的实现补丁骨架。

---

# 0. 总判定

**当前审查状态：INCOMPLETE_BLOCKED_REPO_NOT_ACCESSIBLE。**

在我没有看到源码、测试、日志、resolved config、训练命令和 emission 输出前：

| 层级                                             |      审稿判定 | 是否可声称满足目标                                                |
| ---------------------------------------------- | --------: | -------------------------------------------------------- |
| P0 Frozen SigLIP/SigLIP2 recent-frame baseline |  **不能通过** | 不能声称已满足 no-future、shape、梯度、synthetic overfit             |
| P1 raw-frame online TAD                        |  **不能通过** | 不能声称今天可端到端训练                                             |
| P2 论文级 On-TAD 贡献                               | **直接不成立** | 不能只凭 “SigLIP2 + MATR-style head” 声称 Online/Causal TAD 贡献 |

严厉结论是：**在没有代码证据前，P0/P1/P2 都不能给 PASS；P2 尤其不能成立。**
P0 是一个合理工程起点，但它最多是 **frozen image encoder + causal adapter/head baseline**。它不是 streaming video encoder，不是完整 online TAD 贡献，更不是论文级 On-TAD 主方法。

---

# 1. 必须补交的最小审查包

为了让我做你要求的“引用具体文件/函数”的真实审查，需要提供以下之一：

```bash
git bundle create OpenTAD_OnlineTADClean_253e6a9.bundle --all
```

或：

```bash
git archive --format=tar.gz -o OpenTAD_OnlineTADClean_253e6a9.tar.gz 253e6a9
```

同时附上：

```bash
git show --stat --name-only 253e6a9
git diff <base_commit> 253e6a9 -- \
  configs/causaltad \
  opentad/datasets/raw_frame.py \
  opentad/datasets/transforms/loading.py \
  opentad/models/backbones/online_siglip_adapter.py \
  opentad/models/backbones/online_videomae_adapter.py \
  opentad/models/dense_heads/matr_head.py \
  opentad/models/detectors/single_stage.py \
  opentad/models/projections/causal_proj.py \
  opentad/cores/optimizer.py \
  opentad/models/utils/post_processing/utils.py \
  opentad/utils/online_protocol.py \
  tests/test_siglip_ontad_contracts.py \
  tests/test_siglip_ontad_runtime_smoke.py \
  tests/test_videomae_matr_ontad_contracts.py \
  tests/test_videomae_matr_runtime_smoke.py \
  tests/test_online_protocol_audit.py \
  tools/remote/submit_siglip_ontad_n16r4.sh
```

还要附：

```bash
python -m pytest tests/test_siglip_ontad_contracts.py -q
python -m pytest tests/test_siglip_ontad_runtime_smoke.py -q
python -m pytest tests/test_videomae_matr_ontad_contracts.py -q
python -m pytest tests/test_videomae_matr_runtime_smoke.py -q
python -m pytest tests/test_online_protocol_audit.py -q
```

以及一次 resolved config dump、一次 optimizer named-parameter dump、一次 synthetic no-future 日志、一次 synthetic overfit 曲线、一次 THUMOS 2-video smoke 日志。

没有这些材料，任何“当前实现正确”都是臆测。

---

# 2. P0 / P1 / P2 分层判定

## P0：Frozen SigLIP/SigLIP2 recent-frame baseline

P0 的最低合格定义不是“能 forward”。P0 必须同时满足：

1. raw frame sampling 在任意 emission time `t` 只能读取 `frame <= t`；
2. `memory_size=0` 时没有跨 tick、跨 batch、跨 video 的隐式状态；
3. SigLIP/SigLIP2 tower 完全 frozen；
4. optimizer 只训练 adapter / projection / MATR head；
5. no-future perturbation test 通过；
6. synthetic overfit 通过；
7. frame/grid/seconds roundtrip 无漂移；
8. online emission log 可以被独立 evaluator 读取。

当前我没看到代码，所以 **P0 不能 PASS**。
即使代码能跑训练，只要没有 no-future test 和 optimizer freeze audit，也只能叫 **offline-shaped smoke**，不能叫 causal baseline。

## P1：正式 raw-frame online TAD

P1 的最低合格定义是：从 raw frames 到 online emission 的训练与验证路径都使用同一套时间坐标，不用离线全视频后处理修正结果。

P1 必须证明：

1. 固定 processor、分辨率、FPS、stride；
2. dataset 每个 training sample 的可见帧集合是 prefix 或 causal window；
3. adapter/projection/head 的时序维度是 causal；
4. loss 可以用完整 GT 监督过去时刻，但 input feature 不能用未来；
5. post-processing 不能在完整视频预测集合上做全局 NMS 后再假装 online；
6. eval 读取 emission ledger，而不是完整 video-level detection dump。

当前我没看到这些实现，所以 **P1 不能声称今天可端到端训练**。
最多只能说：这是一条待验证的 raw-frame online TAD 工程路线。

## P2：论文级贡献

P2 如果只是 “Frozen SigLIP2 + MATR-style head”，论文 claim 直接不成立。

SigLIP/SigLIP2 是图文/图像视觉语言编码器路线。SigLIP 使用独立 image/text encoder 生成表示；SigLIP2 在 SigLIP 基础上扩展了预训练配方，包括 captioning-based pretraining、自监督损失和 online data curation。([GitHub][1])
这能提供强图像语义，但不能自动提供 causal motion modeling、streaming memory、边界迟滞控制、在线 emission 协议。

P2 必须至少有一个真正贡献：

1. causal motion branch；
2. streaming-safe boundary/emission protocol；
3. latency-aware online detection objective；
4. prefix-only post-processing/evaluation；
5. 与真正 streaming video encoder 或 offline video encoder 的严格对照。

否则 P2 不能写成论文贡献，只能写成 baseline。

---

# 3. 严重风险清单：按你列的文件映射

以下不是对当前源码的已证实 bug，而是 **必须在这些文件中逐项验证的阻断项**。任意一项失败，online/causal claim 直接判死刑。

## S0 阻断级：未来信息泄漏

### 3.1 `opentad/datasets/raw_frame.py`

必须检查 dataset sample 的时间定义。严禁：

```python
center = ...
start = center - window // 2
end = center + window // 2
```

这在 online TAD 中是偷看未来。合法写法只能是：

```python
end = current_frame
start = max(0, end - (num_frames - 1) * frame_stride)
```

如果 `__getitem__` 为了构造一个 clip 使用了 action proposal center、完整 video length、完整 annotation boundary 来选 window，这也是泄漏。

**合格条件：**

```text
visible_frames(sample, emit_frame) ⊆ [0, emit_frame]
```

### 3.2 `opentad/datasets/transforms/loading.py`

危险点是 transform 名称看起来只是 loading，但实际可能做 temporal crop、resize、pad、uniform sample。

严禁：

1. `TemporalCenterCrop`;
2. `RandomResizedTemporalCrop` 但 crop 范围来自完整视频；
3. `sample_indices` 使用未来 frame；
4. 对短 clip 用右侧 future padding；
5. 训练时用完整 video duration 归一化当前 tick 的相对时间。

短 clip 只能左 pad 或 zero pad，不能右 pad future frames。

### 3.3 `opentad/models/backbones/online_siglip_adapter.py`

SigLIP/SigLIP2 recent-frame encoder 如果只是把每帧作为 image 编码，这是 P0 可接受 baseline，但不能声称 video encoder。

Hugging Face 的 image processor 通常执行 resize、center crop、rescale、normalize、pad 等图像级预处理；SigLIP processor 默认包含 resize、rescale、normalize、RGB conversion 等图像处理。([Hugging Face][2])
这些操作本身不必然泄漏未来，但如果你的 processor batching 以同一视频未来帧共同决定 dynamic size、pad size、normalization stats，就构成泄漏。

**必须固定：**

```text
processor_name
image_size
crop_policy
do_resize
do_center_crop
mean/std
fps
frame_stride
```

并禁止 batch-dependent temporal behavior。

### 3.4 `opentad/models/projections/causal_proj.py`

所有 1D conv、attention、pooling 必须 causal。

不合格写法：

```python
nn.Conv1d(C, C, kernel_size=3, padding=1)
```

这会看右邻 token。合法写法：

```python
x = F.pad(x, (kernel_size - 1, 0))
y = conv_no_padding(x)
```

Transformer 也必须有 lower-triangular causal mask。没有 mask 的 temporal self-attention 是 offline head。

### 3.5 `opentad/models/dense_heads/matr_head.py`

“MATR-style head”不是天然 online。必须检查：

1. query 是否 attend 到完整 temporal memory；
2. decoder 是否一次看完整 T；
3. boundary regression 是否依赖 future token；
4. loss 是否训练未来 emission；
5. decode 是否对完整视频统一 top-k / NMS。

如果 head 在完整序列上输出所有 detection，再交给 post-processing，这就是 offline TAD，不是 On-TAD。

### 3.6 `opentad/models/utils/post_processing/utils.py`

全视频 NMS 是最常见的伪 online 漏洞。

不合格：

```python
detections = collect_all_video_predictions(video)
detections = batched_nms(detections)
```

合格：

```python
for tick in stream:
    prefix_candidates = decode_prefix_only(...)
    emissions = online_emitter.step(prefix_candidates, now=tick)
```

evaluation 可以在事后用完整 GT 算 mAP，但被评估的 detection 必须是在线 emission log，而不是完整视频预测集合。

---

## S1 高危：streaming state 键错误

### `opentad/utils/online_protocol.py`

streaming memory 不能按 batch index 管理。batch index 每个 iteration 都会重排。合法 key 至少包含：

```text
video_name
stream_id
absolute_frame
fps
sample_stride
processor_id
encoder_id
```

如果 state 只按 `video_id` 或 `batch_idx` 存，风险包括：

1. 不同视频状态污染；
2. 同一视频不同 stride/run 污染；
3. DDP worker 间污染；
4. overlap window 重复写入；
5. test-time state 残留到下一视频。

---

## S1 高危：optimizer 误训 frozen tower

### `opentad/cores/optimizer.py`

如果 optimizer 用：

```python
optimizer = AdamW(model.parameters(), ...)
```

即使部分参数 `requires_grad=False`，仍然缺少审计证据。严格实现必须：

1. 显式 freeze tower；
2. 显式列出 trainable prefixes；
3. 构建 param groups 时 assert；
4. backward 后 assert frozen grad is None or zero；
5. log trainable/frozen 参数数量和名称。

P0 只允许训练 adapter/projection/head。P2 motion branch 加入后才允许训练 motion branch。SigLIP/SigLIP2 tower 不允许误训。

---

## S2 中危：frame/grid/seconds roundtrip 不可靠

### `raw_frame.py` / `loading.py` / `online_protocol.py` / `post_processing/utils.py`

在线 TAD 最容易出现 1-2 grid 漂移，最后直接伤害 mAP@0.6/0.7。

必须明确每个 grid token 表示：

1. center frame？
2. end frame？
3. sampled raw frame？
4. encoder stride 后的 token？
5. feature window 的最后可见 frame？

对 online TAD，我建议用 **end-frame anchored grid**：

```text
grid g 的时间戳 = 该 token 可见窗口的最后一帧
```

这样 emission causality 最容易验证。

---

# 4. Frozen SigLIP/SigLIP2 作为在线视觉编码器：合理，但科学短板很大

**合理性：**

1. 部署简单：逐帧 image encoder，不需要未来 clip；
2. 迁移性强：图像语义强，适合低成本 baseline；
3. 训练成本低：frozen tower，只训练轻量 adapter/projection/head；
4. no-future 容易做严谨测试；
5. 可以作为 P0/P1 的 portable visual frontend。

**科学短板：**

1. 它不是 motion encoder；
2. 它不建模光流、速度、方向、动作相位；
3. 相邻帧 temporal relation 全靠下游 causal head 学；
4. 对细边界、高 IoU 定位弱；
5. 对“动作开始前兆”和“动作结束迟滞”没有天然建模；
6. 与真正 video encoder 不公平：VideoMAE 这类路线针对视频 tube masking / temporal redundancy 建模，核心就是用视频时空结构学习表征。([Hugging Face][3])

所以 Frozen SigLIP/SigLIP2 可以是最可移植 P0 baseline，但不能作为 P2 论文贡献的主体。

---

# 5. 从当前路线到可发表 On-TAD 的分阶段计划

## Phase A：先补审计，不许先跑大训练

目标：证明不是伪 online。

必须先补：

1. `StreamKey`;
2. causal raw-frame loader；
3. frame/grid/seconds roundtrip；
4. optimizer param audit；
5. no-future perturbation test；
6. prefix-only online emitter；
7. emission JSONL evaluator。

通过前，不允许写 “online TAD achieved”。

## Phase P0：Frozen SigLIP/SigLIP2 image-frame baseline

只做：

```text
raw frame -> fixed processor -> frozen SigLIP/SigLIP2 image features
-> causal adapter/projection -> causal MATR-style head
-> online emission protocol
```

`memory_size=0`，每个 tick 只允许 current/recent past frames。

P0 成功标准：

1. no-future test pass；
2. shape test pass；
3. frozen grad test pass；
4. synthetic overfit pass；
5. 2-video THUMOS smoke pass；
6. emission log 可独立复算。

## Phase P1：正式 raw-frame online TAD

加入：

1. 固定 FPS/stride；
2. variable-length stream handling；
3. prefix-only NMS；
4. latency-aware metrics；
5. THUMOS train/eval；
6. 与 offline OpenTAD feature baseline 区分。

P1 成功标准：

1. 可以完整训练；
2. train/test 时间坐标一致；
3. online evaluator 不使用完整视频预测；
4. 报告 latency/memory；
5. P0 baseline 与 P1 full route 分开命名。

## Phase P2：论文级贡献

二选一或都做：

### P2-a：Causal motion branch

用过去帧差、过去 feature diff、causal temporal conv 建模 motion。

### P2-b：Streaming-safe boundary/emission protocol

引入：

1. boundary readiness；
2. end hazard；
3. emission delay budget；
4. late/early penalty；
5. prefix-only calibrated score。

P2 成功标准不是 mAP 变高一点，而是：

```text
same visual encoder 下，
causal motion / emission protocol
在 high-IoU、低延迟、低内存条件下稳定优于 P1。
```

---

# 6. 关键 patch-style 代码骨架

下面这些代码是你当前仓库应当具备的最低合同。函数名可按实际项目调整，但语义不能弱化。

---

## 6.1 streaming state key

```python
# opentad/utils/online_protocol.py

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Tuple, Optional, List
import torch


@dataclass(frozen=True)
class StreamKey:
    video_name: str
    stream_id: str
    source_fps_num: int
    source_fps_den: int
    sample_stride: int
    processor_id: str
    encoder_id: str

    @property
    def source_fps(self) -> Fraction:
        return Fraction(self.source_fps_num, self.source_fps_den)


@dataclass
class StreamingState:
    key: StreamKey
    last_seen_frame: int = -1
    last_emitted_grid: int = -1
    memory_frames: List[int] = field(default_factory=list)
    memory_feats: Optional[torch.Tensor] = None  # [T_mem, C]
    pending: List[dict] = field(default_factory=list)

    def assert_monotonic(self, new_frame: int) -> None:
        if new_frame <= self.last_seen_frame:
            raise ValueError(
                f"Non-monotonic stream for {self.key.video_name}: "
                f"new_frame={new_frame}, last_seen={self.last_seen_frame}"
            )

    def append_feature(
        self,
        frame_idx: int,
        feat: torch.Tensor,
        memory_size: int,
    ) -> None:
        self.assert_monotonic(frame_idx)

        self.memory_frames.append(int(frame_idx))
        feat = feat.detach()

        if self.memory_feats is None:
            self.memory_feats = feat[None]
        else:
            self.memory_feats = torch.cat([self.memory_feats, feat[None]], dim=0)

        if memory_size == 0:
            self.memory_frames.clear()
            self.memory_feats = None
        elif len(self.memory_frames) > memory_size:
            keep = len(self.memory_frames) - memory_size
            self.memory_frames = self.memory_frames[keep:]
            self.memory_feats = self.memory_feats[keep:]

        self.last_seen_frame = int(frame_idx)


class StreamStateBank:
    def __init__(self):
        self._states: Dict[StreamKey, StreamingState] = {}

    def get_or_create(self, key: StreamKey) -> StreamingState:
        if key not in self._states:
            self._states[key] = StreamingState(key=key)
        return self._states[key]

    def reset(self, key: StreamKey) -> None:
        self._states.pop(key, None)

    def reset_video(self, video_name: str) -> None:
        dead = [k for k in self._states if k.video_name == video_name]
        for k in dead:
            self._states.pop(k, None)
```

审稿要求：任何 state 如果只按 `batch_idx`、`video_id`、`clip_id` 存，直接不合格。

---

## 6.2 禁用或修正重叠窗口 memory

```python
# opentad/utils/online_protocol.py

def update_stream_no_overlap(
    state: StreamingState,
    frame_indices: torch.Tensor,  # [T]
    feats: torch.Tensor,          # [T, C]
    memory_size: int,
) -> None:
    """
    Only new absolute frames may update state.
    Overlap windows are allowed for compute reuse only, not for duplicate state writes.
    """
    if frame_indices.ndim != 1:
        raise ValueError("frame_indices must be [T]")
    if feats.ndim != 2:
        raise ValueError("feats must be [T, C]")
    if len(frame_indices) != len(feats):
        raise ValueError("frame_indices and feats length mismatch")

    for idx, feat in zip(frame_indices.tolist(), feats):
        idx = int(idx)

        if idx <= state.last_seen_frame:
            # Duplicate overlap frame. Do not append, do not emit, do not update memory.
            continue

        state.append_feature(
            frame_idx=idx,
            feat=feat,
            memory_size=memory_size,
        )
```

严格建议：P0 `memory_size=0` 时先完全禁用 memory。P1/P2 再打开 memory，但必须有 monotonic 和 no-duplicate audit。

---

## 6.3 causal raw-frame window

```python
# opentad/datasets/raw_frame.py

def causal_frame_indices(
    emit_frame: int,
    num_frames: int,
    frame_stride: int,
    video_num_frames: int,
) -> list[int]:
    """
    Return a left-padded causal frame window ending at emit_frame.
    Never returns an index > emit_frame.
    """
    if emit_frame < 0:
        raise ValueError("emit_frame must be non-negative")
    if emit_frame >= video_num_frames:
        emit_frame = video_num_frames - 1

    indices = [
        emit_frame - i * frame_stride
        for i in reversed(range(num_frames))
    ]

    # Left pad by clamping to 0. Never right pad with future frames.
    indices = [max(0, int(i)) for i in indices]

    assert max(indices) <= emit_frame
    assert min(indices) >= 0
    return indices
```

禁止 centered clip：

```python
# 不合格
start = center - half
end = center + half
```

---

## 6.4 frame -> grid -> seconds roundtrip

```python
# opentad/utils/online_protocol.py

from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class GridSpec:
    source_fps_num: int
    source_fps_den: int
    frame_hop: int
    first_grid_end_frame: int = 0

    @property
    def fps(self) -> Fraction:
        return Fraction(self.source_fps_num, self.source_fps_den)

    def grid_to_end_frame(self, grid_idx: int) -> int:
        return int(self.first_grid_end_frame + grid_idx * self.frame_hop)

    def end_frame_to_grid_floor(self, frame_idx: int) -> int:
        return int(math.floor((frame_idx - self.first_grid_end_frame) / self.frame_hop))

    def frame_to_seconds(self, frame_idx: int) -> float:
        return float(Fraction(frame_idx, 1) / self.fps)

    def seconds_to_frame_nearest(self, sec: float) -> int:
        return int(round(sec * float(self.fps)))

    def grid_to_seconds(self, grid_idx: int) -> float:
        return self.frame_to_seconds(self.grid_to_end_frame(grid_idx))


def assert_grid_roundtrip(spec: GridSpec, max_grid: int) -> None:
    for g in range(max_grid + 1):
        f = spec.grid_to_end_frame(g)
        g2 = spec.end_frame_to_grid_floor(f)
        if g != g2:
            raise AssertionError((g, f, g2))

        sec = spec.grid_to_seconds(g)
        f2 = spec.seconds_to_frame_nearest(sec)
        if abs(f2 - f) > 1:
            raise AssertionError((g, f, sec, f2))
```

建议你在每条 emission 里保存：

```json
{
  "video_name": "...",
  "source_fps": "30000/1001",
  "sample_stride": 4,
  "source_grid": 123,
  "grid_end_frame": 492,
  "emit_frame": 520,
  "segment_start_sec": 12.34,
  "segment_end_sec": 13.20,
  "latency_sec": 0.93
}
```

没有这些字段，online eval 不可审计。

---

## 6.5 online emission protocol

```python
# opentad/utils/online_protocol.py

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class Detection:
    video_name: str
    label: int
    score: float
    start_frame: int
    end_frame: int
    emit_frame: int
    source_grid: int


def prefix_nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
    # 只对当前 prefix 已 eligible 的候选做 NMS。
    # 不允许等完整视频结束后统一排序/NMS。
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []

    def iou(a: Detection, b: Detection) -> float:
        inter = max(0, min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame))
        union = max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame)
        return inter / max(union, 1)

    for d in dets:
        if all(iou(d, k) <= iou_thr or d.label != k.label for k in kept):
            kept.append(d)
    return kept


class OnlineEmitter:
    def __init__(
        self,
        grid_spec: GridSpec,
        latency_frames: int,
        nms_iou_thr: float,
        score_thr: float,
    ):
        self.grid_spec = grid_spec
        self.latency_frames = int(latency_frames)
        self.nms_iou_thr = float(nms_iou_thr)
        self.score_thr = float(score_thr)

    def step(
        self,
        *,
        video_name: str,
        now_frame: int,
        state: StreamingState,
        cls_scores: torch.Tensor,     # [T, K]
        offsets: torch.Tensor,        # [T, 2], left/right in frames or grid units
    ) -> List[Detection]:
        """
        Emits only detections whose predicted end is <= now_frame - latency.
        """
        eligible_end_frame = now_frame - self.latency_frames
        if eligible_end_frame < 0:
            return []

        max_grid = self.grid_spec.end_frame_to_grid_floor(eligible_end_frame)
        min_grid = state.last_emitted_grid + 1

        if max_grid < min_grid:
            return []

        candidates: List[Detection] = []

        for g in range(min_grid, min(max_grid + 1, cls_scores.shape[0])):
            anchor_end = self.grid_spec.grid_to_end_frame(g)
            scores_g = cls_scores[g]

            for label in range(scores_g.numel()):
                score = float(scores_g[label])
                if score < self.score_thr:
                    continue

                left = int(round(float(offsets[g, 0])))
                right = int(round(float(offsets[g, 1])))

                start_frame = max(0, anchor_end - left)
                end_frame = anchor_end + right

                # 严格在线：不能 emit 未来结束的 segment。
                if end_frame > eligible_end_frame:
                    continue

                candidates.append(
                    Detection(
                        video_name=video_name,
                        label=label,
                        score=score,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        emit_frame=now_frame,
                        source_grid=g,
                    )
                )

        emitted = prefix_nms(candidates, self.nms_iou_thr)
        state.last_emitted_grid = max_grid
        return emitted
```

这段协议解决两个常见伪 online 问题：

1. 不等完整视频结束后再 NMS；
2. 不 emit 当前时间还没结束的 segment。

---

## 6.6 optimizer param groups

```python
# opentad/cores/optimizer.py

def build_online_ontad_optimizer(model, cfg):
    frozen_prefixes = tuple(cfg.get("frozen_prefixes", [
        "backbone.vision_tower",
        "backbone.siglip",
        "backbone.siglip2",
        "backbone.image_encoder",
        "backbone.processor",
    ]))

    trainable_prefixes = tuple(cfg.get("trainable_prefixes", [
        "backbone.adapter",
        "projection",
        "proj",
        "neck",
        "dense_head",
        "head",
        "motion_branch",
    ]))

    frozen_names = []
    trainable = []

    for name, p in model.named_parameters():
        if name.startswith(frozen_prefixes):
            p.requires_grad_(False)
            frozen_names.append(name)
            continue

        if name.startswith(trainable_prefixes):
            p.requires_grad_(True)
            trainable.append((name, p))
            continue

        # fail-closed: unknown params are frozen unless explicitly allowed.
        p.requires_grad_(False)
        frozen_names.append(name)

    if not trainable:
        raise RuntimeError("No trainable parameters found for online On-TAD route.")

    bad = [
        name for name, p in model.named_parameters()
        if name.startswith(frozen_prefixes) and p.requires_grad
    ]
    if bad:
        raise RuntimeError(f"Frozen tower params still trainable: {bad[:20]}")

    param_groups = [
        {
            "params": [p for _, p in trainable],
            "lr": cfg["lr"],
            "weight_decay": cfg.get("weight_decay", 0.05),
            "name": "adapter_projection_head",
        }
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=cfg.get("betas", (0.9, 0.999)),
        eps=cfg.get("eps", 1e-8),
    )

    # Audit: no frozen parameter object may appear in optimizer.
    opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    frozen_in_opt = [
        name for name, p in model.named_parameters()
        if not p.requires_grad and id(p) in opt_param_ids
    ]
    if frozen_in_opt:
        raise RuntimeError(f"Frozen params included in optimizer: {frozen_in_opt[:20]}")

    print(f"[optimizer-audit] trainable={len(trainable)} frozen={len(frozen_names)}")
    print("[optimizer-audit] first trainable:", [n for n, _ in trainable[:20]])

    return optimizer
```

训练后还要加 backward audit：

```python
def assert_frozen_grads_zero(model, frozen_prefixes):
    bad = []
    for name, p in model.named_parameters():
        if name.startswith(tuple(frozen_prefixes)):
            if p.grad is not None and torch.any(p.grad != 0):
                bad.append(name)
    if bad:
        raise AssertionError(f"Frozen params received gradients: {bad[:20]}")
```

---

## 6.7 no-future causality test

```python
# tests/test_online_no_future.py

import copy
import torch


@torch.no_grad()
def run_online_until(model, frames, meta, stop_frame):
    """
    frames: [T, C, H, W]
    model.step must consume frames causally and emit prefix detections.
    """
    state = model.init_stream(meta)
    emissions = []

    stride = meta["sample_stride"]
    for now in range(0, stop_frame + 1, stride):
        frame = frames[now]
        out = model.step(frame=frame, now_frame=now, state=state, meta=meta)
        emissions.extend(out)

    return canonicalize_emissions(emissions)


def canonicalize_emissions(emissions):
    rows = []
    for e in emissions:
        rows.append((
            e["video_name"],
            int(e["label"]),
            round(float(e["score"]), 5),
            int(e["start_frame"]),
            int(e["end_frame"]),
            int(e["emit_frame"]),
        ))
    return sorted(rows)


def test_no_future_perturbation_online(model_factory):
    torch.manual_seed(0)

    T, C, H, W = 96, 3, 224, 224
    cut = 48

    frames = torch.randn(T, C, H, W)
    frames_perturbed = frames.clone()
    frames_perturbed[cut + 1:] = torch.randn_like(frames_perturbed[cut + 1:]) * 100.0

    meta = {
        "video_name": "synthetic_no_future",
        "stream_id": "test",
        "source_fps_num": 30,
        "source_fps_den": 1,
        "sample_stride": 4,
    }

    model_a = model_factory().eval()
    model_b = copy.deepcopy(model_a).eval()

    out_a = run_online_until(model_a, frames, meta, stop_frame=cut)
    out_b = run_online_until(model_b, frames_perturbed, meta, stop_frame=cut)

    assert out_a == out_b
```

这个测试必须在 P0 就通过。
如果模型 API 只能 `forward(full_video)`，那它不是 online API。

---

## 6.8 causal motion branch

```python
# opentad/models/backbones/causal_motion_branch.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMotionBranch(nn.Module):
    """
    Causal feature-difference motion branch.
    Input:  x [B, T, C]
    Output: y [B, T, C_out]
    No future access: only x[:, <=t] contributes to y[:, t].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")

        self.kernel_size = int(kernel_size)

        self.pre = nn.Linear(in_dim * 2, hidden_dim)
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=self.kernel_size,
            padding=0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must be [B, T, C]")

        prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        diff = x - prev

        z = torch.cat([x, diff], dim=-1)
        z = self.pre(z)

        z_ch = z.transpose(1, 2)              # [B, H, T]
        z_ch = F.pad(z_ch, (self.kernel_size - 1, 0))
        z_ch = self.conv(z_ch)                # [B, H, T]
        z = z_ch.transpose(1, 2)

        z = self.norm(z)
        z = self.drop(z)
        y = self.out(z)

        if valid_mask is not None:
            y = y * valid_mask[..., None].to(y.dtype)

        return y
```

P2 不能只加这个模块名。必须有 ablation：

```text
P1 frozen SigLIP frame-only
P2 frozen SigLIP + causal motion branch
P2 frozen SigLIP + streaming emission loss
P2 both
```

---

# 7. 最小实验矩阵

## 7.1 synthetic no-future

目标：证明 causal。

| Test                           | 修改              | 通过条件                                            |
| ------------------------------ | --------------- | ----------------------------------------------- |
| Future perturbation            | 替换 cut 之后所有帧    | cut 前 emissions 完全一致                            |
| Future deletion                | 删除 cut 之后所有帧    | cut 前 emissions 完全一致                            |
| Future annotation perturbation | 改 cut 之后 GT     | cut 前 loss mask / labels 不影响 input-side outputs |
| Batch reorder                  | 同一视频不同 batch 顺序 | emissions 一致                                    |
| Cross-video contamination      | A/B 视频交错输入      | state 不串流                                       |

## 7.2 synthetic overfit

构造 8-16 个 toy videos：

1. 红色块表示 action；
2. 蓝色块表示 background；
3. action start/end 可控；
4. 加短动作、长动作、back-to-back action；
5. 训练 200-1000 iter。

通过条件：

```text
loss 下降
classification overfit
boundary error < 1-2 grid
online emission latency 符合预算
```

## 7.3 real THUMOS smoke

最小：

```text
2 train videos
2 val videos
1 epoch or 100 iter
batch_size=1/2
num_workers=0 and >0 各一次
```

通过条件：

1. loss finite；
2. no NaN；
3. no future audit pass；
4. frozen grad pass；
5. emission JSONL 非空；
6. evaluator 可以读取 emission JSONL。

## 7.4 latency / memory

必须报告：

| 指标                     | 必须记录            |
| ---------------------- | --------------- |
| per-frame processor ms | mean / p95      |
| encoder ms             | mean / p95      |
| adapter/head ms        | mean / p95      |
| total emission latency | sec/frame       |
| GPU memory             | max allocated   |
| state memory           | per stream MB   |
| emitted detections     | per video count |
| late emissions         | ratio           |

没有 latency/memory，就不要用 “online deployable”。

## 7.5 ablations

最低矩阵：

| Visual                | Temporal      | Emission                | 目的                                    |
| --------------------- | ------------- | ----------------------- | ------------------------------------- |
| frozen SigLIP frame   | none          | offline NMS             | upper/lower diagnostic，不作为 online 主结果 |
| frozen SigLIP frame   | causal proj   | online emitter          | P0/P1 主 baseline                      |
| frozen SigLIP frame   | causal motion | online emitter          | P2-a                                  |
| frozen SigLIP frame   | causal proj   | readiness/emission head | P2-b                                  |
| VideoMAE offline clip | non-causal    | offline                 | 只作 offline reference                  |
| causal video encoder  | causal        | online                  | 强对照                                   |

## 7.6 failure cases

必须单列：

1. very short actions；
2. back-to-back actions；
3. action starts before enough context；
4. action ends near video tail；
5. long background；
6. camera cut / scene transition；
7. variable FPS；
8. corrupted/missing frames；
9. duplicate `video_name`；
10. DDP multi-worker state reset。

---

# 8. README / config / paper claim 必须修改

以下命名和 claim 在没有代码证据前都要降级。

## 8.1 P0 不能叫完整 On-TAD

不合格命名：

```text
end_to_end_online_tad
full_ontad
siglip2_matr_ontad
```

建议命名：

```text
frozen_siglip2_frame_matr_p0_causal_smoke
frozen_siglip2_recent_frame_p0_no_memory
```

## 8.2 不能把 P0 写成 VideoMAE adapter

如果主路线是 SigLIP/SigLIP2 image encoder，README 不能把结果称为 VideoMAE-adapter。
`configs/causaltad/thumos_videomae_adapter_matr_ontad.py` 必须和 SigLIP configs 分开说明，不能混写。

## 8.3 “MATR-style” 不能等于 “online MATR”

只有满足以下条件才能说 online head：

1. causal mask；
2. prefix decode；
3. prefix-only NMS；
4. emission ledger；
5. no-future test pass。

否则只能叫：

```text
MATR-style dense head adapted to causal setting, under validation
```

## 8.4 “SigLIP2 motion” 不能只靠帧差外壳

`thumos_siglip2_motion_matr_ontad_p2.py` 这个名字只有在 causal motion branch 实际启用、进入 optimizer、通过 ablation 后才合格。

否则应改名：

```text
thumos_siglip2_frame_matr_ontad_p1.py
```

## 8.5 “end-to-end raw video” 要谨慎

Frozen tower 路线可以说：

```text
raw frames to online emissions with frozen image encoder and trainable adapter/head
```

不能说：

```text
fully end-to-end trainable video model
```

除非视觉 tower 也参与训练，并且 optimizer audit 证明配置确实如此。

---

# 9. 最短修复路径

第一步不要跑 Slurm 大训练。先补 6 个 fail-closed gate：

```text
G1 repo/code access + resolved config
G2 causal raw-frame loader
G3 frame/grid/sec roundtrip
G4 optimizer freeze audit
G5 no-future perturbation test
G6 online emission JSONL evaluator
```

通过后才进入：

```text
P0 synthetic overfit
P0 THUMOS 2-video smoke
P1 full THUMOS train
P2 motion/emission ablation
```

在这之前，所有 README / paper wording 只能写：

```text
Implementation under causal-audit validation.
No formal online TAD claim yet.
```

---

# 10. 审稿尾部判定

```text
ONLINE_TAD_REVIEW_STATUS: INCOMPLETE_BLOCKED_REPO_NOT_ACCESSIBLE
REPO_ACCESS: FAILED_404
SOURCE_CODE_VISIBLE: NO
FILE_LIBRARY_MATCH: NO
P0_VERDICT: HOLD_NO_PASS_WITHOUT_CAUSAL_TESTS
P1_VERDICT: NO_PASS_WITHOUT_RAW_FRAME_ROUNDTRIP_AND_ONLINE_EVAL
P2_VERDICT: NO_GO_IF_ONLY_FROZEN_SIGLIP2_PLUS_MATR_STYLE_HEAD
CLAIM_STATUS: DO_NOT_CLAIM_ONLINE_CAUSAL_TAD_YET
NEXT_REQUIRED_ACTION: PROVIDE_GIT_BUNDLE_OR_DIFF_BUNDLE_WITH_TEST_LOGS
```

[1]: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip.md?utm_source=chatgpt.com "transformers/docs/source/en/model_doc/siglip.md at main"
[2]: https://huggingface.co/docs/transformers/en/main_classes/image_processor?utm_source=chatgpt.com "Image Processor"
[3]: https://huggingface.co/docs/transformers/en/model_doc/videomae?utm_source=chatgpt.com "VideoMAE"
