# Prefix Route Evidence Protocol V2

## Round-4 固定提交、零信任科学协议复审

## 1. Repository / commit verification

### 1.1 仓库可见性证书

```text
REPOSITORY_VISIBILITY_CERTIFICATE
repository=yuzbo/OpenTAD_OnlineTADClean_20260702
requested_branch=codex/full-petal-implementation

caller_target_literal=TARGET_COMMIT
caller_target_is_40_hex=false
formal_target_commit=UNRESOLVED
formal_target_tree=NOT_VERIFIED

remote_branch_head=48e78e43a77822ee0e51cace9ad758ca935c1450
predecessor_rejected_commit=48e78e43a77822ee0e51cace9ad758ca935c1450
branch_head_equals_rejected_predecessor=true
successor_commit_present=false

local_detached_checkout=NOT_CREATED
independent_target_manifest_rebuild=NOT_PERFORMED
independent_target_lf_scan=NOT_PERFORMED
old_results_or_checkpoints_read=false
training_or_gpu_or_profile_run=false
END_REPOSITORY_VISIBILITY_CERTIFICATE
```

GitHub 官方连接器将分支解析为 `48e78e43a77822ee0e51cace9ad758ca935c1450`，提交信息为 `Close Prefix Route V2 round-three bypasses`；这个 SHA 与调用者明确给出的 rejected predecessor 完全相同。换言之，远端分支上没有一个可与被拒版本区分的新 TARGET。

连接器的比较结果为：

```text
base=48e78e43a77822ee0e51cace9ad758ca935c1450
head=codex/full-petal-implementation
status=identical
ahead_by=0
behind_by=0
total_commits=0
changed_files=[]
```

因此，不允许把移动分支名隐式替换为 `TARGET_COMMIT`。正式 target commit、target tree 和 detached checkout 身份门均未成立。

### 1.2 当前可见的被拒快照

对当前分支头，也就是 rejected predecessor，静态可见信息为：

| 项目                        | 结果                                                                 |
| ------------------------- | ------------------------------------------------------------------ |
| Protocol Git blob SHA-1   | `b98eab4c8e367bb23e56569fd1b215c22587ca6f`                         |
| Policy lock SHA-256       | `c9594ff7ab2e89549e512f5528bddefe127ecda12d3c8b900f3e0e5d9bf598c0` |
| Protocol status           | `PROTOCOL_REVIEW_PENDING`                                          |
| Source registration       | `UNREGISTERED_BLOCK_POPULATION_R0_R1`                              |
| 最大 post-PASS scope        | `READ_ONLY_SOURCE_IDENTITY_REGISTRATION`                           |
| Route PASS authorizes GPU | `false`                                                            |

这些值来自提交中的 canonical one-line protocol。当前状态自身仍禁止 collection、training、GPU、model implementation 等活动。

Committed source manifest 的 Git blob SHA-1 为 `48d44ec0cf45975588013aa664786bee2e46c534`。但由于没有合法 TARGET SHA 和 detached worktree，本轮没有把它独立重建为 target-specific manifest certificate。

`.gitattributes` 确实规定 `.py/.json/.jsonl/.md/.sh` 等文本使用 LF；这只能证明仓库声明了 LF policy，不能替代对未解析 TARGET 的逐 blob、逐字节 LF 核验。

GitHub Actions 查询返回零条 workflow run。它既不阻断 commit 身份，也不构成测试通过证据。

---

# 2. Executive verdict

```text
EXECUTIVE_VERDICT: REVISE_PROTOCOL
PREVIOUS_VERDICT: UNCHANGED
PASS_ATTESTATION: NOT_GENERATED
```

裁决理由有两层，任一层都足以拒绝 PASS：

1. **快照门失败。** `TARGET_COMMIT` 仍是未替换占位符；远端分支没有新 successor，而是精确停留在 rejected predecessor。
2. **即使把当前分支头作为辅助审计对象，A–I 中仍存在多个可导致科学 false-PASS 的 P0。** 当前 `UNREGISTERED` 只是暂时让公共入口提前阻断；它不是下游证据链闭合。协议 PASS 最多授权 source identity registration，一旦注册完成，下述伪 emission、伪 optimizer ledger、序列化 fairness、伪 control transcript 等路径即会变成可达路径。

---

# 3. P0 findings

## P0-00 — 没有可审查的新固定 TARGET

**事实**

* `TARGET_COMMIT` 不是 40 位小写十六进制 SHA。
* 分支头就是已拒绝的 `48e78e43…`。
* 无 successor diff、无 target tree、无 target detached checkout。

**复现探针**

```bash
bash -lc '
  [[ "TARGET_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  { echo INVALID_OR_UNRESOLVED_TARGET; exit 64; }
'
```

结果：

```text
INVALID_OR_UNRESOLVED_TARGET
exit_code=64
```

在不可变对象未确定时，任何 PASS 都会变成对移动分支或旧 rejected snapshot 的签名，这是不可接受的身份替换。

---

## P0-01 — R6 把调用者 JSON emission 直接升级为 “immutable”，并明确关闭 ledger 验证

### 代码事实

`derive_per_video_cell` 接收普通 `emissions: list`，调用 `canonical_emission` 后，重新构造 metric row，并由验证器自身写入：

```python
"immutable": True
```

该值不是从 hash-chained emission ledger 重建，也没有外部 commitment。位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:260-309
```

R6 run schema 直接允许：

```text
run = {
    "arm": ...,
    "seed": ...,
    "provenance": ...,
    "videos": [
        {"video_id": ..., "emissions": [...]}
    ]
}
```

没有 `ledger_path`、`ledger_bytes`、`commitment_path` 或 verified ledger result。位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:1886-1957
```

随后 `_online_ap_from_raw` 主动调用：

```python
OnlineAPBudgeted(
    ...,
    require_ledger=False,
    include_identity_diagnostics=False,
)
```

位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:1789-1808
```

这并非 evaluator 不支持 ledger。`OnlineAPBudgeted` 的默认值本来就是 `require_ledger=True`。

仓库甚至已有真正的 immutable emission ledger：其 envelope 包含 `sequence`、`previous_hash`、`row_hash`，event 必须显式含 `immutable=true`、stream、frame 与 provenance 字段。

**主动攻击**

```python
run["videos"] = [{
    "video_id": "video_test_0000001",
    "emissions": [{
        "emission_id": "chosen:e0",
        "stream_key": "video_test_0000001",
        "sequence_id": 0,
        "start": 0,
        "end": 8,
        "class": "A",
        "score": 1.0,
        "source_frame": 7,
        "emit_frame": 8,
    }],
}]
```

不提供任何 emission ledger 或 commitment。当前代码会把它 canonicalize、写入 `immutable=True`，并使用 `require_ledger=False` 计算 OnlineAP。

**裁决：OPEN。**

---

## P0-02 — “optimizer ledger” 只是自写哈希表，不绑定事件 artifact bytes，也不能完整重放

### 缺失的必要字段

当前 optimizer event schema 只有：

```text
optimizer_event_index
gradient_accumulation_steps
effective_token_count
status
input_batches_sha256
model_state_before_sha256
model_state_after_sha256
optimizer_state_after_sha256
```

不存在：

* `optimizer_state_before_sha256`；
* input batch artifact reference/bytes；
* per-event model-before artifact；
* per-event model-after artifact；
* optimizer-before artifact；
* optimizer-after artifact；
* replay command/environment/RNG state；
* event-level immutable hash chain。

位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:1318-1370
```

验证器只检查这些字符串是否是 64-hex、相邻 `model_after == next model_before`，以及 before/after 字符串不相等。之后的 summary ledger 只把初始/最终 artifact、config、command、environment 和整个 trace JSON 做哈希；它没有证明任一 event hash 是从对应 artifact bytes 得到的。位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:1378-1496
```

Fairness live audit 也没有重放全部事件。`_runtime_budget` 遍历所有字符串记录，却只返回：

```python
dict(trace["events"][0])
```

给 live profiler。位置：

```text
opentad/utils/prefix_route_fairness_v2.py:186-327
```

`_measure_runtime` 随后只执行和核对这一个 `profile_event`。

更严重的是，作者测试用三个完全任意的哈希事件作为正向 fixture，并断言 `_runtime_budget` 接受 `optimizer_event_count=3`、`effective_token_count=300`。

**主动攻击**

```python
events = [
    {
        "optimizer_event_index": i,
        "gradient_accumulation_steps": 2,
        "effective_token_count": 100,
        "status": "APPLIED_FINITE",
        "input_batches_sha256": "a" * 64,
        "model_state_before_sha256": f"{i + 1:064x}",
        "model_state_after_sha256": f"{i + 2:064x}",
        "optimizer_state_after_sha256": "b" * 64,
    }
    for i in range(3)
]
```

只需保证相邻 model 字符串连起来；无需提供或重放对应输入、模型或 optimizer bytes。

**裁决：OPEN。**

---

## P0-03 — R6 接受 serialized fairness；验证过程中主动构造 `model=None, optimizer=None`

### 代码事实

`derive_fairness_audit` 的 live 入口本身是合理的：它要求实际 `ArmRuntimeAdapter`，并调用 Torch/CUDA live measurement。

但 R6 没有重新执行这个入口。它读取调用者提供的 fairness JSON，然后调用：

```python
validate_fairness_audit_record(...)
```

并只检查其 `status` 字符串。位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:2101-2111
```

这个 serialized validator 内部为每个 arm 主动创建：

```python
ArmRuntimeAdapter(
    arm=arm,
    model=None,
    optimizer=None,
    optimizer_event_forwards=lambda: None,
    inference_forward=lambda: None,
    reset_runtime_state=lambda: None,
    ...
)
```

然后只从作者提交的 budget JSON 重新计算计数。位置：

```text
opentad/utils/prefix_route_fairness_v2.py:760-779
```

所谓 gradient inventory 也只是验证调用者给出的 `name/numel/gradient_l1` 为正数，并没有 live parameter 对象与之绑定。

**零模型 fairness 探针**

* 对 `derive_fairness_audit` 直接传 `model=None`：会正确拒绝。
* 对 R6 实际使用的 `validate_fairness_audit_record`：构造自洽 serialized rows、任意正 profiler scalars、任意正 gradient inventory 和 hash-verified 自写 budget records，即可经过无模型验证路径。

这违反“serialized record、model=None、optimizer=None 均不得产生 PASS”。

**裁决：OPEN。**

---

## P0-04 — Control 只绑定任意哈希字符串，没有重新执行冻结算法

### 代码事实

`_validate_control_construction` 确实读取一个 canonical JSON transcript，但每个视频只有：

```text
video_id
source_sha256
constructed_sha256
```

验证器只要求两个值是合法 SHA-256 且不相等。它不读取对应 source artifact、constructed artifact，不根据冻结算法重新构造 control，也不进行 bytewise comparison。位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:1080-1210
```

实际冻结算法存在于：

```text
opentad/utils/prefix_route_controls_v2.py:39-64
opentad/utils/prefix_route_controls_v2.py:122-249
```

包括 feature-time shuffle 和 semantic derangement。

但 R6 control validator 没有调用这些算法。

作者的正向测试直接使用：

```python
"source_sha256": "c" * 64,
"constructed_sha256": "d" * 64,
```

并断言 `_validate_control_construction` 返回成功。

**主动攻击**

复制该测试，只把 control、algorithm、parameters 填成冻结值；source/constructed 内容完全不存在，仍可通过 control construction 层。

**裁决：OPEN。**

---

## P0-05 — `LEDGER_ONLY` 声明 no-learning，却被统一 schema 强迫伪造 optimizer updates

Protocol 代码明确规定：

```text
LEDGER_ONLY:
    input = same_raw_B0_candidates
    learning = False
```

然而 `_validate_run_provenance` 对所有 arm 一律要求：

* `events` 非空；
* 每个 event 为 `APPLIED_FINITE`；
* `model_state_before != model_state_after`；
* optimizer event count、token count、accumulation 都严格大于零。

所以：

```text
LEDGER_ONLY + zero optimizer events
```

会被拒绝；而：

```text
LEDGER_ONLY + arbitrary fake APPLIED_FINITE update hashes
```

反而满足 generic schema。

这不是小的 schema mismatch，而是协议要求 no-learning control 提交虚假“发生了学习”的证据。

**裁决：OPEN。**

---

## P0-06 — B2 的 risk set、cost matrix、assignment、target、transition 与 emission 未进入同一链

### 已实现但彼此分离的组件

B2 文件确实分别实现了：

* delayed un-emitted risk set 和 `first_emission_target`；
* 64 newborn × targets+64 dustbins assignment；
* decision coordinate 检查；
* temporal transition 与 emissions。

但是 `temporal_motr_transition` 的输入只有：

```text
previous_tracks
decoder_rows
threshold
decision coordinates
serials
```

它不接收、验证或提交：

* risk-set transcript；
* 64-dustbin cost matrix；
* Hungarian assignment；
* propagated/newborn GT assignment；
* first-emission training target；
* loss；
* optimizer event identity。

其返回 emission 也没有 assignment/risk/cost provenance commitment。

R6 对 B2 没有任何专用字段。B2 与其他 arm 一样，只提交普通 inline emissions。

**脱离 B2 transcript 的 emission 探针**

```python
run = {
    "arm": "B2",
    "seed": 705,
    "provenance": generic_provenance,
    "videos": [{"video_id": "v0", "emissions": chosen_emissions}],
}
```

不存在 risk set、matrix、assignment 或 transition transcript 字段，因此 R6 无从判断这些 emissions 是否由 B2 执行链产生。

**裁决：OPEN。**

---

## P0-07 — 手写完整 inference dict 可直接得到 terminal PASS

公共 R6 路径已经改为 literal 10,000 次 bootstrap，这一局部修改是成立的。

但 `_terminal_route_decision(inference)` 仍是可直接调用的 Python 函数。它只调用 `_validate_inference_schema`，后者检查：

* 字段集合；
* 固定 resample/seed/alpha 数值；
* metrics/contrasts 完整；
* 数值 finite；
* interval lower ≤ upper。

它不要求 raw-cell digest、bootstrap transcript、RNG transcript 或 sealed source capability，也不重新计算 interval。位置：

```text
opentad/evaluations/prefix_route_r6_v2.py:761-904
opentad/evaluations/prefix_route_r6_v2.py:907-1066
```

仓库测试本身提供 `_complete_r6_inference()`，完全手填 `arm_estimates` 和所有 intervals，作为 terminal 函数的正向输入。

**手写 inference PASS 探针**

在现有测试 helper 基础上：

1. `eligible_stress_families=["same_bin_end_start"]`；
2. 为所有 arms、contrasts 添加 `same_bin_end_start_recall`；
3. 所有 B4-vs-* interval 设为例如 `[0.10, 0.20]`；
4. margins 和 multiplicity 按 schema 填齐；
5. 调用：

```python
decision = r6._terminal_route_decision(forged_inference)
assert decision["status"] == "PASS_B4_ROUTE_SURVIVES"
```

该调用不需要一个 raw cell，更没有进行 10,000 次 bootstrap。

函数以下划线开头且未列入 `__all__`，但 Python 没有 private access control，仓库测试也已经直接调用它。仅隐藏名称不构成科学证据边界。

**裁决：OPEN。**

---

## P0-08 — 同一有效 MP4 可冒充多个独立视频身份

Population validator 的确增加了 OpenCV 解码检查：要求 MP4 可打开、第一帧可解码、frame count 和尺寸为正，并在解码后重新读取 bytes 防止 TOCTOU。

但 inventory 层只检查：

```python
source_id unique
canonical_id unique
```

没有：

* artifact SHA-256 全局唯一；
* artifact path 全局唯一；
* decoded-content fingerprint 唯一；
* canonical ID 与官方 per-video bytes manifest 的独立绑定；
* 显式合法 alias policy。

位置：

```text
opentad/utils/prefix_route_protocol_v2.py:2168-2238
```

**同一 MP4 多 ID 探针**

```text
vA.mp4 = valid_video_bytes
vB.mp4 = exact_same_valid_video_bytes

entry A:
  source_video_id=A
  canonical_video_id=A
  artifact_sha256=H

entry B:
  source_video_id=B
  canonical_video_id=B
  artifact_sha256=H
```

两个 ID 都唯一、两个文件都可解码、两个引用都与 inventory 自洽。当前验证器没有 `seen_artifact_hashes` 或官方 per-video bytes manifest 对比，因此会把同一视频内容计为两个独立视频。

现有视频 hostile test只验证 ASCII placeholder 不是可解码视频，没有覆盖“同一有效视频、多身份”的攻击。

**裁决：OPEN。**

---

# 4. P1 findings

## P1-01 — Source manifest 仍不是递归 import closure

Protocol validator 使用手写的 `mandatory_sources` 集合，再要求这些路径属于 `required_paths`。`load_source_manifest` 也只是遍历 manifest 已列出的路径并核对哈希；没有 AST/importlib 递归闭包重建。位置：

```text
opentad/utils/prefix_route_protocol_v2.py:1536-1563
opentad/utils/prefix_route_protocol_v2.py:1600-1638
```

正式 R1 extractor 会导入：

```python
from opentad.models.backbones.online_siglip_adapter \
    import OnlineSigLIPFrameEncoder
```

标准 Python import 会先执行：

```text
opentad/models/__init__.py
opentad/models/backbones/__init__.py
```

前者在 import time 导入 dense heads 和 detectors，并可进一步动态注册多个模型包。

后者在 import time 导入多种 backbone、adapter 及 wrapper。

但这两个 package initializer 不在手写 mandatory set，提交的 manifest entry set也没有递归补入它们及其 import closure。

**未绑定 import 探针**

修改一个未列入 manifest 的 `opentad/models/backbones/__init__.py`，例如在 import 时替换或包装 `OnlineSigLIPFrameEncoder`，保持所有 manifest-listed 文件不变。当前 `load_source_manifest` 没有路径可读该文件，因此 manifest gate不会发现变更。

**裁决：OPEN。**

---

## P1-02 — Hostile regressions 不完整，且若干正向 fixture 本身采用伪证据

现有测试包含以下科学上不允许作为正向 evidence fixture 的对象：

| Fixture                                   | 当前测试行为                             |
| ----------------------------------------- | ---------------------------------- |
| 任意 optimizer/input/model 哈希               | 作为 `_runtime_budget` 正向输入并断言计数成功。  |
| 任意 control source/constructed 哈希          | 作为 control construction 正向输入并断言成功。 |
| 手填 estimates/intervals 的完整 inference dict | 作为 terminal validator 正向输入。        |

以下要求的 hostile regressions 不存在或没有验证完整路径：

* same valid MP4 under multiple identities；
* formal R6 `require_ledger=False` 必须拒绝；
* every optimizer event artifact-byte replay；
* optimizer-before/after artifacts；
* B2 risk→matrix→assignment→target→transition→emission 一体 transcript；
* recursive package `__init__` import closure；
* LEDGER_ONLY zero-event schema；
* complete favorable hand-written inference 必须拒绝；
* serialized fairness record 必须不能独立产生 PASS。

**裁决：OPEN。**

---

# 5. P2 findings 与局部已闭合项

## P2-01 — LF 是声明性 policy，不是当前 TARGET 的独立 byte certificate

`.gitattributes` 的 LF 规则存在，但本轮没有合法 TARGET checkout，不能把该规则提升为 target-wide LF 复现结果。

## P2-02 — 无 CI corroboration

查询到的 workflow run 数为零。它不构成失败原因，但作者不能以 CI/test badge 作为当前提交通过的旁证。

## 局部真实修复

以下组件级工作是真实存在的，但不足以关闭端到端证据链：

* hash-chained immutable emission ledger utility；
* OnlineAP 默认 `require_ledger=True`；
* B2 augmented 64-dustbin assignment；
* B2 decision coordinate check；
* OpenCV 视频解码与 decode 后重新读取；
* 公共 R6 路径使用 literal 10,000 bootstrap 参数；
* population/R0 公共入口要求 path-backed protocol 和 signed review。

这些局部项不抵消 P0 integration bypass。

---

# 6. Round-4 closure matrix

| Requirement                                                                 |          状态 | 裁决依据                                                                               |
| --------------------------------------------------------------------------- | ----------: | ---------------------------------------------------------------------------------- |
| **A. R6 emission 只能由 immutable ledger 重建；OnlineAP require_ledger=True**     |    **OPEN** | R6 inline 接受普通 emissions，自写 `immutable=True`，且明确 `require_ledger=False`。           |
| **B. 每个 optimizer event 绑定 input/model/optimizer before-after bytes，可完整重放** |    **OPEN** | 只有字符串哈希；无 optimizer-before；无 event artifacts；只 live replay 第一个 event。              |
| **C. Fairness 必须 same-run live，serialized/model=None 不得 PASS**              |    **OPEN** | R6 使用 serialized validator；validator 内部主动创建 `model=None, optimizer=None`。          |
| **D. Controls 绑定 source/constructed bytes并重跑算法；no-learning 为零 event**       |    **OPEN** | 只检查任意不等哈希；不重跑算法；LEDGER_ONLY 被强迫伪造正 optimizer event。                                |
| **E. B2 全链一体可重放**                                                           |    **OPEN** | risk、assignment、transition 是分离函数；R6 只看 detached emissions。                         |
| **F. Terminal 从 source-bound raw cells执行 sealed 10,000 bootstrap**          | **PARTIAL** | 公共路径使用 sealed bootstrap；但直接 terminal helper可接受手填完整 inference 并输出 PASS。             |
| **G. Population 绑定独立官方 per-video manifest，拒绝重复 artifact identity**          |    **OPEN** | 有可解码检查；无 payload/hash/content identity uniqueness，也无独立官方 per-video bytes manifest。 |
| **H. Manifest 覆盖完整递归 import closure**                                       |    **OPEN** | 仍为手写路径集合；遗漏 import-time package initializers 及递归依赖。                                |
| **I. 所有 bypass 均有安全 hostile regression**                                    |    **OPEN** | 多项缺失；若干伪哈希/手填统计本身仍是正向 fixture。                                                     |

```text
CLOSED=0
PARTIAL=1
OPEN=8
```

---

# 7. Hostile-probe result register

由于没有合法 TARGET detached checkout，下表严格区分“本轮实际进程执行”与“由精确源码直接决定的结果”。

| Probe                       | 当前结果                                                                | 证据模式                               |
| --------------------------- | ------------------------------------------------------------------- | ---------------------------------- |
| 零模型 fairness                | **BYPASS**：live derive 拒绝，但 R6 serialized validator 使用 `model=None` | SOURCE-DETERMINISTIC               |
| 伪 optimizer ledger          | **ACCEPTABLE**：任意链式哈希和正计数可过结构层                                      | SOURCE + EXISTING POSITIVE FIXTURE |
| 同一 MP4 多 ID                 | **ACCEPTABLE**：无 artifact digest uniqueness                         | SOURCE-DETERMINISTIC               |
| 手写 inference PASS           | **ACCEPTABLE**：完整有利 dict可直接喂 terminal helper                        | SOURCE-DETERMINISTIC               |
| 任意 control hash             | **ACCEPTED**                                                        | EXISTING POSITIVE FIXTURE          |
| 脱离 B2 transcript 的 emission | **ACCEPTABLE**                                                      | SOURCE-DETERMINISTIC               |
| `require_ledger=False`      | **ACTIVE IN PRODUCTION R6 PATH**                                    | DIRECT SOURCE                      |
| 未绑定 import                  | **ACCEPTABLE AT MANIFEST GATE**                                     | SOURCE-DETERMINISTIC               |
| 篡改 source artifact          | 直接篡改 manifest-listed file会拒绝；篡改 omitted import-time source仍可通过      | PARTIAL BYPASS                     |
| no-learning 假训练账本           | zero-event会拒绝；伪 positive event反而满足 schema                           | DIRECT SOURCE                      |

---

# 8. Commands、退出码、测试数与 skip 原因

## 8.1 本地命令

| 命令                                                                              | 结果                                                                   |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `[[ "TARGET_COMMIT" =~ ^[0-9a-f]{40}$ ]]`                                       | exit `64`; `INVALID_OR_UNRESOLVED_TARGET`                            |
| `git ls-remote --heads <repository> refs/heads/codex/full-petal-implementation` | exit `128`; local container DNS：`Could not resolve host: github.com` |
| detached checkout                                                               | 未创建                                                                  |
| `pytest`                                                                        | 未调用                                                                  |
| GPU/CUDA/Slurm/profile/training                                                 | 按任务边界未调用                                                             |
| R0/R1 collection                                                                | 未调用                                                                  |
| checkpoint/result inspection                                                    | 未调用                                                                  |

## 8.2 GitHub 官方连接器

| 操作                                | 结果                                    |
| --------------------------------- | ------------------------------------- |
| Fetch branch ref                  | OK；解析为 `48e78e43…`                    |
| Fetch rejected commit             | OK                                    |
| Compare rejected commit vs branch | OK；`identical`, ahead=0, behind=0     |
| Fetch cited source files          | OK                                    |
| Fetch workflow runs               | OK；0 runs                             |
| Fetch `opentad/__init__.py`       | 404；该路径在仓库中不存在，未作为 closure finding 使用 |

## 8.3 测试统计

```text
local_pytest_invocations=0
passed=N/A
failed=N/A
skipped=N/A
```

不是把测试“视为跳过后通过”，而是**没有测试执行结果可报告**。原因是：

1. formal TARGET SHA 不存在；
2. branch head 是 rejected predecessor；
3. local detached checkout 无法建立。

在这种身份状态下运行旧分支测试也不能产生针对 TARGET 的证据。

---

# 9. 仍可产生科学 false-PASS 的最短攻击路径

## 9.1 最短代码级路径

```text
1. 手写满足字段与 fixed-constant 检查的 inference dict。
2. 将 eligible family 设为 same_bin_end_start。
3. 给 B4_vs_B2、B4_vs_A4、controls 等填入有利 intervals。
4. 调用 _terminal_route_decision(inference)。
5. 得到 status=PASS_B4_ROUTE_SURVIVES。
```

不需要 raw cells，不需要 bootstrap，不需要模型，不需要 ledger。

## 9.2 最短公共 R6 科学路径——在未来合法 registration 后

```text
1. 完成真实 source registration 和 signed protocol review。
2. 提交自组 serialized fairness JSON：
   任意正 profiler scalars、任意正 gradient inventory、作者自写 budget records。
3. 为每个 arm/seed 提交：
   非空任意 model artifacts；
   任意串联的 model/input/optimizer hash strings；
   自洽 summary execution ledger。
4. 对所有视频直接填写 GT-perfect inline emissions。
5. 对 controls 填任意不等 source/constructed hashes。
6. 对 LEDGER_ONLY 填伪 APPLIED_FINITE optimizer events。
7. R6 把 inline rows升级为 immutable，使用 require_ledger=False 计算指标。
8. sealed bootstrap 对伪造的“完美” raw rows产生真实但无来源的有利区间。
9. terminal 返回 PASS。
```

这里 bootstrap 数学本身可以完全正确；问题是其输入证据是调用者自组的。**对伪造 raw evidence 做正确统计，仍然是科学 false-PASS。**

---

# 10. 最小修复合同

以下全部是必要条件；不得以隐藏函数、增加布尔字段或继续依赖 `UNREGISTERED` 状态替代。

1. **发布真实 immutable successor。**
   `TARGET_COMMIT` 必须是明确 40-hex，且不得等于 `48e78e43…`；branch、commit、tree、protocol bytes、manifest bytes、LF 必须可在 detached checkout 独立复现。

2. **R6 移除 inline emission authority。**
   `videos[*].emissions` 必须替换为 ledger artifact 与 external commitment references。R6 必须调用 verified ledger loader；formal OnlineAP 固定 `require_ledger=True`。禁止验证器自行写入 `immutable=True`。

3. **Optimizer event 改为 artifact-backed replay schema。**
   每个 event 必须绑定：

   * input microbatch bytes；
   * model-before bytes；
   * model-after bytes；
   * optimizer-before bytes；
   * optimizer-after bytes；
   * RNG、AMP/scaler、command、environment；
   * previous-event hash 与 event hash。
     Verifier 必须逐事件恢复和重放全部 event，并比较实际状态 bytes或冻结的 canonical serialization。

4. **Fairness 与 run identity 合并。**
   Formal fairness 只能在同一次验证调用中由 live model、live optimizer 和完整 event replay产生。独立 serialized fairness JSON最多作为输出缓存，不能作为 PASS 输入。Fairness artifact必须绑定同一个 run identity和最终 emission ledger。

5. **Control 必须重新执行。**
   每个视频提供 source artifact、constructed artifact 和 frozen algorithm inputs。Verifier必须调用冻结 control implementation并逐字节比较 constructed output。仅提供 hash字符串不得通过。

6. **No-learning control 使用独立零事件 schema。**
   `LEDGER_ONLY` 必须要求 `optimizer_event_count=0`、无 model update、无 optimizer state transition。给它任何 `APPLIED_FINITE` update都应拒绝。

7. **B2 建立单一 per-decision transcript。**
   同一 hash-chained record必须包含：

   * previous state；
   * risk set；
   * 64-dustbin cost matrix；
   * assignment；
   * first-emission target；
   * loss；
   * optimizer event；
   * transition inputs/outputs；
   * emitted ledger rows。
     R6 必须逐 decision重放并确认 emission是该 transcript 的直接结果。

8. **Terminal 必须具有 source capability，而非 plain dict。**
   `_terminal_route_decision` 不得接受普通 mapping。可选择：

   * 在函数内部从 raw cells重跑 sealed bootstrap；或
   * 接收不可伪造的内部 sealed object，其构造器验证 raw-cell digest和完整 bootstrap transcript。
     仅改名、移除 `__all__` 或加 `_private` 前缀不算修复。

9. **Population 增加独立 official per-video manifest。**
   对每个 canonical ID绑定官方文件 identity、bytes digest、可解码 fingerprint。默认拒绝重复 artifact hash、重复 path 和重复 decoded-content fingerprint；若存在合法 alias，必须在冻结协议中逐项注册和解释。

10. **Manifest 自动重建递归本地 import closure。**
    必须覆盖 package `__init__`、函数内 import、adapter、tool、动态 registration 及所有本地 transitive imports。审查器从 entry points独立重建 closure，并要求其与 manifest path set精确相等。

11. **补齐 hostile regressions。**
    本轮列出的十类攻击必须一项一 test，且每项断言 formal public path fail-closed。正向 fixture必须使用：

    * 真实最小可解码视频；
    * 真实 ledger/commitment；
    * 真实可重放 optimizer artifacts；
    * 真实 control source/constructed bytes；
    * 由 raw cells实际运行的 bootstrap。
      任意哈希、伪 ledger、伪视频、手写 estimates/intervals 不得作为科学 PASS fixture。

---

# 11. Attestation 与授权边界

```text
PASS_ATTESTATION_GENERATED=false
reason=UNRESOLVED_TARGET_AND_OPEN_P0_P1_FINDINGS

CURRENTLY_AUTHORIZED_SCOPE=NONE
COLLECTION_AUTHORIZED=false
MODEL_IMPLEMENTATION_AUTHORIZED=false
PROFILE_AUTHORIZED=false
TRAINING_AUTHORIZED=false
GPU_AUTHORIZED=false
```

即使未来全部 P0/P1 关闭并获得 `PASS_PROTOCOL`，本轮规定的最大授权也只能是：

```text
READ_ONLY_SOURCE_IDENTITY_REGISTRATION
```

不能自动授权 collection、model implementation、profile、training 或 GPU。当前提交尚未达到这一最低授权门槛。


# Prefix Route Evidence Protocol V2 — Fail-Closed 复审结果

## 1. Repository / commit verification

### 1.1 身份门证书

| 字段                            | 结果                                                         |
| ----------------------------- | ---------------------------------------------------------- |
| Repository                    | `yuzbo/OpenTAD_OnlineTADClean_20260702`，仅作为调用者提供的远端标识      |
| Context branch                | `codex/full-petal-implementation`，**可移动，仅可导航**             |
| Caller-supplied target        | `TARGET_COMMIT`                                            |
| Target 是否为 40 位 Git object ID | **否**                                                      |
| 可绑定的 immutable commit         | **无**                                                      |
| Detached checkout             | **未建立；身份门阻断**                                              |
| Review tree                   | **UNBOUND**                                                |
| Protocol bytes                | **UNBOUND**                                                |
| Policy-lock bytes             | **UNBOUND**                                                |
| Manifest bytes                | **UNBOUND**                                                |
| LF 检查                         | **NOT RUN**                                                |
| Predecessor                   | `48e78e43a77822ee0e51cace9ad758ca935c1450`，格式有效，但不能唯一确定其后继 |
| 远端分支头是否可替代 TARGET_COMMIT      | **否**                                                      |
| 最终身份状态                        | **FAIL_CLOSED_AT_IMMUTABLE_TARGET_GATE**                   |

`TARGET_COMMIT` 是未展开的模板占位符，不是 Git SHA。即使独立解析当前远端分支头，也只能得到一个 **reviewer-discovered candidate**，不能把它伪装成调用者冻结的目标提交。否则审查对象将由审查员临时选择，而不是由审查请求预先承诺。

此外，仅给出被拒绝提交也不能唯一推导待审提交：该提交可以存在多个后继、非直接后继、合并后继，分支也可能被 force-push。因而不能用“分支上看起来最新的后继”补全缺失的审查身份。

### 1.2 可复现身份探针

执行的确定性预检为：

```bash
python3 - <<'PY'
import re
import sys

target = "TARGET_COMMIT"

if re.fullmatch(r"[0-9a-f]{40}", target) is None:
    print("INVALID_TARGET_COMMIT_LITERAL")
    sys.exit(64)

sys.exit(0)
PY
```

结果：

```text
INVALID_TARGET_COMMIT_LITERAL
exit_code=64
```

依据 fail-closed stop-chain，以下操作不得继续：

```bash
git fetch --no-tags origin "${TARGET_COMMIT}"
git cat-file -e "${TARGET_COMMIT}^{commit}"
git checkout --detach "${TARGET_COMMIT}"
git rev-parse HEAD^{tree}
git diff 48e78e43a77822ee0e51cace9ad758ca935c1450.."${TARGET_COMMIT}"
```

它们的状态是 **NOT_RUN_AFTER_IDENTITY_FAILURE**，而不是 PASS、SKIP-PASS 或“通过分支头间接完成”。

---

# 2. Executive verdict

## **REVISE_PROTOCOL**

更精确地说：

```text
REVISE_PROTOCOL
blocking_layer: immutable review-object registration
code-level closure verdict: NOT ADMISSIBLE
previous verdict superseded: NO
previous verdict remains: REVISE_PROTOCOL_BEFORE_COLLECTION
PASS attestation: NOT ISSUED
authorization: NONE
```

这不是认定目标代码一定错误，而是认定当前请求没有提供可审计的目标代码对象。没有 commit/tree 绑定时，任何关于 A–I 已关闭、某项测试通过、某个文件某行正确的陈述，都可能属于审查员针对错误快照制造的科学 false-PASS。

---

# 3. Findings

## P0-01 — 缺失不可变目标提交

**状态：OPEN / BLOCKING**

调用者字段是：

```text
Target commit:
TARGET_COMMIT
```

该值不满足：

```regex
^[0-9a-f]{40}$
```

因此无法证明：

* commit object 的真实存在；
* detached checkout 的 `HEAD`；
* commit tree；
* 与拒绝提交的直接父子或祖先关系；
* protocol、policy lock、manifest 的确切字节；
* 行号所对应的版本；
* 测试究竟执行于哪个代码对象；
* hostile regression 是否属于目标提交。

**精确 `file:line`：不适用。**
在 commit/tree 尚未绑定时提供源文件行号，反而会违反本轮“固定 Git 提交”的证据合同。任何行号都会隐含一个未经调用者注册的快照选择。

### 可复现攻击

```bash
# 错误审查实现
TARGET="$(git ls-remote origin \
  refs/heads/codex/full-petal-implementation | awk '{print $1}')"

# 审查员随后把 TARGET 当成调用者提供的目标
git fetch origin "$TARGET"
git checkout --detach "$TARGET"
```

此流程可以运行，但不能证明该 SHA 就是用户意图审查的 `TARGET_COMMIT`。

---

## P0-02 — 移动分支替代固定提交会形成最短身份 false-PASS

**状态：OPEN / BLOCKING**

以下替代是不允许的：

```text
TARGET_COMMIT := current remote branch head
```

原因不是分支头一定有问题，而是两者的信任语义不同：

* `TARGET_COMMIT=<40-hex>`：调用者在审查开始前绑定对象；
* `branch head`：审查员在某个时刻观察到的可移动状态。

如果允许静默替代，则即使 A–I 的代码逻辑全部正确，也可以产生以下错误证明：

1. 作者意图提交为 `X`，但未写入请求；
2. 审查员观察分支得到 `Y`；
3. 审查员审核 `Y` 并签发针对“TARGET_COMMIT”的 PASS；
4. 报告没有证明 `X = Y`；
5. 后续 registration 或 collection 使用 `X`、`Z` 或移动后的分支头；
6. PASS 与正式执行对象脱钩。

这是比伪 ledger、伪视频或手填 bootstrap 更短的攻击路径，因为它在所有科学验证之前就破坏了 source identity。

---

## P0-03 — 被拒绝前序提交不能充当目标选择规则

**状态：OPEN / BLOCKING**

`48e78e43a77822ee0e51cace9ad758ca935c1450` 只标识被拒绝对象。它没有定义：

* 待审提交是否必须是其直接单亲后继；
* 是否允许中间修复提交；
* 是否允许 merge commit；
* 应审核哪个 descendant；
* 对比范围应为 `predecessor..target` 还是 `predecessor...target`；
* 分支发生重写时应采用哪个历史。

因此，不能通过寻找“某个看起来包含修复的后继”来补全 TARGET。

---

## P0-04 — `file:line`、测试与 hostile-probe 证据在当前状态下均不可注册

**状态：OPEN / BLOCKING**

文件行号是 `(commit tree, path, line range)` 的三元组，而不是单纯的 `path:line`。缺少第一项时：

* 相同行号可对应不同字节；
* 测试名可对应不同测试实现；
* fixture 可在后继提交中被改写；
* manifest closure 可随 `__init__.py`、adapter 或动态 import 变化；
* hostile regression 可以在移动分支上通过、在待审对象中不存在。

因此本轮没有把任何测试通过、源码模式匹配或当前分支内容纳入正式证据。

---

## P1 / P2

没有可在 P0 身份门之前合法裁决的代码级 P1/P2 finding。

这不等于 P1/P2 为零，也不等于 A–I 已通过；其含义只是：

```text
P1_count_admissibly_reviewed = 0
P2_count_admissibly_reviewed = 0
reason = P0 immutable-target failure
```

---

# 4. Round-4 closure matrix

由于没有 source-bound target，前一轮结论不能自动继承为闭合证据。所有项目均保持 OPEN，而不是 PARTIAL 或 CLOSED。

| 项目                                                                            | 状态       | 本轮可接受证据 | 阻断原因                                         |
| ----------------------------------------------------------------------------- | -------- | ------- | -------------------------------------------- |
| A. R6 只能从 immutable hash-chained ledger 重建；OnlineAP 强制 `require_ledger=True`  | **OPEN** | 无       | 无 target tree，无法确认实现、默认值、调用链和 hostile test   |
| B. optimizer event 绑定真实 input/model/optimizer before-after bytes 并可全量重放       | **OPEN** | 无       | 无法绑定 schema、artifact verifier 与 replay code  |
| C. fairness 必须同 run-bound live replay；拒绝 serialized/model=None/optimizer=None | **OPEN** | 无       | 无法核验生产路径和零模型绕过                               |
| D. control 绑定 source/constructed bytes 并重执行算法；no-learning 为零 optimizer event  | **OPEN** | 无       | 无法核验 control registry、算法执行和事件 schema         |
| E. B2 risk set、64-dustbin、assignment、target、transition、emission 同链            | **OPEN** | 无       | 无法核验 transcript schema 与 emission provenance |
| F. terminal 从 source-bound raw cells 重做固定 10,000 bootstrap                    | **OPEN** | 无       | 无法核验 bootstrap 实现及手填统计拒绝逻辑                   |
| G. 独立官方 per-video population manifest；真实解码；禁止重复 hash 多身份                      | **OPEN** | 无       | 无法核验 population、decoder 和 uniqueness checks  |
| H. manifest 覆盖递归 import closure，包括 `__init__` 与 adapter                       | **OPEN** | 无       | 无 commit tree，无法重建 import graph              |
| I. 每个绕过均有 hostile negative regression，且无伪 fixture 正向洗白                        | **OPEN** | 无       | 无法核验测试字节、fixture 或 mutation sensitivity      |

不得将上表解释为“代码已发现九项缺陷”；它表示九项均未获得与不可变源码对象绑定的闭合证明。

---

# 5. Commands、退出码、测试数和 probe 状态

## 5.1 命令台账

| 序号 | 操作                                       | 状态        |    退出码 |
| -: | ---------------------------------------- | --------- | -----: |
|  1 | 验证 `TARGET_COMMIT` 是否为 40 位小写十六进制        | 执行        | **64** |
|  2 | 从远端 fetch 固定 commit object               | 未执行：身份门失败 |    N/A |
|  3 | `git cat-file -e <sha>^{commit}`         | 未执行       |    N/A |
|  4 | 建立独立 detached checkout                   | 未执行       |    N/A |
|  5 | 核验 commit/tree/parent/branch containment | 未执行       |    N/A |
|  6 | 计算 protocol/policy/manifest byte hashes  | 未执行       |    N/A |
|  7 | 检查 CRLF/LF 和工作树污染                        | 未执行       |    N/A |
|  8 | 独立重建递归 import closure                    | 未执行       |    N/A |
|  9 | 收集 Prefix Route 相关测试                     | 未执行       |    N/A |
| 10 | 执行 CPU-only hostile regressions          | 未执行       |    N/A |

## 5.2 测试统计

```text
pytest_invocations = 0
tests_collected = 0
tests_run = 0
tests_passed = 0
tests_failed = 0
tests_skipped_by_pytest = 0
tests_blocked_by_review_stop_chain = ALL
```

这里的零测试不是“没有测试失败”，更不是 PASS。它表示测试证据链没有获得运行资格。

## 5.3 主动攻击探针结果

| 探针                          | 结果                 | 原因                                |
| --------------------------- | ------------------ | --------------------------------- |
| 零模型 fairness                | **NOT_RUN / OPEN** | 无 source-bound fairness 实现        |
| 伪 optimizer ledger          | **NOT_RUN / OPEN** | 无 source-bound schema/verifier    |
| 同一 MP4 多 video ID           | **NOT_RUN / OPEN** | 无冻结 population manifest 与 decoder |
| 手写 inference PASS           | **NOT_RUN / OPEN** | 无 source-bound terminal evaluator |
| 任意 control hash             | **NOT_RUN / OPEN** | 无 source-bound control registry   |
| 脱离 B2 transcript 的 emission | **NOT_RUN / OPEN** | 无 source-bound B2/R6 chain        |
| `require_ledger=False`      | **NOT_RUN / OPEN** | 无 source-bound OnlineAP 调用链       |
| 未绑定 import                  | **NOT_RUN / OPEN** | 无 target import closure           |
| 篡改 source artifact          | **NOT_RUN / OPEN** | 无冻结 artifact verifier             |
| no-learning 假训练账本           | **NOT_RUN / OPEN** | 无冻结 control/event schema          |

这些 probe 没有被记为 skip-pass，也没有利用当前分支中的任意测试 fixture替代独立攻击。

---

# 6. 当前仍可产生科学 false-PASS 的最短攻击路径

```text
TARGET_COMMIT 保持为占位符
    ↓
审查员自行读取移动分支头 Y
    ↓
在报告中省略“Y 是审查员选择而非调用者注册”
    ↓
对 Y 的测试或静态检查给出 PASS
    ↓
正式 registration/collection 使用 X、Z 或更新后的分支头
    ↓
PASS 与执行源码不一致
```

该路径只需一次未经授权的“分支头替代 commit”操作，不需要伪造 ledger，也不需要突破 bootstrap、population 或 optimizer replay。因此，在 source identity 关闭前审查 A–I，不能消除最短 false-PASS。

---

# 7. 最小修复合同

不得通过弱化“固定提交”要求来修复。最小可接受输入是：

```text
Repository:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702

Context branch:
codex/full-petal-implementation

TARGET_COMMIT:
<精确的 40 位小写 Git SHA>

PREDECESSOR_REJECTED_COMMIT:
48e78e43a77822ee0e51cace9ad758ca935c1450

RELATION_CONTRACT:
DIRECT_PARENT
```

若目标不是直接父子修复，则必须将最后一项明确写为：

```text
RELATION_CONTRACT:
DESCENDANT

EXPECTED_DIFF_BASE:
48e78e43a77822ee0e51cace9ad758ca935c1450
```

建议同时外部冻结以下注册字段，以免仓库内部 lock 文件自我声明真实性：

```text
EXPECTED_TREE_SHA=<40-hex>
PROTOCOL_PATH=<repo-relative path>
EXPECTED_PROTOCOL_SHA256=<64-hex>
POLICY_LOCK_PATH=<repo-relative path>
EXPECTED_POLICY_LOCK_SHA256=<64-hex>
MANIFEST_PATH=<repo-relative path>
EXPECTED_MANIFEST_SHA256=<64-hex>
```

后续复审必须从新的空目录执行：

```bash
git init audit-repo
cd audit-repo
git remote add origin \
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702.git

git fetch --no-tags origin "${TARGET_COMMIT}"
git checkout --detach "${TARGET_COMMIT}"

test "$(git rev-parse HEAD)" = "${TARGET_COMMIT}"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

在此之后才允许进入：

1. commit/tree/parent 与字节核验；
2. LF、symlink、submodule、工作树和来源分类检查；
3. 独立 manifest 重建；
4. 完整静态与运行时 import closure；
5. A–I 逐项源码审计；
6. 独立 hostile probes 与 mutation sensitivity；
7. CPU-only 测试计数与失败证据；
8. 最终 closure matrix 和可能的 attestation。

---

# 8. PASS attestation

```text
PREFIX_ROUTE_V2_PASS_ATTESTATION
status: NOT_ISSUED
reason: P0 immutable target identity is OPEN
p0_closed: false
p1_closed: not_evaluable
authorized_scope: NONE
collection_authorized: false
training_authorized: false
gpu_authorized: false
profiling_authorized: false
model_implementation_authorized: false
read_only_source_identity_registration_authorized: false
END_ATTESTATION
```

即使未来所有 P0/P1 均关闭，PASS 也最多授权：

```text
READ_ONLY_SOURCE_IDENTITY_REGISTRATION
```

本轮由于不可变目标对象尚未注册，连该最小授权也不成立。
