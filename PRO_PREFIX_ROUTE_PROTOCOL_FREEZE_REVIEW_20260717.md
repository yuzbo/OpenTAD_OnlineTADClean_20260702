# A. Executive Verdict

## 最终裁决：`REVISE_PROTOCOL_BEFORE_COLLECTION`

**置信度：0.96**

固定提交已经正确撤销 DR‑045，把 R‑A 从默认方法降为 B4 候选，并把 Q2/R1、模型实现、模型 P0、真实数据效果实验、GPU profile、正式训练、视觉微调和 raw-video 训练继续保持阻断。这个治理方向是正确的。DR‑046 也正确地把下一问题改写为：持续载体是否产生无法被无身份集合或 temporal MOTR 重建的、可由正式输出观察到的实例一致性收益。

但是，当前 `prefix-route-identifiability-gate-20260717.md` 仍是一份**高质量检查清单**，不是可在看到证据前执行的冻结协议。它反复使用以下尚未定义的判断词：

* “too rare for a powered main claim”；
* “matched parameter count”；
* “fail readiness”；
* “equivalent amount”；
* “equivalence region”；
* “where the model family permits”；
* “hidden structural OOD”。

这些词在看到 R0 普查结果或 R1 缓存证书后仍允许改变统计单位、阈值、主子集、比较臂或等价边界，会形成新的路线选择自由度。当前 R0–R6 只冻结了**需要收集什么类别的证据**，尚未冻结**怎样计算、怎样判断以及失败后怎样处置**。

因此当前授权矩阵为：

| 动作                              | 本轮裁决                                |
| ------------------------------- | ----------------------------------- |
| 修改并提交 R0–R6 精确证据协议              | `ALLOW`                             |
| 运行新 R0 标注普查                     | `BLOCKED_PENDING_PROTOCOL_REVISION` |
| 运行新 R1 缓存因果审计                   | `BLOCKED_PENDING_PROTOCOL_REVISION` |
| 冻结 B0–B4 模型定义                   | `BLOCKED`                           |
| 编写 B0–B4 模型代码                   | `BLOCKED`                           |
| 恢复历史 R‑A P0                     | `BLOCKED`                           |
| 访问模型预测、checkpoint 或效果报告         | `BLOCKED`                           |
| 真实数据效果实验                        | `BLOCKED`                           |
| GPU/profile/formal/raw-video 工作 | `BLOCKED`                           |
| GPU 授权                          | **0 小时**                            |

本裁决不是 `BLOCK_REPOSITORY_OR_DATA_UNVERIFIABLE`：固定 GitHub 快照和相关代码均可核验。也不是 `KILL_PERSISTENT_CARRIER_ROUTE`：数据可测性和简单基线充分性仍是未知项，尚不足以永久杀死整个持续载体家族。

---

# B. Repository/commit verification

## B.1 固定快照

| 项目         | 核验结论                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------ |
| Repository | `[代码核验]` `yuzbo/OpenTAD_OnlineTADClean_20260702`                                           |
| 固定提交       | `[代码核验]` `b70baf3bb5d819d1df25ca47534c5d3bfbb6e695`                                        |
| 提交信息       | `[代码核验]` `Record R-A preimplementation route revision`。                                    |
| 快照使用       | `[代码核验]` 所有文档、配置和代码读取均显式指定上述 SHA，没有使用默认分支或移动 HEAD。                                         |
| 与前一提交关系    | `[代码核验]` 相对 `f9419612…` 只前进一个提交；变更集中在审查文档、wiki、路线 gate 和 MOTR/TadTR 文献节点，没有新增或修改 R‑A 模型实现。 |
| 执行纪律       | `[代码核验]` 本轮没有修改仓库、创建 commit/PR、运行脚本、读取模型结果或使用 GPU。                                         |

核心文件的固定 blob 已成功解析：

* 路线审查：blob `6df429…`；
* 独立吸收：blob `2fe497…`；
* route-identifiability gate：blob `642787…`；
* 已撤回 P0：blob `4cec4d…`；
* source map：blob `0bb019…`。

## B.2 决策链一致性

* `[实验核验]` DR‑044 继续维持 Q2/R1 terminal KILL。历史绑定记录为：实际 `5517 emissions / 2206 exhaustions`，`-2` arm 为 `10/0`，其中 seeds 705/706 完全静默，因此零 exhaustion 是退化的 birth suppression，而不是共享生命周期修复。
* `[代码核验]` DR‑045 已明确标为被 DR‑046 supersede。
* `[代码核验]` T39 记录 R‑A 默认地位被撤销，R‑A 只能作为 B4，路线证据先于任何模型合同。
* `[代码核验]` 历史 R‑A P0 已标记为 `withdrawn-before-preregistration`、未实现且未运行；旧 seeds、margin 和 synthetic families 只是历史记录，不是当前可执行合同。

## B.3 当前数据与模型范围

当前 Q2 配置仍指向外部 THUMOS 标注、class map、split manifest 和缓存特征：

* development universe：200 个 training 视频；
* fit-core/calibration：160/40，seed `20260713`；
* historical reporting：211；
* canonical expected reporting：213；
* feature stride：8；
* feature dimension：768；
* chunk size：64；
* 旧 Q2 `K=4`、hidden 256、memory 192。

特别重要的是，配置同时保留 `historical_count=211` 和 `canonical_expected_count=213`，并承认 reporting population 已经有历史项目暴露、不是 untouched test set。当前 R0 协议没有决定普查和未来报告究竟绑定 211 还是 213，这是一个直接阻断项。

---

# C. 对任务价值、领域 gap 和数据可测性的最强反方论证

## C.1 最强反方

> 标准 On‑TAL 是一个合法但已经相当拥挤的任务。Q2 的失败由一个局部且可精确定位的训练顺序错误造成：在当前 token 的模型 decode/release 之前读取旧 hard runtime availability，再用它过滤 GT birth eligibility；未分配 birth 又被永久标为 born/exhausted。只要使用普通 per-prefix set matching、普通 persistent query 或 temporal MOTR，并确保 GT matching 不修改运行状态，这一错误即可消失。没有证据证明需要 continuous occupancy、UOT、transport consistency、model-only latch 或新的“潜在事件过滤器”。THUMOS 上同类重叠又可能极少，因此 R‑A 可能是在一个不支持其核心主张的数据集上，把 TrackFormer/MOTR 的轨迹 query 重命名为 carrier。

这条反方目前**成立到足以阻止证据收集协议直接 PASS**。

## C.2 标准 On‑TAL 是否仍有价值？

`[文献核验]` On‑TAL 的任务约束是有意义的：CAG‑QIL 将其定义为在流式视频中不访问未来帧，并且不能用后处理回改过去 proposal 的实例级动作定位。([CVF Open Access][1])

`[文献核验]` 该领域仍在发展：

* OAT 已处理在线 anchor 和重复 proposal 抑制；([ECVA][2])
* SimOn 表明简单的当前帧 query 加过去上下文即可形成强 On‑TAL 基线；([arXiv][3])
* MATR 已使用长期记忆，从当前片段估计终点并从历史记忆估计起点；([ECVA][4])
* HAT 已占据长短期历史增强；([ECVA][5])
* ActionSwitch 已明确针对并发和同类重叠；([ECVA][6])
* 2026 年 OZ‑TAL 和 OnPoint 又分别扩展到 zero-shot 和 point-supervised online TAL。([arXiv][7])

这些事实证明**任务仍有研究活动**，但同时削弱了“记忆、持续状态、重叠处理或逐时刻实例输出本身就是 gap”的说法。

### 裁决

* **任务价值：`CONDITIONALLY_ESTABLISHED`**
* **持续载体领域缺口：`NOT_ESTABLISHED`**
* **R‑A 必要性：`NOT_ESTABLISHED`**

## C.3 Q2 是领域失败还是局部 bug？

`[代码核验]` Q2 训练循环在每个 token 中：

1. 从旧 `state.slot_status` 中读取 FREE slots；
2. 执行 `head.step`；
3. 使用步骤 1 的 slots 运行 supervision transition；
4. 计算 loss；
5. 最后才执行 `decode_step`。

`[代码核验]` supervision transition 又把运行状态允许的槽与 canonical-free 槽相交形成 birth candidates；随后无论是否成功分配，全部 birth IDs 都被加入 `born_instance_ids`，未分配者计为 exhaustion。

这足以证明：

* Q2 有真实且严重的合同错误；
* 该错误不是简单 off-by-one；
* 但它只杀死“hard runtime availability 控制 GT canonical ownership”的方法族；
* 它不证明所有普通集合预测、普通持续 query 或 temporal MOTR 都会失败。

## C.4 数据是否支持持续实例一致性主张？

`[代码核验]` 当前 evaluator 观察的是最终不可变 emissions：区间、类别、分数、提交时刻、source 时刻。duplicate 和 fragmentation 通过最终区间与 GT 的确定性匹配定义；内部 query/carrier 是否交换不是正式结果。

因此：

> 内部 identity 只有在它改变 duplicate、fragmentation、same-class recall、事件数、延迟或 mOnlineAP 时才有科学意义。

`[实验核验：历史暴露]` 旧 THUMOS 普查曾报告：

| Split      |  视频 |    实例 | 有任意重叠的视频 | 有同类重叠的视频 |
| ---------- | --: | ----: | -------: | -------: |
| training   | 200 | 3,003 |       23 |    **2** |
| validation | 211 | 3,325 |       32 |    **3** |

最大总并发和同类并发均为 2。

这不是合格的新 R0：

* 使用的是旧全 training/validation population，而不是当前 160/40/213 路线角色；
* 脚本只统计最大并发、视频级 overlap 和 duration；
* 没有重复间隔、pair overlap duration、same-bin end/start、短动作、空视频、class map hash、split hash或盲性报告；
* 其结果已经暴露，不能再把相同字段称为“未见 outcome”。

旧脚本的实际输出范围确实只覆盖 duration、最大并发、同类最大并发和 per-video instance count。

### 裁决

* THUMOS 对一般 event recall、duplicate 和 fragmentation：**可测**；
* 对“同类重叠是主问题”：**现有暴露强烈提示样本不足**；
* 对“内部 carrier identity”：**不可直接测，只能由正式输出间接测**；
* 对 R‑A 是否值得做主方法：**必须等待精确 R0，当前为 UNKNOWN**。

---

# D. R0–R6 逐项裁决

| Gate                             | 协议裁决      | 已经做对的部分                                                                 | 当前阻断                                                                                     |
| -------------------------------- | --------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **R0 标注普查**                      | `REVISE`  | 已列出 split/hash、重复、重叠、same-bin、短动作、时长、并发、空视频等正确对象。                       | 没有精确定义、211/213 决策、统计单位、置信区间、主张资格阈值、既有暴露登记和 outcome-blind 执行规则。                           |
| **R1 缓存严格因果证书**                  | `REVISE`  | 正确指出 source-frame 单调不等于 encoder 无未来，要求 encoder/weight/support set。      | 没有证书 schema、权重/processor/raw-video hash、静态支持证明、动态扰动算法、现有 cache 无法验证时的处置。                 |
| **R2 B0–B4 公平合同**                | `REVISE`  | 五臂覆盖无身份、普通 persistence、MOTR、局部修复和 R‑A。                                  | B1/B2/B3 仍可能重叠；common head、参数匹配、调参预算、threshold calibration、token/update equality 没有精确规则。 |
| **R3 负对照**                       | `REVISE`  | 控制组方向完整。                                                                | 各控制组输入权限和构造未定义；“must fail readiness”没有指标和等价边界；threshold sweep 被错误地列为模型负对照。               |
| **R4 机制删除**                      | `REVISE`  | 能覆盖 R‑A 大部分候选部件。                                                        | 删除项过多且未区分核心机制、输出头和 stress test；必要性判断没有统计规则。                                              |
| **R5 Structural OOD**            | `REVISE`  | 正确认识到只换 random seed 不足。                                                 | 没有 generator grammar、训练支持集、隐藏组合、样本量、独立生成者、内容哈希和 PASS/KILL 规则。                            |
| **R6 temporal-MOTR exact delta** | `REVISE`  | 正确把 TrackFormer/MOTR 设为 novelty killer，并排除重命名、interval head 和共享 ledger。 | 当前没有 actual delta table；没有数值等价区域；没有明确哪一项是 carrier route 核心。                              |
| R0/R1 实际结果                       | `UNKNOWN` | 尚未运行新的合格证据收集。                                                           | 外部 annotation、split artifact、cache、weight snapshot 和 raw media 不在固定 GitHub commit 中。     |

当前没有任何一个 R0–R6 应被标为 PASS，也没有足够证据把整个 carrier route 标为永久 KILL。

---

# E. 所有缺失定义、统计量、哈希、阈值和防泄漏规则

## E.1 R0 必须冻结的精确定义

### E.1.1 数据人口

在运行 census 之前必须唯一决定：

1. `development_universe_200`：THUMOS `training` 的 200 视频；
2. `fit_core_160`；
3. `calibration_40`；
4. 正式 reporting population 是：

   * canonical 213；或
   * historical 211。

当前不能同时保留 211 和 213，再在看到统计或未来模型结果后选择。推荐冻结：

> `reporting_population = canonical_213`；
> `historical_211` 只作为差异审计，不得成为可选择的第二主结果人口。

当前 split builder 已能绑定 200、160/40、seed、raw/canonical annotation hash 和 exact ID lists。

### E.1.2 时间和 decision-bin

建议冻结以下唯一口径：

[
\alpha_v=\frac{F_v}{D_v},\qquad
s_i^{(f)}=\alpha_v s_i^{(sec)},\qquad
e_i^{(f)}=\alpha_v e_i^{(sec)}.
]

其中 (F_v) 是 annotation 中 frame count，(D_v) 是 duration。

对于 stride (q=8)，每个 token 的 decision frame 为：

[
d_j=\min((j+1)q,F_v)-1 .
]

这与当前缓存代码选择每个 packet 最后一个 0-based frame 的规则一致。

事件时间 (x) 的 decision bin 定义为：

[
b_v(x)=\min{j:x\le d_j}.
]

prefix schedule 必须继续采用：

[
d_{j-1}<s_i\le d_j \Rightarrow birth,
]
[
s_i\le d_j<e_i \Rightarrow active,
]
[
d_{j-1}<e_i\le d_j \Rightarrow end.
]

这与现有 schedule 代码一致。

### E.1.3 区间与重叠

所有标注按半开区间 ([s,e)) 解释。接触但不交叠，即 (e_i=s_k)，不算 overlap。旧脚本也采用同 timestamp 上 end 先于 start 的规则。

成对重叠长度：

[
o(i,k)=\max\bigl(0,\min(e_i,e_k)-\max(s_i,s_k)\bigr).
]

必须分别报告：

* same-class overlap pairs；
* cross-class overlap pairs；
* overlap GT instances；
* overlap videos；
* overlap duration / union duration；
* 每个视频的 pair count，不能把 pair 当作独立统计样本。

### E.1.4 重复、same-bin 与短动作

唯一冻结定义：

* **same-class repeated**：同视频同类至少两个实例；
* **sequential gap**：同类实例按 start 排序后相邻实例
  [
  g=e_i-s_{i+1}
  ]
  更直观地报告 (s_{i+1}-e_i)，负数为 overlap，0 为 touching；
* **same-bin end/start**：不同实例 (i\ne k) 满足
  [
  b(e_i)=b(s_k);
  ]
* 同类和异类 same-bin 必须分开；
* **direct-complete/短动作候选**：
  [
  b(s_i)=b(e_i);
  ]
* long-before-memory：动作 duration 超过未来候选共同 memory horizon；在该 horizon 冻结前只能报告 duration-in-bins，不能据此选择 memory 或 K。

### E.1.5 Ambiguous、空视频与 class map

当前数据加载器排除 label 为 `Ambiguous` 的区间。R0 必须：

* hash class-map 原始字节；
* 报告 class-map 顺序和类数；
* 将 Ambiguous 的视频数、区间数和时长单列；
* 在正式统计中排除 Ambiguous；
* “zero-action video”指排除 Ambiguous 后没有合法目标区间的视频。

### E.1.6 主张资格和样本量

当前“too rare for a powered main claim”不能执行。建议在 census 前冻结以下规则：

一个 stress subset 只有同时满足以下条件，才可成为未来**主张级**子集：

1. 至少 **30 个独立视频**；
2. 至少 **100 个 GT 实例**；
3. 至少占相关 positive videos 的 **5%**；
4. 覆盖至少 **3 个动作类别**；
5. 以视频为 cluster 的预先功效模拟显示：对未来 matched comparison 的 **10 个百分点绝对 recall 改善**，在 family-wise (\alpha=0.05) 下 power ≥ 0.80。

分级：

* `PRIMARY_ELIGIBLE`：满足全部五项；
* `SECONDARY_ONLY`：至少 10 视频且 30 实例，但未达到主项；
* `DESCRIPTIVE_ONLY`：低于上述门槛。

依据历史暴露，THUMOS same-class overlap 的 2/3 视频显然不能成为主 claim；若新 R0 仍如此，该 claim 必须从 THUMOS 主表删除，而不是放宽门槛。

### E.1.7 R0 防泄漏

“outcome-blind”必须改称并定义为：

> **blind to all B0–B4 model outcomes，but not annotation-unseen。**

因为并发、时长和 overlap 的部分结果已经在 2026‑07‑12 暴露。必须先建立 `prior_exposure_ledger`，至少记录：

* 旧普查脚本 blob/hash；
* 旧报告中的 200/211 视频、3003/3325 实例；
* 最大并发 2；
* overlap video counts；
* duration summaries；
* 任何作者已经看到的 per-video hard-case IDs。

新 R0 执行规则：

* 独立审计进程只能挂载 annotation、class map、split manifests 和 frozen census source；
* 禁止挂载预测、checkpoint、run summaries、阈值文件和模型 work dirs；
* 作者在运行前只看到 protocol hash，不看到新 hard-case ID lists；
* 正式输出给作者的主 artifact 只包含 aggregate statistics；
* per-video detail 进入独立 reviewer-only artifact，并以 hash 绑定；
* 统计定义或门槛在执行后不得改变；
* 一次性写入临时目录，flush/hash 后原子重命名；
* 现有 evidence root 拒绝覆盖。

---

## E.2 R1 缓存特征因果证书

### E.2.1 当前代码的积极证据

`[代码核验]` 当前 cache extractor：

* 每个 stride packet 只选择最后一帧；
* 将这些单帧组成 batch；
* 每个输入帧产生一个二维 feature vector。

`[代码核验]` `OnlineSigLIPFrameEncoder` 明确按帧独立编码，并把 `[B,T,C,H,W]` 展平为独立图像 batch；只有显式启用 motion branch 时才跨时间混合。cache builder 没有启用该 branch。

因此对当前源码路径：

[
\operatorname{support}(z_{v,j})={d_{v,j}}
]

是可信的静态数据流结论。

### E.2.2 当前 artifact 身份仍不闭合

现有 cache manifest 只记录：

* `encoder_id` 字符串；
* annotation SHA；
* stride；
* dtype/dim；
* source frames；
* 每个 feature array SHA。

cache builder 创建 encoder 时只传 `model_name`，没有传 Hugging Face `revision`；虽然 encoder 类支持 revision，但这里为默认 `None`。

因此当前 manifest **没有绑定**：

* Hugging Face snapshot commit；
* config/processor JSON；
* weight shard hashes；
* transformers/PyTorch 版本；
* OpenTAD extraction source commit；
* resolved encoder config；
* raw video hashes；
* OpenCV 版本和解码参数；
* resize/preprocessing identity；
* exact command/environment；
* 每个 token 的 raw-frame support map hash。

现有 cache 不能仅凭 `encoder_id` 获得 R1 PASS。

### E.2.3 必须冻结的证书对象

R1 certificate 至少包含：

```text
schema_version
repository_commit
extractor_source_sha256
resolved_command_sha256
python_lock_sha256
torch_transformers_opencv_versions
hf_snapshot_revision
model_config_sha256
processor_config_sha256
weight_shard_sha256[]
raw_video_sha256_by_video
annotation_raw_sha256
cache_manifest_sha256
feature_array_sha256_by_video
support_map_sha256
decision_time_convention
decoded_rgb_stream_scope
motion_branch_enabled=false
```

每个 token 记录：

```text
video_id
token_index
source_frame
decision_frame
support_frames
support_min
support_max
```

PASS 的必要且充分条件：

[
\forall v,j,\quad
\max \operatorname{support}(z_{v,j})\le d_{v,j}.
]

对当前单帧编码路线还必须满足：

[
\operatorname{support}(z_{v,j})={d_{v,j}}.
]

### E.2.4 可执行审计

除静态源码证明外，必须预先固定一个 CPU-only 动态抽样：

* 由 `SHA256(video_id || token_index || audit_seed)` 选择每个边界类别的 token；
* 覆盖首 token、中间 token、最后完整 bin、最后 partial bin；
* 修改所有 (d_j) 后的 decoded RGB frames；
* 重新提取 (z_j)；
* 同一 CPU/FP32 deterministic 环境下要求 byte-identical；
* 改变 batch 中其他图像和 batch 顺序，目标 token 也必须 byte-identical；
* 随机改变当前 support frame 时，feature 必须发生非零变化，避免测试实际上没有消费输入。

R1 应明确只证明**decoded RGB frame stream causality**。如果未来声称从压缩视频 bitstream 到决策的墙钟在线性，还需另行审计 PTS/DTS、B-frame buffering 和 decode latency；当前 route evidence 不应偷偷包含该 claim。

### E.2.5 当前 cache 无法认证时的处置

必须预先冻结：

```text
missing exact weight snapshot
OR sample re-extraction does not match existing cache
OR raw source support cannot be reconstructed
=> CURRENT_CACHE_R1=FAIL_UNVERIFIABLE
=> B0-B4 may not use this cache
=> no replacement cache creation is automatically authorized
```

---

## E.3 R2：B0–B4 唯一化和公平规则

### E.3.1 五臂是否必要

五臂可以保留，但只有按以下方式唯一化后才不重复：

| Arm                                  | 必须冻结的唯一语义                                                                                                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **B0 无身份 prefix completion set**     | 每个 decision fresh action queries；可读取与其他臂相同的 causal feature memory；query tensor 不跨 decision；普通 per-prefix set matching；无 carrier identity。                      |
| **B1 ordinary persistent query**     | K 个 query 跨 decision 传播；所有 query 始终存在；无 newborn/track-query 分池、无 hard FREE/ACTIVE、无 UOT、无 continuous occupancy、无 consistency、无 latch/reseed。                   |
| **B2 temporal MOTR**                 | 明确区分 newborn queries 与 propagated track queries；track query 逐时刻更新；使用 temporal tracklet-aware training assignment；共用 On‑TAL interval/completion heads 和 ledger。 |
| **B3 clean order/risk-set baseline** | 新模块，不修改旧 Q2；保留一个简单 hard model-only lifecycle；先运行与推理相同的 transition/release，再计算 loss；GT 不持久化 canonical identity；普通 Hungarian+dustbin；无 UOT 和连续 occupancy。        |
| **B4 R‑A**                           | 只允许包含经 R6 确认为 exact delta 的机制。persistent queries、interval head 和 ledger 不能算 B4 特有组成。                                                                           |

TrackFormer 已经使用静态 object queries 初始化新轨迹，并以 identity-preserving track queries 自回归传播已有轨迹；MOTR 又加入逐帧 track-query propagation、newborn queries 和 tracklet-aware assignment。([arXiv][8])

TadTR 已用 action queries 直接输出动作区间和类别，因此 query set 和 interval head 不能算 R‑A 差异。([arXiv][9])

### E.3.2 公平匹配规则

建议冻结下表：

| 维度                       | 唯一公平规则                                                                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Input                    | 完全相同的 cache manifest SHA、feature bytes、source frames、decision times、mask 和 split IDs。                                               |
| Feature memory           | 所有臂获得同一长度的 causal visual memory；B0 只禁用 query identity，不得被剥夺视觉历史。                                                                    |
| Prediction interface     | 相同 class/start/end/completion/score 语义和相同输出坐标；arm-specific state 只在内部。                                                              |
| Ledger                   | 使用同一个 ledger 实现、schema、hash 和 evaluator；不得 arm-specific NMS、merge、dedup 或 score filtering。                                          |
| Trainable parameters     | 排除 frozen encoder 后，全体 trainable parameters 相对共同目标 (P^*) 在 ±5%；计入所有 arm-specific modules；禁止 dummy、未调用或 frozen padding 参数。           |
| Token budget             | 每个 seed 的视频顺序、token IDs、token multiplicity、TBPTT cut、mask 必须逐字节相同。                                                                  |
| Update budget            | optimizer event 数完全相同；gradient accumulation 和有效 token denominator 相同；禁止 early stopping；任一 skipped/nonfinite update 使 paired run 无效。 |
| Optimizer                | 相同 optimizer family、schedule shape、precision 和 clip；若调参，所有臂使用相同预登记网格、相同 trial 数和相同选择规则，禁止手工 arm-specific tuning。                    |
| Hyperparameter selection | 在 fit-core 160 内做固定 nested video split；calibration 40 只用于 score/threshold calibration，不同时承担 architecture selection。                 |
| Threshold                | 不是强制相同数值，而是对每臂使用相同、预登记的 calibration algorithm；同时报告未经阈值后处理的 AP/PR 和固定 false-emission 或 matched-recall operating point。               |
| Reporting                | reporting population 一次性打开；所有臂/seed 全部完成并 hash 后同时评估。                                                                               |
| Statistical unit         | 以 video 为 cluster，臂间使用相同 bootstrap resamples；seed 为嵌套重复。                                                                            |
| Search cost              | 相同超参数 trial 数、训练 token 数和 optimizer events；参数量和训练 MAC 另报告，MAC 建议控制在 ±10%。                                                           |

当前 gate 只列“identical input/head/ledger、parameter matching、token/update matching”，没有上述可执行口径。

---

## E.4 R3 负对照的精确处置

| 控制                        | 必须固定的输入权限                                                                            | 正确用途                    |
| ------------------------- | ------------------------------------------------------------------------------------ | ----------------------- |
| count-only                | 只能看 prefix index、训练 event-count prior 和类别 prior；不能看 feature 值                        | 排除只学事件数量                |
| template timing           | 只能看时间索引及训练 duration/gap 分布                                                           | 排除固定模板计时                |
| feature-time shuffle      | 在每条序列内用冻结 permutation 改变 token 时间顺序，同时保持 token multiset、event count 和 label marginal | 验证 temporal order 是否被使用 |
| label-feature permutation | 冻结全局 class permutation，使 feature 与原 GT class 对应破坏                                    | 验证类别语义                  |
| ledger-only               | 接受 B0 的同一 raw candidates，仅使用共享 ledger 的确定性规则                                         | 隔离收益是否只来自 ledger        |
| history-off               | 对 B1/B2/B3/B4 清空跨 decision state，但保留同一当前 prefix memory                               | 测持续历史的实际贡献              |
| threshold sweep           | 不是负模型；是 calibration sensitivity analysis                                             | 防止低提交率伪造 precision      |

统一判定：

> 若 count-only、template-timing 或 ledger-only 在全部主指标上落入最强真实模型的冻结等价区域，则 benchmark 的 lifecycle identifiability 为 FAIL，而不是宣布复杂模型 PASS。

feature shuffle 和 label permutation不要求所有指标都下降；必须预登记其应影响的指标：

* feature-time shuffle：interval/timing/latency；
* label-feature permutation：class AP；
* count-only：count error 可能较好，但 interval/class/instance metrics 必须失败。

---

## E.5 R4 机制删除：删除不必要项目

当前十项删除过宽。最小而充分的 B4 核心删除只应由 R6 exact delta 决定。

必须保留的候选核心删除：

1. UOT → Hungarian+dustbin；
2. continuous occupancy → binary validity；
3. adjacent-prefix consistency off；
4. release-before-reseed → no same-bin reuse；
5. latch excluded versus model-output-only latch included。

以下项目应移出“所有 B4 都必须做”的 route-core ablation：

* `K versus K+1`：这是 capacity stress，移入 R5；
* start posterior → scalar：只是输出头比较，除非 R‑A 把它列为核心差异；
* direct-complete：只有 R0 证明短动作子集有主张资格时才允许；
* partial matching：只有 UOT 相对 Hungarian 的收益成立后才作为第二层数学消融。

机制“必要”的冻结规则必须是：

* 完整 B4 相对删除版在预声明 lifecycle metric 上超过最小实质差异；
* 同时 standard mOnlineAP、recall、false-emission 不发生超 margin 恶化；
* 使用同一 paired simultaneous CI；
* 仅 loss 下降、gradient 非零或内部 swap 下降不构成机制必要性。

---

## E.6 R5 Structural OOD

当前十个 OOD 名词应压缩成四个独立因子族：

1. **事件拓扑**：count、并发、overlap graph、same-bin transition；
2. **时间几何**：duration、gap、endpoint delay；
3. **语义映射**：class permutation；
4. **观测分布**：feature basis rotation/mixture、noise family。

必须冻结三层集合：

* `IID_HOLDOUT`：相同 factor support、不同 seed；
* `SINGLE_SHIFT_OOD`：每次只改变一个因子；
* `COMPOUND_OOD`：至少两个未见因子同时改变。

必要规则：

* generator grammar 和所有公开 factor levels 在模型实现前 hash；
* compound OOD 的组合表和 hidden seed 由独立 reviewer 保存；
* training、IID、single-shift、compound 的 canonical sequence hashes 无交集；
* 每个主 OOD family 满足与 R0 相同的最低样本/视频等价要求；
* 不能在看到哪个模型失败后新增 OOD family；
* 负对照若也通过 compound OOD，benchmark KILL。

---

## E.7 R6：temporal-MOTR exact delta

### E.7.1 当前映射

| R‑A 候选元素                    | TrackFormer/MOTR/TAD 最近机制                                   | 当前可主张的精确差异                                                        | 目标 On‑TAL 错误       | 隔离实验                     | 当前裁决                    |
| --------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- | ------------------ | ------------------------ | ----------------------- |
| persistent carriers         | track queries                                               | **无**；只是命名不同                                                      | 长动作历史              | B1/B2                    | `NO_DELTA`              |
| newborn/reseed              | static object queries、newborn queries                       | 只有“完成后在同一 decision bin 原子 release/reseed”可能有差异                    | same-bin end/start | 禁用 same-bin reuse        | `PROVISIONAL_DELTA`     |
| continuous occupancy        | objectness/alive/track confidence                           | 只有 occupancy mass 真正参与 release/reseed，而非普通 validity head 时才有差异    | 硬状态误释放/容量          | binary validity ablation | `PROVISIONAL_DELTA`     |
| loss-only UOT               | Hungarian/tracklet-aware assignment                         | ephemeral unbalanced mass、dustbin，且不建立 canonical runtime identity | Q2 target discard  | Hungarian+dustbin        | `PROVISIONAL_DELTA`     |
| adjacent-prefix consistency | tracklet-aware identity continuity、collective temporal loss | 很可能只是训练身份平滑                                                       | transport swap     | consistency off          | `WEAK_OR_ANTIDELTA`     |
| start posterior             | action-query interval head、MATR memory start estimate       | before-memory censoring可算任务适配，但不是 carrier novelty                 | 起点遗忘               | scalar start             | `HEAD_DELTA_ONLY`       |
| first-completion/end hazard | tracker termination + On‑TAL endpoint head                  | first-event survival语义是 On‑TAL 适配，不自动证明 carrier 必要                | 延时/重复 end          | binary completion        | `TASK_ADAPTATION`       |
| model-only latch            | active-track validity/termination bookkeeping               | 若只阻止重复 commit，属于硬 controller；若进入神经状态，必须证明无 GT taint               | 重复提交               | latch dataflow variants  | `NOT_ESTABLISHED`       |
| immutable ledger            | online output protocol                                      | **无**；所有臂共享                                                       | 不可回改               | ledger-only control      | `NO_DELTA`              |
| direct-complete             | short-action special branch                                 | 数据驱动补丁，不是 carrier 创新                                              | 单 bin 动作           | disable                  | `NOT_ALLOWED_BEFORE_R0` |

TrackFormer/MOTR 已覆盖 query birth、identity propagation、逐帧更新和 track-aware assignment；TadTR 已覆盖 action query 与直接 interval prediction。([arXiv][8])

当前 R‑A 最多剩下两个待证实的 route-level delta：

* **D1：原子 release-before-same-bin-reseed；**
* **D2：不建立 GT canonical runtime identity 的 continuous mass + ephemeral UOT。**

如果最终合同不能把 D1 或 D2 写成与 temporal MOTR 不同的状态转移或训练 estimand，应该在模型实现前直接 `KILL_PERSISTENT_CARRIER_ROUTE`。

### E.7.2 必须冻结的等价区域

当前 gate 只有“falls inside the frozen equivalence region”，但没有区域。建议在 R0 前冻结公式：

* average mOnlineAP：
  [
  \delta_{\mathrm{mAP}}=0.005
  ]
  即 0.5 个百分点；
* rate metric，分母 (N)：
  [
  \delta_{\mathrm{rate}}(N)=\max(0.01,;2/N);
  ]
* stress-subset recall：
  [
  \delta_{\mathrm{stress}}(N)=\max(0.02,;2/N);
  ]
* latency：
  [
  \delta_{\mathrm{latency}}=1\ \text{decision bin}.
  ]

这里 (N) 只来自先冻结的 R0 census，因此不会依赖模型结果。

统计规则：

* 10,000 次 paired video-cluster bootstrap；
* 每次对所有臂使用相同视频 resample；
* seed 在视频 cluster 内嵌套；
* 对全部 primary metrics 使用 simultaneous 95% CI；
* 不允许看到结果后选择最有利 metric。

B4 路线存活必须同时满足：

1. mOnlineAP 和 event recall 对 B2 非劣；
2. 至少一个预声明 lifecycle metric 的改善 CI 完全超过其 (\delta)；
3. 其他 primary lifecycle metrics 不发生超 margin 恶化；
4. 负对照未进入同一等价区域。

若 B4 与 B2 在所有 primary metrics 上的 simultaneous CI 全落入等价区域，或 B4 更差，则：

```text
R_A_ROUTE_STATUS=KILL_TEMPORAL_MOTR_EQUIVALENT
```

---

# F. 最小而充分的路线证据包

当前协议列出的内容可以压缩为六个不可变对象。

## 必须保留

### 1. Route protocol manifest

绑定：

* exact repository commit/tree；
* R0–R6 protocol file hashes；
* schema version；
* reviewer identity；
* prior exposure ledger hash；
* allowed inputs；
* forbidden model artifacts；
* terminal rules。

### 2. R0 annotation census package

只含：

* annotation/class-map/split hashes；
* exact population IDs 的 hashes；
* frozen definitions；
* aggregate report；
* reviewer-only detailed rows hash；
* statistical eligibility verdict；
* stdout/stderr/environment；
* atomic publication commitment。

### 3. R1 cache causality package

只含：

* encoder/source/weight/processor/environment identity；
* raw video/cache hashes；
* token support map；
* static dataflow proof；
* CPU future/batch perturbation audit；
* cache linkage verdict。

### 4. B0–B4 route comparison contract

只冻结：

* 五臂语义边界；
* common input/head/ledger；
* parameter/token/update/tuning/evaluation matching算法；
* no model dimensions or weights。

### 5. R3–R5 identifiability contract

只冻结：

* exact negative controls；
* core mechanism deletions；
* compact四因子 OOD grammar；
* hidden-generation ownership；
* benchmark-identifiability KILL rule。

### 6. R6 exact-delta/equivalence object

只含：

* mechanism mapping table；
* D1/D2 等候选 exact delta；
* numeric/formula equivalence margins；
* paired statistical rule；
* temporal-MOTR equivalence KILL。

## 应删除或继续保持 inactive

* 历史 P0 的 9101/9102/9103 seeds；
* 512/256/64 synthetic counts；
* 0.90/0.80 旧 margins；
* K、hidden dim、memory size；
* UOT epsilon/solver；
* optimizer/LR/update count；
* ledger thresholds；
* direct-complete 默认值；
* 完整十项机制删除；
* GPU/profile/VRAM/runtime budget；
* raw-video finetuning；
* Q2/R1 tickets 和旧 P0 evidence schema。

这些项目属于未来 model-P0 合同，不属于路线证据包。

---

# G. 可以开始只读证据收集前必须完成的修改

必须先产生一个新的、只含协议修改的 immutable commit，并同时关闭以下九项：

1. **解决 211/213 reporting population。**
2. **把 R0 的时间、区间、重叠、重复、same-bin、短动作、空视频和 Ambiguous 定义写成公式。**
3. **冻结主张资格门槛、置信区间和 cluster-aware power 规则。**
4. **新增 prior-exposure ledger，承认旧 overlap/concurrency/duration 已暴露。**
5. **定义 outcome-blind 为 model-outcome-blind，并冻结 R0 执行隔离、aggregate-only disclosure 和 reviewer-only detail。**
6. **新增 R1 certificate schema、权重/processor/raw-video hashes、support map、CPU perturbation和 current-cache-unverifiable 处置。**
7. **按 E.3 唯一化 B0–B4，冻结参数/token/update/调参/threshold/evaluator matching 算法。**
8. **把 R3–R5 改为 exact control constructions、core ablations 和 compact OOD grammar。**
9. **提交完整 R6 exact-delta table 和数值/公式化等价区域。**

完成后仍不能直接运行 R0/R1。必须由另一位只读 reviewer 对该新 protocol commit 选择：

```text
PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION
或
REVISE_PROTOCOL_BEFORE_COLLECTION
```

本轮不能用当前回答本身替代仓库中的哈希协议。

---

# H. 收集证据后下一次路线复审所需的唯一输入

下一位路线 reviewer 不应接收作者总结作为证据。唯一输入应是一个只读 evidence-root manifest，引用：

```text
protocol_commit_sha
protocol_tree_sha
route_protocol_manifest_path + sha256
prior_exposure_ledger_path + sha256

r0/
  evidence_manifest.json
  annotation_file_record.json
  class_map_file_record.json
  split_records.json
  census_definitions.json
  aggregate_report.json
  reviewer_detail_commitment.json
  statistical_eligibility.json
  stdout.txt
  stderr.txt
  environment.json

r1/
  evidence_manifest.json
  encoder_provenance.json
  weight_processor_records.json
  raw_media_records.json
  cache_records.json
  token_support_map.json
  static_dataflow_certificate.json
  cpu_perturbation_audit.json
  cache_linkage_verdict.json

route_contract/
  b0_b4_arm_taxonomy.json
  fairness_contract.json
  negative_controls.json
  mechanism_deletions.json
  structural_ood_contract.json
  temporal_motr_exact_delta.json
  equivalence_region.json

terminal_verdict.json
```

明确禁止作为下一轮输入：

* B0–B4 模型代码；
* model predictions；
* checkpoint；
* training logs；
* effect metrics；
* GPU profile；
* raw-video finetuning结果；
* 作者筛选后的 hard-case examples；
* 未绑定的截图或手工表格。

下一轮 reviewer 只回答：

1. 数据是否足以验证至少一个 carrier-specific observable claim；
2. 当前 cache 是否获得 strict decoded-RGB causality PASS；
3. B0–B4 是否唯一且公平；
4. R3–R5 是否能识别生命周期而非模板；
5. R6 是否留下至少一个非 MOTR 重命名的 exact delta；
6. 是否授权设计 model-P0 合同。

即使上述全部通过，下一次最多只能选择：

```text
PASS_AUTHORIZE_MODEL_P0_CONTRACT_FREEZE
```

仍不能授权实现、真实数据效果实验或 GPU 工作。

---

# I. 机器可读 JSON

```json
{
  "repository": "yuzbo/OpenTAD_OnlineTADClean_20260702",
  "reviewed_commit": "b70baf3bb5d819d1df25ca47534c5d3bfbb6e695",
  "commit_verified": true,
  "moving_head_used": false,
  "final_verdict": "REVISE_PROTOCOL_BEFORE_COLLECTION",
  "confidence": 0.96,

  "q2_r1_terminal_kill_unchanged": true,
  "dr045_active": false,
  "dr046_active": true,
  "r_a_role": "B4_CANDIDATE_ONLY",
  "historical_r_a_p0_status": "WITHDRAWN_NOT_IMPLEMENTED_NOT_RUN",

  "standard_ontal_task_value": "CONDITIONALLY_ESTABLISHED",
  "persistent_carrier_field_gap": "NOT_ESTABLISHED",
  "simple_baselines_insufficient": "NOT_ESTABLISHED",
  "r_a_necessity": "NOT_ESTABLISHED",
  "r_a_distinct_from_temporal_motr": "NOT_ESTABLISHED",
  "thumos_supports_internal_identity_claim": false,
  "thumos_supports_output_level_identity_metrics": true,

  "r0_protocol": "REVISE",
  "r0_evidence_status": "NOT_RUN_NEW_PROTOCOL",
  "r0_prior_annotation_exposure_exists": true,
  "r0_reporting_population_211_vs_213_resolved": false,
  "r0_power_thresholds_frozen": false,

  "r1_protocol": "REVISE",
  "r1_algorithmic_frame_support": "LIKELY_SINGLE_FRAME_CAUSAL",
  "r1_existing_cache_artifact_identity": "UNVERIFIED",
  "r1_exact_weight_snapshot_bound": false,
  "r1_token_support_map_bound": false,
  "r1_dynamic_future_perturbation_run": false,

  "r2_protocol": "REVISE",
  "b0_required": true,
  "b1_required_if_uniquely_defined": true,
  "b2_required": true,
  "b3_required_if_isolated_to_order_risk_repair": true,
  "b4_default_method": false,
  "parameter_matching_rule_frozen": false,
  "token_update_matching_rule_frozen": false,
  "threshold_calibration_rule_frozen": false,

  "r3_protocol": "REVISE",
  "r4_protocol": "REVISE",
  "r5_protocol": "REVISE",
  "r6_protocol": "REVISE",
  "temporal_motr_equivalence_region_frozen": false,

  "plausible_r_a_exact_deltas": [
    "ATOMIC_RELEASE_BEFORE_SAME_BIN_RESEED",
    "CONTINUOUS_CARRIER_MASS_WITH_EPHEMERAL_NONCANONICAL_UOT"
  ],
  "non_deltas": [
    "PERSISTENT_QUERY_RENAMED_AS_CARRIER",
    "INTERVAL_HEAD",
    "CLASS_HEAD",
    "START_HEAD_ALONE",
    "IMMUTABLE_SHARED_LEDGER",
    "GENERIC_MEMORY",
    "GENERIC_MATCHING"
  ],

  "read_only_annotation_census_allowed_now": false,
  "read_only_cache_causality_audit_allowed_now": false,
  "model_contract_freeze_allowed": false,
  "model_implementation_allowed": false,
  "historical_r_a_p0_allowed": false,
  "real_data_effectiveness_allowed": false,
  "gpu_profile_allowed": false,
  "formal_training_allowed": false,
  "visual_finetuning_allowed": false,
  "raw_video_training_allowed": false,
  "gpu_hours_authorized": 0,

  "blocking_revisions": [
    "RESOLVE_REPORTING_211_VS_213",
    "FREEZE_R0_FORMULAS_AND_POWER_RULES",
    "REGISTER_PRIOR_ANNOTATION_EXPOSURE",
    "FREEZE_MODEL_OUTCOME_BLIND_EXECUTION",
    "FREEZE_R1_CERTIFICATE_AND_CACHE_LINKAGE",
    "UNIQUIFY_B0_B4_ARMS",
    "FREEZE_PARAMETER_TOKEN_UPDATE_TUNING_AND_THRESHOLD_FAIRNESS",
    "FREEZE_NEGATIVE_CONTROL_AND_STRUCTURAL_OOD_CONSTRUCTIONS",
    "PROVIDE_TEMPORAL_MOTR_EXACT_DELTA_TABLE",
    "FREEZE_NUMERIC_OR_FORMULA_EQUIVALENCE_REGION"
  ],

  "next_possible_decision_after_protocol_revision": [
    "PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION",
    "REVISE_PROTOCOL_BEFORE_COLLECTION",
    "KILL_PERSISTENT_CARRIER_ROUTE",
    "BLOCK_REPOSITORY_OR_DATA_UNVERIFIABLE"
  ]
}
```

[1]: https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html "https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html"
[2]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2307_ECCV_2022_paper.php "https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2307_ECCV_2022_paper.php"
[3]: https://arxiv.org/abs/2211.04905 "https://arxiv.org/abs/2211.04905"
[4]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2834_ECCV_2024_paper.php "https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2834_ECCV_2024_paper.php"
[5]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3153_ECCV_2024_paper.php "https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3153_ECCV_2024_paper.php"
[6]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1621_ECCV_2024_paper.php "https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1621_ECCV_2024_paper.php"
[7]: https://arxiv.org/abs/2605.09976 "https://arxiv.org/abs/2605.09976"
[8]: https://arxiv.org/abs/2101.02702 "[2101.02702] TrackFormer: Multi-Object Tracking with Transformers"
[9]: https://arxiv.org/abs/2106.10271 "[2106.10271] End-to-end Temporal Action Detection with Transformer"
