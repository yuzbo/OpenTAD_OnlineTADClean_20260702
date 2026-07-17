# Prefix-Shared Route Preimplementation Review: Independent Absorption

Date: 2026-07-17 Asia/Shanghai
Source file: `PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md`
Source attachment SHA-256:
`B5BAC9E38C40883687193E37F5364A6A165C67EEDF0F496607A59B46FD88A3D4`
Source size: 50,146 bytes, 653 text lines
Reviewed repository anchor:
`f9419612cc90860019cd9a124833568489c9d38c`

## Absorption Verdict

我认可这份审查的核心裁决和立即行动边界，但不把其中每一个
`false` 都解释成已经被科学证明为假。

准确结论是：

> 接受 `REVISE_ROUTE_AND_REVIEW_AGAIN`。撤销 DR-045 对 R-A
> 预登记、实现和 CPU P0 的当前授权。R-A 不再是默认主方法，只保留为
> B0-B4 路线比较中的 B4 候选臂。在完成只读数据普查、缓存特征因果证书、
> 简单基线合同、结构 OOD、负对照和 temporal-MOTR exact-delta 之前，
> 不冻结 R-A 合同，不写 R-A 模型代码，不做真实数据效果实验，不使用
> GPU。

这一裁决不改变既有终止结论：

- CRS-EPS empty-state replay 仍为终止 `KILL`；
- Q2 仍只是一份有效的局部失败证据；
- Q2/R1 和 `birth_prior_bias_m2` 不得重新包装或修改后重跑；
- profile、正式训练、视觉微调和 raw-video 训练继续禁止；
- GPU 授权保持 0 小时。

## 完全认可的核心意见

### 1. Q2 证据不能推出 R-A 必要性

现有证据证明的是 Q2 在 decode 前使用硬 runtime availability
筛选 GT birth，并可能永久丢弃监督。这足以终止 Q2/R1，却不足以证明：

- 标准 On-TAL 普遍缺少持续潜在身份；
- continuous occupancy 是必要变量；
- UOT 比普通 matching 必要；
- R-A 是唯一或最简单的修复。

DR-045 把一个局部实现失败过早提升成了路线授权。新审查正确地撤销了
这个推理跳跃。

### 2. 一维 TrackFormer/MOTR 是必须正面回答的重建攻击

TrackFormer 已经用静态 object queries 产生新轨迹，并用跨帧传播的
track queries 保持已有轨迹。MOTR 同样逐帧更新 track queries，并用
tracklet-aware assignment 训练出生与存续。TadTR 则证明了 action
queries 可以直接预测时间区间。

因此，持续查询、出生/存续、集合匹配和区间头的组合本身不能成为
R-A 的创新。R-A 必须给出相对一维 TrackFormer/MOTR 的 exact-delta，
并通过匹配预算实验显示这些差异带来不可重建的输出层实例一致性收益。

### 3. 当前 P0 没有路线辨识力

八类 scripted case、因果等价、ledger 不可修改和 anti-silence
适合发现程序错误，但不能排除：

- 模板时间预测；
- 事件计数；
- 保守阈值策略；
- 普通 prefix set prediction；
- 普通 persistent query；
- ledger-only 去重。

种子分离也不能替代结构 OOD。没有 B0-B4、负对照和机制删除时，
P0 即使通过，也只能证明一个程序拟合了同一生成规则。

### 4. 真实数据可测性必须先于机制复杂化

当前项目 split 中同类重复、同类重叠、same-bin end/start、短动作和
最大并发的发生率尚未被绑定证据核验。如果这些现象稀少，围绕它们设计
复杂载体、direct-complete 或固定 K 容量机制，容易变成解决人为问题。

必须先做 outcome-blind、只读、哈希绑定的 annotation census。该普查
只描述任务结构，不观察任何模型结果，也不构成效果实验。

### 5. 缓存特征的严格因果性是硬前提

source frame 不晚于当前时刻，只能证明缓存索引顺序合法。若生成该
token 的视觉编码器 clip 在时间上使用了未来帧，则后续所有严格在线
结论都失效。必须绑定编码器代码、权重、采样方式、clip 边界和精确
temporal receptive field，形成可复核证书。

### 6. B0-B4 和机制删除应成为同一比较合同

应保留以下候选臂：

- B0: 无持续身份的 prefix completion set；
- B1: 普通 persistent query 加同一组区间/类别/completion heads；
- B2: 一维 TrackFormer/MOTR reconstruction；
- B3: 新建且不修改旧 Q2 的 order/risk-set baseline；
- B4: R-A。

所有臂必须共享因果输入、输出头、ledger、生成器、评测器、反静默门禁，
并预先冻结参数量和更新预算匹配规则。UOT、occupancy、consistency、
release/reseed、latch、start posterior 和 direct-complete 都必须有
删除或替代对照。

## 不逐字完全认可的部分

### 1. `false` 应解释为“未建立”，不是“已证伪”

机器决定中的：

- `"问题属于领域缺口": false`
- `"问题不是旧代码局部错误": false`
- `"简单基线不足": false`

在当前证据下应规范化为 `NOT_ESTABLISHED`，而不是反向声称：

- 领域缺口已经被证明不存在；
- 所有问题都被证明只是 Q2 bug；
- 简单基线已经被证明足够。

我们现在只有足够证据阻止 R-A 获得默认路线地位，没有足够证据永久
终止持续载体家族。

### 2. 内部身份不可直接观测，不等于内部身份无科学价值

标准评测只观察最终区间、类别、分数和提交时间，因此不能直接验证
query ID 是否交换。这确实禁止把“内部 identity 保持”本身当作主结果。

但内部状态仍可能通过可观测结果体现价值，例如降低 duplicate、
fragmentation 和同类重叠漏检，或改善 recall-latency Pareto。正确要求
是把机制 claim 映射到这些输出指标，而不是先验否定内部状态。

### 3. R-A 应降级，而不是现在永久删除

R-A 的 continuous occupancy、release-before-reseed、loss-only
transport 和 immutable lifecycle 可能形成超出 temporal MOTR 的
有效差异，但目前没有证据。最合理处置是 B4 候选臂和预声明等价即
KILL，不是把未来所有类似路线提前判死。

### 4. 当前先冻结路线证据协议，不冻结五个模型的实现细节

下一步应先冻结：

- census 字段、split、哈希和盲性；
- causal-feature certificate 的证明对象；
- B0-B4 公平性原则和 kill criteria；
- 结构 OOD 与负对照族；
- temporal-MOTR equivalence region。

在这些路线级对象通过重新审查前，不应进一步选择 R-A 的 K、UOT、
threshold 或优化器，也不应编写五臂的逐文件实现计划。

## 重新校准后的科学问题

原问题：

> 如何实现 Prefix-Shared Latent Event Filter？

现问题：

> 在标准、严格因果、不可回改正式提交的 On-TAL 中，持续载体是否相对
> 无身份 prefix 集合和一维 temporal-MOTR，在预算匹配条件下带来无法由
> 简单重建解释的在线实例一致性收益？

输出层的“实例一致性收益”必须预先落到：

- duplicate；
- fragmentation；
- same-class repetition/overlap recall；
- same-bin correctness；
- event recall 与 false emission；
- endpoint/commit latency；
- mAP-latency 或 recall-latency Pareto。

任何内部 query swap 指标只能作为诊断，不能代替正式输出证据。

## 新的路线级门禁

```text
G-R0 outcome-blind annotation census
-> G-R1 cached-feature strict-causality certificate
-> G-R2 B0-B4 shared comparison and budget contract
-> G-R3 structural OOD, negative controls, mechanism deletions
-> G-R4 temporal-MOTR exact-delta and equivalence-KILL rule
-> independent route re-review
-> only then decide whether any model P0 may be frozen
```

任一门禁未闭合时：

```text
R_A_DEFAULT_ROUTE=false
R_A_MODEL_CODE_ALLOWED=false
MODEL_P0_ALLOWED=false
REAL_DATA_EFFECTIVENESS_ALLOWED=false
GPU_ALLOWED=false
GPU_HOURS_AUTHORIZED=0
```

## 当前授权矩阵

| 动作 | 状态 |
|---|---|
| 原样归档并吸收本次审查 | `ALLOW` |
| 更新 wiki、决策与路线问题 | `ALLOW` |
| 设计只读 annotation census 合同 | `ALLOW` |
| 设计缓存特征因果证书合同 | `ALLOW` |
| 冻结路线级 B0-B4/OOD/kill 原则 | `ALLOW` |
| 编写或运行 R-A 模型 P0 | `BLOCKED` |
| 冻结 R-A 状态/UOT/loss/threshold | `BLOCKED` |
| 真实数据效果实验 | `BLOCKED` |
| GPU profile 或训练 | `BLOCKED` |
| 正式或 raw-video 训练 | `BLOCKED` |
| GPU 小时 | `0` |

## 独立来源复核

本次吸收只对审查的中心竞争关系做了定向复核，不声称完成穷尽式
2026 新颖性检索：

- CAG-QIL: https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html
- SimOn: https://arxiv.org/abs/2211.04905
- OAT: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2307_ECCV_2022_paper.php
- TrackFormer: https://arxiv.org/abs/2101.02702
- MOTR: https://arxiv.org/abs/2105.03247
- TadTR: https://arxiv.org/abs/2106.10271
- MATR: https://arxiv.org/abs/2408.02957
- ActionSwitch: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1621_ECCV_2024_paper.php

这些原始来源支持“任务真实存在、简单在线路线和持续 query 先例存在、
重叠与长期历史是已研究问题”。它们不单独证明 R-A 必要或不必要。

## Machine-Readable Absorption

```json
{
  "source_sha256": "B5BAC9E38C40883687193E37F5364A6A165C67EEDF0F496607A59B46FD88A3D4",
  "reviewed_commit": "f9419612cc90860019cd9a124833568489c9d38c",
  "core_verdict_accepted": true,
  "word_for_word_acceptance": false,
  "normalized_route_verdict": "REVISE_ROUTE_AND_REVIEW_AGAIN",
  "field_gap_status": "NOT_ESTABLISHED",
  "simple_baseline_insufficiency": "NOT_ESTABLISHED",
  "r_a_status": "CANDIDATE_ARM_B4_ONLY",
  "dr_045_authorization_active": false,
  "route_evidence_design_allowed": true,
  "r_a_model_code_allowed": false,
  "model_p0_allowed": false,
  "real_data_effectiveness_allowed": false,
  "gpu_allowed": false,
  "gpu_hours_authorized": 0
}
```
