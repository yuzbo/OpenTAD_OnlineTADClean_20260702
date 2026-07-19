# 1. Executive verdict

**Gate A 未闭合。**

`48e78e43a77822ee0e51cace9ad758ca935c1450` 确实关闭了上一轮若干具体缺口：UNREGISTERED 的正式 population/R0 路径被阻断；B2 已改为目标列加独立 dustbin 列的增强 Hungarian；R5 已逐行冻结重生成；R6 正式入口也开始绑定 protocol、manifest、review、population、R0、run provenance 和 control construction。

但仍存在至少五类能够破坏科学 fail-closed 性的确定性问题：

1. **序列化 fairness record 仍可自组为 PASS**：R6 消费端不是重新测量活模型，而是构造 `model=None, optimizer=None` 的适配器，只核对预算计划及自报标量的内部一致性。
2. **terminal helper 仍可直接消费手工 inference dict**；只要填满正确字段和有利区间，即可返回 `PASS_B4_ROUTE_SURVIVES`，且正式 Python API 可在同一进程中被 monkeypatch。
3. **R0、R5 仍有脱离 protocol/manifest/review 的低层 PASS/certificate-like 输出路径**。
4. **R6 没有验证真实 append-only emission ledger**；它把 caller emission 转成指标输入时直接附加 `immutable=True`，并令 evaluator `require_ledger=False`。
5. **manifest 未覆盖正式 R1/cache import 的父包传播面**，至少遗漏 `opentad/models/__init__.py` 和 `opentad/models/backbones/__init__.py` 及其传递导入。

因此本轮严格 stop-chain 在 Gate A 终止。**没有进入 ChronoTransport 代码适配审查，也没有审核或批准 Streaming CT kill-test。**

---

# 2. Exact-commit access report

| 对象 | 访问结果 | 本轮使用范围 |
| --- | --- | --- |
| `yuzbo/OpenTAD_OnlineTADClean_20260702@48e78e43a77822ee0e51cace9ad758ca935c1450` | **READABLE**。GitHub 返回的 exact commit 为 `48e78e4…`，提交信息为 `Close Prefix Route V2 round-three bypasses`。 | 所有 Gate A 文件均通过 `ref=48e78e4…` 固定读取；未使用移动分支 HEAD。 |
| `yuzbo/OpenTAD_C3_CoarseClean_20260702@b1dc4821f1b04b832dd294c2c3aa5401726130ab` | **READABLE**。GitHub 返回的 exact commit 为 `b1dc482…`，提交信息为 `docs: make Pro snapshot failures diagnostic`。 | 只完成 immutable commit 可见性确认。由于 Gate A 失败，未进入任何 CT 必读生产文件。 |

审查全程只读。未运行测试、脚本、训练、推理、profile 或 GPU；未读取 checkpoint、训练日志、预测、性能结果或既有 Gate artifact；未创建 commit、branch、PR、registration 或实验工单。

---

# 3. Protocol V2 A1–A12 closure table

| 项 | 裁决 | Exact file:line 与可利用路径 | 最小修复 |
| --- | --- | --- | --- |
| **A1** R0、R5、R6、terminal 只能由 repository-owned、commit/review-bound 正式入口产生 | **FAIL** | R6 正式入口相对严格，但并非唯一能产生结论的路径。`collect_r0_census` 是公开函数并直接返回 `status="PASS_R0_COMPLETE"`；`audit_sequence_sets` 接受任意内存 dict 并返回 `complete_frozen_set_contract=True`；`_terminal_route_decision` 接受完整 inference dict 并直接返回 route PASS/KILL。`opentad/utils/prefix_route_r0_v2.py:513-763`；`opentad/utils/prefix_route_ood_v2.py:669-827`；`opentad/evaluations/prefix_route_r6_v2.py:907-1066`。 | R0/R5 低层函数只能返回无科学状态的 diagnostics。只有一个干净子进程正式入口能够生成带 protocol、manifest、review、raw-evidence digest 的 certificate。terminal 不得接受独立 caller inference dict。 |
| **A2** UNREGISTERED、伪 population、伪视频、占位 revision、任意 caller dict 不能产生 PASS/certificate | **FAIL** | 正式 `validate_population_bundle`/`validate_r0_bundle` 的 UNREGISTERED 阻断已修复；但公开 R0 collector 完全不知道 registration 状态，任意 annotation、class list、video list 和 caller bootstrap 参数仍可得到 `PASS_R0_COMPLETE`。当前 protocol 确为 `PROTOCOL_REVIEW_PENDING`、`UNREGISTERED_BLOCK_POPULATION_R0_R1`。此外注册状态只拒绝两个精确 sentinel revision，再接受任意非空文本；全局 placeholder 扫描只识别 `TODO/TBD/PLACEHOLDER` 字样。因此 `fixture-release-1` 一类语义占位值不被拒绝。`configs/causaltad/protocols/prefix_route_identifiability_v2.json:1`；`prefix_route_protocol_v2.py:200-207,427-485`。 | 从 R0 collector 删除 PASS 字段；注册 revision 改为精确枚举或独立签名的 source-origin certificate；任何 decodable 视频只有在同一 source-origin certificate 中才可进入 population。 |
| **A3** registration 重读 exact protocol、manifest、source、review、population、annotation，而非相信 caller hash | **FAIL** | 正式 CLI 路径会重读 protocol、manifest、worktree 和 signed review，这是已闭合部分：`tools/validate_prefix_route_protocol_v2.py:128-170,205-264`。但公开 `validate_population_bundle`/`validate_r0_bundle` 只调用 `_require_loaded_protocol_record` 和 `_require_signed_pass_review_record`；后者不接收 manifest，不比较 `source_manifest_sha256`，不重跑 `verify_frozen_git_tree`，也不要求 current HEAD。`prefix_route_protocol_v2.py:1937-2051`。因此 path-backed caller record 仍是可达生产输入，无法满足“所有入口均重新验证”。 | 将当前对象级 validator 改为私有核心；公开 population/R0 API 只接受 canonical 文件引用，并在内部固定加载仓库 protocol/manifest、signed review、HEAD/tree。 |
| **A4** R6 完整绑定模型、checkpoint、config、command、ledger、fairness、population、controls、append-only emissions | **FAIL** | `_validate_run_provenance` 确实要求上述引用及哈希，但模型/checkpoint/config/command/environment 只是“非空字节+哈希”；代码不加载 checkpoint，也不把实际 tensor-state digest 与 trace 首尾状态绑定。execution trace 和 ledger 是 caller 提供的 canonical JSON，只检查内部链一致性。`r6.py:1213-1495`。更严重的是 emission 只检查字段、序号和 `emit_frame` 单调；`derive_per_video_cell` 直接添加 `immutable=True`，在线 AP evaluator 使用 `require_ledger=False`。`r6.py:205-303,1749-1806`。 | 加载并规范化 checkpoint state，计算实际 tensor digest 并绑定 trace 首尾；R6 只能从验证通过的 hash-chained `ImmutableEventLedger` 导出 emissions；evaluator 改为 `require_ledger=True`。 |
| **A5** B2 是真正 Hungarian-plus-dustbin | **PASS** | 矩阵维度为 `64 × (target_count+64)`；目标列和 64 个 dustbin 列在同一次 `linear_sum_assignment` 中优化，未再采用“先强制匹配再删除高成本 pair”。`opentad/utils/prefix_route_b2_contract_v2.py:169-208`。对应 hostile test 验证高成本目标 B 可保持未匹配。`tests/test_prefix_route_protocol_v2.py:1101-1120`。 | 无算法性修复；仍需在未来 model-P0 中证明训练调用链实际使用这一实现，但这不改变本项 PASS。 |
| **A6** B2 tie-break 真正改变最优选择并给出唯一确定解 | **FAIL** | tie 项为 `(slot+1)*(column+1)*1e-12`，虽不是简单行常数+列常数，但不是无碰撞的总序。例如三个 slot/column 时，排列 `[0,2,1]` 的乘积和为 `1·1+2·3+3·2=13`，排列 `[1,0,2]` 为 `1·2+2·1+3·3=13`。不同 assignment 仍可同 tie 总和，结果继续依赖 SciPy/版本内部选择。`prefix_route_b2_contract_v2.py:191-203`。 | 对量化整数主成本先求精确最优值，再在全部主最优解中执行明确的逐行 lexicographic constrained assignment；不要以浮点微扰冒充唯一 tie-break。 |
| **A7** decision_bin 与 observation_count、stream clock、合法范围绑定 | **FAIL** | 局部关系 `decision_bin=(observation_count-1)//8` 已实现；无效 `999/8` 也有拒绝测试。但 transition 的全部时钟字段仍由 caller 提供，只有 `previous.last_decision_bin < decision_bin`，没有“当前决策必须是上一合法决策的唯一后继”、重复/跳跃/重放拒绝或 repository-owned clock token。空 track 状态下可直接跳到任意未来合法 bin。`prefix_route_b2_contract_v2.py:318-354`；`tests/...:1187-1197`。 | 将 clock、observation count、decision index、track serial、emission sequence 纳入单一不可分割 `B2StreamState`；只能消费正式 online runner 产生的下一时钟状态。 |
| **A8** fairness 拒绝 `model=None`、自报参数/MAC/update budget 和 caller record | **FAIL** | live producer `_measure_runtime` 确实要求真实 model、optimizer 和 CUDA；但 R6 使用的是序列化消费者 `validate_fairness_audit_record`。该函数主动创建 `ArmRuntimeAdapter(model=None, optimizer=None, ...)`，只从四个 budget 文件重算 update/count 字段；参数量、MAC、显存、延迟和 gradient inventory 均来自 caller row，只检查正数及内部总和。最后从这些自报 rows 重算 checks/status。`prefix_route_fairness_v2.py:651-829`；R6 在 `prefix_route_r6_v2.py:2101-2111` 调用它。 | 序列化 fairness record 只能得到 `UNVERIFIED_SERIALIZED_AUDIT`。科学 PASS 必须由 clean-process live producer 生成，并绑定实际 checkpoint tensor digest、optimizer、输入批次、CUDA profiler 原始记录和 run identity；consumer 不得构造 `model=None`。 |
| **A9** R5 每行按 factor_spec+seed+index+set_name 重生成并 exact-byte 比较 | **PASS** | auditor 使用每行 `factor_spec`、冻结 seed、`sequence_index`、`set_name` 重调 `generate_sequence`，要求完整 canonical JSON byte equality；随后核验 scientific hash、row commitment 和冻结 set commitment。`prefix_route_ood_v2.py:727-813`。测试交换两行 payload/hash 后明确要求拒绝。`tests/...:844-855`。 | 内容生成部分无需修改；只需把最终 R5 scientific certificate 放入正式 protocol/review-bound wrapper，解决 A1。 |
| **A10** bootstrap/统计/terminal 不能 monkeypatch、低层调用或手工 interval 绕过 | **FAIL** | R6 正式 bootstrap 已改用字面量 `10000/2026071707/0.05`，这部分关闭。可是 `_terminal_route_decision` 仍只验证 inference 字段与数值，不验证其来自 raw cells/bootstrap。caller 可设置 eligible family 并为 B4-vs-controls、B4-vs-B2、D1/D2 填入有利区间，得到 `PASS_B4_ROUTE_SURVIVES`。测试本身构造完整手工 inference 并直接调用 terminal。`r6.py:736-754,761-1066`；`tests/...:1343-1395`。此外正式 R6 是同进程 Python API，caller 可在调用前替换 `_sealed_paired_crossed_bootstrap`、`_terminal_route_decision` 或导入的 validator；磁盘 manifest 校验不证明当前 `sys.modules` 中函数对象未被替换。 | terminal 必须只消费带 raw-evidence、cell-set、bootstrap-output commitments 的不可变内部产物；科学 certificate 只能由 fresh isolated process/CLI 生成，并在导入后核对完整 module-origin/source closure。 |
| **A11** manifest 覆盖全部影响正式执行的 imported modules，包括 `__init__` 传播面 | **FAIL** | manifest 包含 leaf `opentad/models/backbones/online_siglip_adapter.py`，但未包含 `opentad/models/__init__.py` 或 `opentad/models/backbones/__init__.py`。正式 cache 工具通过普通包导入 leaf；Python 会先执行这两个父包 initializer。前者导入 builder、dense heads、detectors；后者导入大量其他 backbone，实现和注册面均不在 manifest。`manifest.json:1`；`tools/cache_ontad_features.py:200-206`；`opentad/models/__init__.py:7-13`；`opentad/models/backbones/__init__.py:3-11`。 | 从 clean interpreter 记录每个正式入口实际加载的全部 repository-owned `module.__file__`，将其闭包全部纳入 manifest；运行中出现任何未登记 repo module 即 fail。 |
| **A12** hostile probes 真正覆盖所有上述攻击 | **FAIL** | 已有好的 RED probes：R5 payload 交换、真实 dustbin、错误 decision-bin、无签名 CLI、ASCII 假 MP4。缺失项包括：语义占位 revision；可解码合成视频但无 source-origin；伪造 `validate_fairness_audit_record` PASS；手工完整 terminal inference 必须拒绝；monkeypatch 后正式入口必须拒绝；checkpoint tensor-state mismatch；无 hash-chain emission ledger；父包 import closure；stream repeat/skip/reorder；tie-break 碰撞。现有 terminal 测试反而把手工 inference 作为合法输入。 | 增加下述修复清单中的逐项 hostile tests；所有测试必须断言正式入口拒绝，而不是只检查低层结构或正向 fixture。 |

---

# 4. Gate A verdict

## Gate A terminal decision

**REVISE_PREFIX_PROTOCOL_BEFORE_COLLECTION**

这不是因为当前 config 处于 pending 或尚无签名 PASS，而是因为即使未来填入 registration 并获得签名 review，现有生产代码仍不能证明 fairness、terminal inference、emission provenance 和 import identity 只能来自获批执行链。

## 文档—生产代码冲突

1. 文档宣称“serialized rows are rejected”“only live model/optimizer/CUDA audit may produce the fairness record”。生产消费端却以 `model=None` 重验 caller rows，并允许这些 rows决定 PASS。文档：`PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md:256-275`；生产代码：`prefix_route_fairness_v2.py:651-829`。
2. 文档宣称 R6 只有一个 public entry，且 terminal 在 source chain 闭合后内部计算。生产代码仍保留可直接调用的 inference/terminal helper，并由测试手工构造输入。文档：`PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md:317-351`；生产：`r6.py:761-1066`。
3. 文档宣称 source manifest 绑定正式执行面；实际 manifest 漏掉 leaf import 必经的父包 initializer。
4. 文档宣称 immutable emissions；R6 指标路径实际使用 caller rows、代码附加的 `immutable=True` 和 `require_ledger=False`。

生产代码优先，因此这些文档声明当前均不能作为闭合证据。

## 最小闭合修复清单

| 文件/函数 | 必须修改的最小内容 | 必须新增的拒绝测试 |
| --- | --- | --- |
| `opentad/utils/prefix_route_r0_v2.py::collect_r0_census` | 删除 collector 输出中的 `PASS_R0_COMPLETE`；collector 只返回 diagnostics。正式 envelope 才能添加 scientific status。低层 caller-selected bootstrap 结果不得成为 certificate。 | `test_low_level_r0_census_never_emits_pass_or_certificate`；`test_unregistered_low_level_r0_cannot_be_relabelled_as_formal`。 |
| `opentad/utils/prefix_route_ood_v2.py::audit_sequence_sets` | 保留逐行重生成，但返回值标记为 diagnostic；新增 protocol/manifest/review-bound 的 R5 bundle wrapper，只有 wrapper 可产生正式 R5 certificate。 | `test_caller_dict_r5_audit_is_diagnostic_only`；`test_formal_r5_reloads_manifest_review_and_head`。 |
| `opentad/utils/prefix_route_protocol_v2.py::_require_signed_pass_review_record`、`validate_population_bundle`、`validate_r0_bundle` | 公共入口内部重新加载 canonical protocol、manifest、review 和 tree；当前对象级验证器改为私有。校验 attestation 的 manifest hash、commit/tree、current HEAD。 | `test_population_low_level_objects_cannot_bypass_manifest_or_tree`。 |
| 同文件 source-registration validator | `release_revision` 改为精确冻结值或 reviewer-signed source-origin record；增加 source-origin commitment。仅“非空字符串+文件可解码”不足以证明官方来源。 | `test_semantic_placeholder_revision_rejected`；`test_decodable_synthetic_population_rejected_without_source_origin_attestation`。 |
| `opentad/utils/prefix_route_fairness_v2.py::validate_fairness_audit_record` | 删除 `model=None` 验证模式产生 PASS 的能力。序列化记录只能是未验证存档；正式 PASS 必须绑定 clean-process live producer、实际 checkpoint tensor digest、optimizer、raw profiler trace、CUDA identity 和输入批次。 | `test_forged_serialized_fairness_pass_is_rejected_even_when_internally_consistent`；`test_fairness_checkpoint_state_must_match_profiled_model`。 |
| `opentad/evaluations/prefix_route_r6_v2.py::_validate_run_provenance` | 以安全方式加载 checkpoint，计算规范 tensor-state digest，并与 execution trace 首尾逐字节绑定；config/command/environment 应有固定 schema，而非只要求非空。 | `test_arbitrary_nonempty_checkpoint_bytes_cannot_satisfy_run_provenance`；`test_trace_state_hash_must_derive_from_checkpoint_tensors`。 |
| 同文件 emission path | raw evidence 不再携带普通 emissions list；改为 immutable ledger reference+commitment。调用 `ImmutableEventLedger` verifier，验证 genesis、previous hash、row hash、sequence、禁止修改和 provenance，再从 verified ledger 导出 evaluator rows。`OnlineAPBudgeted(require_ledger=True)`。 | `test_r6_rejects_emissions_without_hash_chained_ledger`；`test_r6_rejects_rewritten_or_reordered_ledger`；`test_r6_does_not_synthesize_immutable_true`。 |
| 同文件 bootstrap/terminal | 删除“独立 inference dict → scientific decision”接口。terminal 输入必须包含 raw-evidence digest、cell-set digest、dataset-metric digest 和 sealed bootstrap digest，并由正式 runner现场构造。 | `test_hand_built_complete_inference_cannot_return_route_pass`。 |
| 新的 repository-owned R6 CLI/runner | 科学 certificate 只能由 fresh isolated process 产生；固定 executable、argv、repo root、HEAD、环境及 module origins。调用者不得传 callback/module object。 | `test_monkeypatched_terminal_or_validator_is_rejected_by_clean_runner`。 |
| `prefix_route_b2_contract_v2.py` | 用精确二阶段 lexicographic assignment 替代浮点乘积微扰；引入不可分割 per-stream clock/state，拒绝重复、跳 bin、倒序及 caller 重置 sequence。 | `test_b2_tie_collision_has_unique_frozen_solution`；`test_b2_clock_rejects_repeat_skip_reorder_and_sequence_reset`。 |
| protocol JSON 与 manifest JSON | 加入正式入口 clean-import closure 规则；至少纳入 `opentad/models/__init__.py`、`opentad/models/backbones/__init__.py` 及其全部 repo-owned transitives。 | `test_manifest_equals_clean_process_repository_import_closure`。 |
| `tests/test_prefix_route_protocol_v2.py` | 将现有“手工完整 inference 可被 terminal 接受”的测试改为拒绝断言；加入上述所有 hostile probes。 | 所有攻击必须在正式入口层 RED，而非只验证 helper 的字段检查。 |

以上修复完成前，不能生成新的 population、R0、R1、fairness、R5 或 R6 scientific certificate。

---

# 5. Strict-online mismatch table

**STOP-CHAIN: NOT REACHED.**

Gate A 未通过，因此未读取或评判 ChronoTransport 的 actions、cache、transport、runtime、scheduler、risk、protocol、cost、profiler、Stage B/C 配置或测试。不能基于本轮给出任何 strict-online mismatch 结论。

---

# 6. CT reuse/rewrite/delete code map

**STOP-CHAIN: NOT REACHED.**

没有形成“可直接复用 / 仅复用思想 / 必须重写”分类。此时给出分类会违反先闭合 Prefix Protocol 的依赖顺序。

---

# 7. B2/B4/等待路线辨识的选择

**STOP-CHAIN: NOT REACHED.**

本轮不能在“只用 B2、同时 B2/B4、等待 B0–B4、完全拒绝”之间作 CT 实验选择。任何选择都会提前把尚未可信的 Prefix evidence infrastructure 当作已闭合基础。

---

# 8. 最小 Streaming CT kill-test 合同

**STOP-CHAIN: NOT REACHED；不得编写、冻结或批准。**

候选 P0–P5、成本边界、阈值、population、种子数和统计规则均未进入审核。它们不能被本答复引用为获批规格。

---

# 9. Claim map

| Claim family | 当前状态 | 本轮允许的表述 |
| --- | --- | --- |
| **event lifecycle / identity** | **BLOCKED_BY_PROTOCOL** | B2 的局部 dustbin assignment 已修复，但 Prefix Route scientific evidence chain 仍未闭合；不能形成路线 effectiveness 或 identifiability claim。 |
| **compute reduction** | **NOT_REACHED** | 未进入 CT 生产代码，不能声称适配、节省或不适配。 |
| **transport mechanism** | **NOT_REACHED** | 未评判 TRANSPORT 与 HOLD、feature propagation 或 reuse gate 的差异。 |
| **adaptive scheduling** | **NOT_REACHED** | 未审核 scheduler 的因果输入、训练来源或成本匹配。 |
| **joint Prefix+CT** | **FORBIDDEN_AT_THIS_GATE** | 不能讨论“结合更强”，也不能建立联合实验。 |
| **end-to-end Online TAL** | **BLOCKED** | 当前没有可信的 Prefix route certificate，更没有 CT 的 arrival-to-emission、p95、endpoint 或短动作证据。 |

用户列出的十个跨路线核心判断均为：

`NOT_REACHED_BY_STOP_CHAIN`

这不是默认支持或默认拒绝 CT，而是拒绝在失效的前置证据层上继续增加变量。

---

# 10. 下一项唯一允许任务

创建一个**仅修复上述 Gate A blockers** 的新 immutable Prefix Route commit，并对该新 SHA 重新执行独立、零信任 Protocol V2 closure review。

该提交的范围应严格限制为：

* formal-entry exclusivity；
* source/registration identity；
* live fairness provenance；
* checkpoint-state 与 immutable emission-ledger binding；
* sealed terminal/clean-process execution；
* transitive import manifest；
* B2 exact tie-break 与 stream clock；
* 对应 hostile RED tests。

不得在同一提交中加入 ChronoTransport、B4 模型实现、实验配置、数据收集或性能结果。

---

# 11. 阻断动作清单

在新 Prefix closure review 返回 `PASS_PREFIX_PROTOCOL_FOR_ROUTE_EVIDENCE_DESIGN` 之前，以下动作全部阻断：

* 编写、冻结或批准 Streaming ChronoTransport kill-test；
* 修改 ON-TAD 以接入 CT；
* 为 CT 选择 B2/B4 head；
* source registration、population certificate、R0/R1 collection；
* fairness/profile artifact 生成；
* B0–B4 model-P0、训练、GPU、profiling；
* checkpoint、预测或效果表检查；
* 以本轮审查支持 CT、Prefix 或二者联合的论文主张；
* 把代码存在、测试存在或 internally consistent JSON 当作科学证据。

REVISE_PREFIX_PROTOCOL_BEFORE_ANY_CT_WORK
