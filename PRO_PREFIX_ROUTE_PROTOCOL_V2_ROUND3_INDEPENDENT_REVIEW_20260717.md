**固定锚点**
- 远端分支与 detached `HEAD`：`b0ac8b1c1d29fc252b472a6dc64c854524426593`
- Tree：`5183da4a3c382df449eddab89c1eee01cb374982`
- Protocol SHA-256：`77dac32a878a07316e3384b7bae2d7768d57830479a3916eaf0273b788c8ecf8`
- 22-entry manifest SHA-256：`ddd2dbaff706d12e8ebaed7e40aa53919b0836e623ace9efa5dfd80e1b1cdc27`
- 上轮报告 SHA-256：`e411804f5bb744cbbe776a77b63701957ec607a3ca2293b4d523a23a656ec4cf`

**P0 发现**
1. **R6 仍可由完全自组、无真实运行来源的证据产生科学 PASS。**
   `evaluate_r6_raw_evidence` 接受任意 protocol 路径和仅含 arm/seed/video/emissions 的记录，却不核验 frozen commit/tree/manifest、review、模型/权重/命令、fairness 或运行 ledger（`opentad/evaluations/prefix_route_r6_v2.py:1191-1314`）。调用者 emissions 被代码直接标记为 `immutable=True`（`:227-241`）；R0 只验证自哈希 envelope（`:1033-1116`）。作者测试本身以伪 annotation/review/population 哈希和手工 GT-perfect emissions 得到 PASS（`tests/test_prefix_route_protocol_v2.py:1083-1227`）。
   修复：R6 必须只接受签名、commit-bound 的不可变运行 ledger，并绑定模型、权重、配置、命令、fairness、R0 population certificate 和每个 control 的构造记录。

2. **当前 UNREGISTERED 状态仍可通过公开低层 API 生成 `PASS_R0_COMPLETE`。**
   `derive_r0_envelope` 不检查 source-registration 状态、签名 review 或 canonical-213 population，只信任传入 dict（`opentad/utils/prefix_route_protocol_v2.py:2073-2129`），且被公开导出（`:2207-2234`）。探针用当前 UNREGISTERED protocol 和单视频伪 population 得到 PASS。注册语义还允许占位 revision、任意 40-hex registration commit（`:427-471`），并允许 population 与 R1 annotation 哈希不一致（`:699-772`）。历史 `.mp4` 只要求非空（`:1902-1950`）；正向测试使用 ASCII 文本冒充视频（`tests/test_prefix_route_protocol_v2.py:361-457`）。
   修复：所有能产生 PASS/certificate 的入口必须重新读取 frozen protocol、signed review 和 source bytes；注册必须锁定真实 release revision、审查提交，并解码验证 artifact 身份及 population/R1 同源性。

**P1 发现**
1. **B2 assignment 不是声明的 Hungarian-plus-dustbin。** 矩阵没有 dustbin 行/列，而是先强制匹配所有 target、再删除成本大于 dustbin 的 pair（`opentad/utils/prefix_route_b2_contract_v2.py:169-197`）。这会把可合法匹配的 target 从最优 slot 挤走。所谓 tie-break 是可分离加项，其总和对 permutation 不变（`:190-206`）。`decision_bin` 也未绑定 observation count（`:321-335`）。
   修复：使用真正的 augmented dustbin assignment、非可分离确定性 tie-break，并把 assignment、risk set、transition 和训练目标绑定为同一执行链。

2. **Fairness 的精确预算仍来自作者可自填记录。** 四类记录只有自带哈希（`opentad/utils/prefix_route_fairness_v2.py:104-116,145-263`）；实时测量仅重放第一项 optimizer event（`:322-370`），其余 event、trial、token 和 calibration 声明未与训练状态绑定。模型、权重及 fairness 输出也未绑定 R6 emissions。
   修复：绑定完整 optimizer-event ledger、每次输入/token、前后模型状态哈希及 run identity，并让 R6 验证同一模型运行。

3. **R5 内容可以与 factor 标签交换。** `sequence_sha256` 排除了 `factor_spec/set_name/index/seed`（`opentad/utils/prefix_route_ood_v2.py:495-511`）；审计只分别检查内容哈希和标签平衡，不按 metadata 重跑 generator（`:659-798`）。交换两行 payload/hash、保留原 factor 标签后，完整 3,800 序列仍通过。
   修复：对每行按精确 `factor_spec+seed+index+set_name` 重新生成并逐字节比较，同时把这些字段纳入 commitment。

4. **R6 的固定统计参数和终局仍可绕过。** “固定”10,000 次由可变模块全局量提供（`prefix_route_r6_v2.py:82-87,672-687`）；作者测试明确 monkeypatch 为 64 后仍断言 PASS（`tests/test_prefix_route_protocol_v2.py:1207-1227`）。`_terminal_route_decision` 可直接消费手工 intervals 并返回 PASS（`:697-856`）。

**P2 发现**
- Manifest、Git blob 和 LF 重现均通过，但 manifest 未绑定会在导入时执行的 `opentad/utils/__init__.py`；验证只覆盖 manifest entries（`prefix_route_protocol_v2.py:1487-1525,1542-1593`）。测试还把伪 population/R6 正向 fixture 固化为 PASS，未杀死上述 bypass。
- Windows 8.3 回归本轮通过；`.gitattributes:1-8`、JSON/JSONL LF、仓内/仓外 manifest 重建均已闭合。

**六类 hostile probes**
1. 占位 release、任意 registration commit、population/R1 不同 annotation：**接受**。
2. 当前 UNREGISTERED + 单视频伪 population：返回 `PASS_R0_COMPLETE`。
3. R5 两行 payload/hash 与 factor 标签错配：完整 3,800 audit **接受**。
4. B2：A 被分到 slot 1 而非最优 slot 0；`decision_bin=999, observation_count=8` **接受**。
5. `model=None` 的自填 fairness budget records：**接受**并生成预算证据。
6. 伪 R6 envelope/emissions、bootstrap resamples=1：返回 `PASS_B4_ROUTE_SURVIVES`；手工 interval dict 同样返回 PASS。

**V1 Closure Matrix**

| V1 项 | 状态 |
|---|---|
| P0-1 独立 review 可自伪造 | CLOSED |
| P0-2 population/R0/R1 状态可断言 | OPEN |
| P1-1 语义 fail-closed | OPEN |
| P1-2 R0 estimand 不唯一 | CLOSED |
| P1-3 exposure 控制不足 | CLOSED |
| P1-4 B0-B4 fairness、D1/D2 | OPEN |
| P1-5 controls/R5 | OPEN |
| P1-6 R6 inference/terminal | OPEN |
| P2-1 source binding/tests | OPEN |

**测试结果**
- Manifest 仓内及仓外重建：22 entries，2,671 bytes，逐字节相同。
- `validate-protocol`：成功并返回预期 protocol/manifest 哈希。
- 无 review 的 `authorize-collection`：blocked，exit `2`。
- V2 专测：`40 passed, 1 warning`。
- 十文件 focused 回归：`99 passed, 4 skipped, 1 warning`；4 项因隔离子进程 Torch import 失败。
- Manifest 与全部 22 个绑定文件：`i/lf w/lf attr/text eol=lf`。
- `git diff --check`、cached check：exit `0`；detached checkout clean。
- 未访问真实 R0/R1、模型输出或 checkpoint；未训练、profile 或使用 GPU。

REVISE_PROTOCOL_BEFORE_COLLECTION
