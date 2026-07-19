# Prefix-Route Protocol V2 Round 4 Independent Review

**审查身份与锚点**
- Reviewer / task ID：`019f6f63-496d-75a0-a77b-91425a8e7ea1`
- 完成时间：`2026-07-17T15:35:19.376Z`
- 独立 detached checkout：`C:\Users\skywalker\.codex\tmp\protocol-v2-round4-48e78e43a778\audit-src`
- Commit：`48e78e43a77822ee0e51cace9ad758ca935c1450`
- Tree：`8b5359f3d4dd10c4d629e16f7244a759c1e6846e`
- Protocol SHA-256：`493a132bbc9998ad7dd1d4756ec882dee858ac8c739dbf96c618b1459be3ac65`
- Policy-lock SHA-256：`c9594ff7ab2e89549e512f5528bddefe127ecda12d3c8b900f3e0e5d9bf598c0`
- 32-entry manifest SHA-256：`05bf8a8925b465522bcb7470d1810d2421b99c3f7e15512ad3c87c6435aa8c25`
- 上轮报告 SHA-256：`039fcb7868f31da9a7153dadf6cb60e29c14916dd6dba81f926f7ee2b28a28a9`
- 远端分支、commit/tree、canonical protocol、独立重算 policy-lock、32 个 Git blob 字节及 LF 均完全一致；checkout 始终 clean。未访问禁用证据，未训练、profile 或使用 GPU。

**命令与结果**
- `git ls-remote`、detached checkout、`git rev-parse HEAD/HEAD^{tree}`：exit `0`，锚点一致。
- 两次仓库外 `build-manifest --output ...`：exit `0`，均为 32 entries、相同 3,713 bytes 和预期 SHA。
- `validate-protocol`：exit `0`，返回 `PROTOCOL_V2_VALID_REVIEW_REQUIRED`。
- 无签名 `authorize-collection`：exit `2`，`authorized=false`。
- V2 专测：`46 passed, 1 warning`，exit `0`。
- 十文件 focused 回归：`178 passed, 1 skipped, 1 warning`，exit `0`；skip 为隔离环境 Torch 不可用。
- `hostile_probes_round4.py`：exit `0`；结果见下。
- 递归 import closure：exit `0`；58 个可达本地 Python 文件中 33 个未绑定。
- 32 项 `git ls-files --eol`：全部 `i/lf w/lf attr/text eol=lf`。
- `git diff --check b0ac8b1..48e78e4`、worktree `git diff --check/status`：exit `0`，clean detached。
- 附加 symlink pytest：exit `4`，在测试收集阶段因本机 Torch `c10.dll` 初始化失败，测试没有执行，不能视为协议 PASS。无 Torch 的替代探针确认绝对路径和 `..` 逃逸被拒；当前账户无创建 symlink 权限。V2 内置 Windows 8.3 回归实际通过。

**P0 发现**
1. **R6 仍接受完全自填、未实测的科学证据链。**

   序列化 fairness 验证器以 `model=None/optimizer=None` 只解析自填记录并可返回 fairness PASS：`opentad/utils/prefix_route_fairness_v2.py:651`、`opentad/utils/prefix_route_fairness_v2.py:760`。Control validator 只检查两个任意 SHA 格式及“不相等”，不读取 source/constructed payload 或重执行算法：`opentad/evaluations/prefix_route_r6_v2.py:1079`。Run provenance 只核对作者自填哈希链；模型 artifact 哈希与 event model-state 哈希没有字节关系：`opentad/evaluations/prefix_route_r6_v2.py:1213`、`opentad/evaluations/prefix_route_r6_v2.py:1407`。

   调用者 JSON emissions 被直接标记 `immutable=True`，OnlineAP 明确使用 `require_ledger=False`：`opentad/evaluations/prefix_route_r6_v2.py:291`、`opentad/evaluations/prefix_route_r6_v2.py:1800`。独立 probe 在零模型、零 CUDA 下得到 fairness PASS，并让任意 control hashes 与伪 run provenance 通过；初始 artifact SHA 与声称的 model-state SHA 不同。

2. **完整手写 inference 仍可产生科学 PASS。**

   Schema validator 只检查字段、常量和区间形状，不校验 `arm_estimates`、interval estimate/bounds 与 bootstrap 原始样本的一致性：`opentad/evaluations/prefix_route_r6_v2.py:761`。终局函数直接消费该 dict：`opentad/evaluations/prefix_route_r6_v2.py:907`。Probe 令所有 arm estimates 为零、手填正区间，仍返回 `PASS_B4_ROUTE_SURVIVES`、`route_claim_allowed=true`。不完整 dict 和 `resamples=1` 虽已被拒，但未杀死完整伪造路径。

**P1 发现**
1. **B2 局部算法已修，但未形成可验证的同一执行链。** Augmented 64-dustbin、risk set、坐标检查本身正确：`opentad/utils/prefix_route_b2_contract_v2.py:58`、`opentad/utils/prefix_route_b2_contract_v2.py:120`、`opentad/utils/prefix_route_b2_contract_v2.py:316`。但三函数在生产代码无调用者，R6 的 B2 run 只接收 provenance 与 emissions，不绑定 risk-set/cost/assignment/target/transition transcript：`opentad/evaluations/prefix_route_r6_v2.py:1886`。
2. **No-learning control 与强制 optimizer ledger 自相矛盾。** `LEDGER_ONLY` 冻结为 `learning=False`：`opentad/evaluations/prefix_route_r6_v2.py:119`，但所有 control 都必须有正数 optimizer events 且每次改变模型状态：`opentad/evaluations/prefix_route_r6_v2.py:1332`、`opentad/evaluations/prefix_route_r6_v2.py:1460`。合法实现不可执行，伪训练账本反而通过。
3. **历史视频只证明可解码，不证明身份。** 验证仅打开首帧并检查非空几何：`opentad/utils/prefix_route_protocol_v2.py:1903`。Inventory 约束 ID 唯一，却不要求 artifact SHA 唯一或与官方 per-video manifest 对应：`opentad/utils/prefix_route_protocol_v2.py:2171`。同一个合法 MP4 可被登记为多个历史 ID。

**P2 发现**
- R1 import-time source closure 未闭合。`cache_ontad_features.py` 导入 adapter：`tools/cache_ontad_features.py:203`，必然先执行未绑定的 `opentad/models/__init__.py:5` 与 `opentad/models/backbones/__init__.py:1`。递归检查得到 33 个未绑定可达本地文件。
- 作者测试中存在明确假正向 fixture：任意 `c*64/d*64` control hashes 被断言接受：`tests/test_prefix_route_protocol_v2.py:1226`。现有 fairness 测试只阻止顶层 serialized API，却未杀死 R6 实际调用的 record validator。

**六类 Probe**

| 类别 | 结果 |
|---|---|
| UNREGISTERED R0 / public R6 | CLOSED：均 fail-closed |
| R5 3,800、七组 hash、payload/metadata/no-effect | CLOSED：全部 hash 重现，篡改均拒绝 |
| B2 A/B、dustbin、delayed target、bin=999 | 局部 CLOSED：A→slot0、B→dustbin、999 拒绝 |
| Serialized fairness | FALSE PASS：零模型/零 CUDA 得到 fairness PASS |
| Control/run provenance | FALSE PASS：任意 transcript 与伪账本通过 |
| R6 terminal | PARTIAL：不完整输入拒绝，但完整手写 dict 得到 route PASS |

**上一轮 Closure Matrix**

| 项 | 状态 |
|---|---|
| V1 P0-1 reviewer/signature 自伪造 | CLOSED |
| V1 P0-2 population/R0/R1 状态与来源 | PARTIAL |
| V1 P1-1 全局 fail-closed | OPEN |
| V1 P1-2 R0 estimand | CLOSED |
| V1 P1-3 exposure controls | CLOSED |
| V1 P1-4 fairness、B2、D1/D2 | OPEN |
| V1 P1-5 controls/R5 | PARTIAL：R5 closed，controls open |
| V1 P1-6 R6 inference/terminal | OPEN |
| V1 P2-1 source binding/tests | OPEN |

**最小修复合同**
1. R6 必须从 hash-chained immutable ledger 重建 emissions，并启用 `require_ledger=True`。
2. 每个 optimizer event 必须提供实际输入与前后 model/optimizer state artifacts；验证器重放全部 event，而非信任哈希文本或只重放首 event。
3. Fairness record 只能由同一 run-bound live replay 产生；删除可独立返回 PASS 的 serialized validator。
4. Control 必须绑定实际 source/constructed bytes并重执行冻结算法；no-learning controls 使用明确的零 optimizer-event schema。
5. B2 必须保存并重放逐 decision risk set、cost matrix、assignment、first-emission target、transition，且由该链唯一导出 emissions。
6. Terminal 必须从 source-bound cells 重新执行 sealed bootstrap；禁止独立 dict 入口。Manifest 必须纳入完整递归 import-time closure并增加 fail-closed 回归。
7. Population 注册须绑定独立冻结的官方 per-video manifest，并拒绝未解释的重复 artifact hashes。

当前 `UNREGISTERED` 状态仍阻断全部 R0/R1；本报告不授权 source registration、collection、模型实现、profile、训练或 GPU。由于裁决为修订，未生成或签署 PASS attestation。

REVISE_PROTOCOL_BEFORE_COLLECTION
