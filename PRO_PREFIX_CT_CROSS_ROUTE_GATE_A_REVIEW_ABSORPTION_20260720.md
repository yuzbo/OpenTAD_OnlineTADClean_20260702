---
type: review_absorption
status: absorbed_with_qualified_implementation_guidance
reviewed_commit: 48e78e43a77822ee0e51cace9ad758ca935c1450
reviewed_tree: 8b5359f3d4dd10c4d629e16f7244a759c1e6846e
updated: 2026-07-20
---

# Prefix Route Gate A 与 ChronoTransport Stop-Chain 审计吸收

## 最终判断

我**完全认可审查的核心裁决和 stop-chain 后果**：

> `REVISE_PREFIX_PROTOCOL_BEFORE_ANY_CT_WORK`

Gate A 在 exact commit
`48e78e43a77822ee0e51cace9ad758ca935c1450` 上没有闭合。A5 的增强
Hungarian-plus-dustbin 和 A9 的 R5 逐行重生成已经成立，但它们不能抵消
fairness、terminal、emission ledger、formal-entry、source registration、
import closure、B2 tie-break 和 stream clock 中仍然存在的 false-PASS 面。

因此：

- 当前不能 collection、registration、R0/R1、fairness artifact、profile、
  training、GPU 或读取模型结果；
- 当前不能把 Prefix Route、B2、B4 或 ChronoTransport 写成已获科学支持；
- 本轮没有到达 ChronoTransport 代码适配、Streaming CT kill-test 或
  B2/B4 head 选择；
- 下一项唯一实施任务是一个只修 Gate A 的零 GPU、无实验、不可变代码
  快照，再由独立审查者审核 exact commit。

我不把“完全认可核心裁决”解释成逐字照搬每一条实现措辞。下文记录了
必须保留的技术边界；这些限定不改变 Gate A 的 FAIL，也不放宽任何阻断。

## 来源与记录完整性

- 原始附件：
  `ae85b0a1-c69b-463b-8b39-36d85a22121b/pasted-text.txt`
- 原始附件 SHA-256：
  `6A05072FA0168E7E17C0DB9DF04B586D0706C991403CCAA507854987EABB8022`
- 原始附件大小与行数：32,229 bytes，172 行，UTF-8、CRLF。
- 仓库归档：
  `PRO_PREFIX_CT_CROSS_ROUTE_GATE_A_REVIEW_20260720.md`
- 仓库归档 SHA-256：
  `DBE6E2BF10A0A1C1316D96958758417BCC9610B7DFD194EFE68A81D5423589D9`
- 仓库归档大小与行数：22,078 bytes，172 行，UTF-8、LF。

仓库归档完整保留了标题、A1–A12、Gate A 裁决、修复清单、stop-chain、
claim map、下一步和终局 token；为符合仓库 LF 和紧凑 Markdown 表格规则，
删除了原附件表格中的对齐空格，所以它是**内容完整的规范化归档**，不是
原附件的 byte-identical 副本。原始附件 SHA 单独保留，禁止混用两个哈希。

## 本地 exact-commit 复核

本次吸收没有运行测试、脚本、训练、推理、profile 或 GPU，也没有读取
checkpoint、预测或既有实验结果。只在 detached 审核工作树中读取了
`48e78e43...` 的生产代码、协议、manifest 和测试。

| 项 | 独立复核 | 吸收状态 |
| --- | --- | --- |
| A1 | `collect_r0_census` 返回 `PASS_R0_COMPLETE`；`audit_sequence_sets` 返回 certificate-like true；terminal 接受 inference dict | 接受 FAIL |
| A2 | 正式 UNREGISTERED 路径已阻断，但低层 R0 仍可给 PASS；registered revision 只做有限 sentinel/非空校验 | 接受 FAIL |
| A3 | population/R0 对象级 API 重验 protocol/review，但没有在该边界重验 manifest 与完整执行树 | 接受 FAIL |
| A4 | R6 读取 artifact bytes 和自报哈希链，但不从 checkpoint tensors 导出 trace 首尾；普通 emission list 被代码加上 `immutable=True`，且 `require_ledger=False` | 接受 FAIL |
| A5 | 一个 `64 × (targets+64)` 增强矩阵进入同一次 Hungarian | 接受 PASS |
| A6 | `(slot+1)*(column+1)*1e-12` 存在 assignment 总和碰撞，不是唯一总序 | 接受 FAIL |
| A7 | bin/count 局部公式成立，但 caller 仍提供 clock、serial、sequence 和前态，不能证明连续消费 | 接受 FAIL |
| A8 | R6 调用的 serialized fairness consumer 明确构造 `model=None, optimizer=None`，其余核心资源标量来自 row | 接受 FAIL |
| A9 | R5 逐行按冻结输入重生成并比较 canonical bytes 与 commitments | 接受 PASS |
| A10 | terminal 只验证完整 inference schema/数值；测试也直接手工构造 inference 调 helper；同进程 monkeypatch 面存在 | 接受 FAIL |
| A11 | manifest 有 leaf adapter，却没有 Python 导入必经的两个父包 `__init__.py` | 接受 FAIL |
| A12 | 已有若干 hostile tests，但审查列出的 forged fairness、完整手工 terminal、tensor mismatch、ledger、import closure、clock、tie collision 等攻击没有正式入口拒绝覆盖 | 接受 FAIL |

这项复核证明审查的核心证据与 exact commit 一致。它不是一次新的完整
代码审计，也没有生成独立 Protocol PASS/REVISE certificate。

## 必须原样吸收的 P0 修复原则

### 1. 正式结论的唯一能力边界

低层 collector、auditor、schema validator 和统计 helper 可以计算
diagnostics，但不能返回 `PASS`、`AUTHORIZED`、`certificate` 或可被正式
consumer 直接升级为这些状态的字段。正式科学状态只能由 repository-owned
runner 从 canonical protocol、manifest、signed review、raw evidence 和
execution provenance 一次性导出。

### 2. Fairness 不能由自报标量自证

任意 caller JSON 即使内部自洽，也不能证明参数、梯度、MAC、显存、延迟、
输入批次、优化器事件和模型状态确实来自同一 run。正式 fairness 必须绑定：

- 实际模型与 checkpoint tensor state；
- 实际 optimizer state 和每次 update 的输入；
- raw profiler trace 与执行设备；
- frozen budget records；
- run identity、protocol、manifest 和 review。

### 3. Terminal 不能消费独立的有利区间

手工 inference dict 只能用于无科学状态的单元测试或 diagnostics。正式
terminal 必须从被 commitments 绑定的 raw cells、metric cells 和 sealed
bootstrap 唯一导出，调用者不能注入 interval、estimate、eligible family
或 terminal helper。

### 4. Emission 不可变性必须由 ledger 证明

代码附加布尔值 `immutable=True` 不是证据。正式 R6 只能从验证通过的
append-only ledger 导出 evaluator rows，至少核验 genesis、previous hash、
row hash、stream/sequence 单调性、run provenance、禁止重排/重写，并使用
`require_ledger=True`。

### 5. 执行源码闭包必须完整

manifest 必须覆盖正式进程实际加载并能影响结果的所有 repository-owned
代码，包括 Python 父包 initializer。正式执行出现未登记的 repo module
必须 fail closed。

### 6. B2 必须有唯一 assignment 和不可拆分的流状态

主成本先规范化为精确可比较的值，再执行明确的 lexicographic tie-break；
不能依赖浮点 epsilon 或 SciPy 的隐含选择。clock、observation count、
decision index、track serial、emission sequence 和前态必须属于一个正式
runner 管理的状态机，拒绝未授权的重复、跳跃、倒序和 reset。

## 对具体实现措辞的六项限定

这些限定用于防止下一轮把修复做成新的形式主义漏洞；它们不改变任何
FAIL 或 block。

### Q1. “公开函数改私有”不是安全边界本身

Python 下划线不能阻止调用。低层 helper 可以保留为公开诊断 API，前提是
它的 schema 从结构上不含正式状态，而且正式 runner 不接受 caller 提供的
helper 输出。真正的边界是 capability separation 和 provenance closure，
不是函数命名。

### Q2. “fresh isolated process”只强制用于正式 certificate

普通单元测试和纯函数计算不必每次启动子进程；但任何可授权 collection、
route claim 或 scientific PASS 的产物必须由干净进程产生，并绑定
executable、argv、repo、exact commit/tree、环境和 module origins。

### Q3. 验证 exact commit/tree，不绑定移动分支的偶然 HEAD

正式 run 必须确认其执行字节属于被审查的 immutable commit/tree；detached
HEAD 是合法状态。不能把“当前移动分支 HEAD 必须相等”当成必要条件，否则
分支前移会使历史证书失效。正确要求是执行 checkout 的 exact HEAD/tree
与 attestation 相等、工作树干净且 source closure 相等。

### Q4. 序列化记录可以被验证，但不能自证

审查提出 serialized fairness 一律返回 `UNVERIFIED_SERIALIZED_AUDIT`，
其目的正确：禁止任意 caller rows 自升级。更精确的实现是：

- 裸记录只能是 unverified archive；
- 由 live producer 生成、带完整 raw artifacts、hash chain 和签名/attestation
  的序列化 bundle，可以由独立 consumer 离线验证后获得正式状态；
- consumer 不能仅凭记录内自报哈希和标量给 PASS。

否则会错误地把“可移植、可复核的证据包”也永久降为不可验证。

### Q5. CUDA 身份只证明设备相关资源量

参数量、checkpoint tensor digest、optimizer-event 计数等可在 CPU 上独立
验证；GPU/CUDA 原始记录对 GPU MAC、显存和延迟等设备相关指标是强制的。
最终 combined fairness PASS 仍必须具备其声称的全部 GPU 证据，不能用
CPU 验证替代 latency/profile。

### Q6. Hash chain 只证明篡改可检测，不等于物理不可变

正式 ledger 还必须有 append-only writer、唯一 sequence 分配、原子写入、
禁止覆盖的 artifact policy 和完整 replay verifier。仅加入 previous hash
仍可能允许整条链被重新生成；链头/链尾还必须绑定 run attestation 和外部
不可变 evidence bundle。

## 不应在本轮扩大或误改的内容

- 不修改 ChronoTransport 代码，不生成 Streaming CT protocol。
- 不实现 B4 模型，不改实验 config，不开训练或 profile。
- 不借 Gate A 修复重新设计 D1/D2、阈值、population、seed 或统计故事。
- 不删除 A5/A9 已闭合的实现；只补它们到正式执行链的绑定。
- 不把此前本地测试通过重写为科学证据。
- 不读取模型结果来决定修复优先级。

## 下一项任务的严格范围

新 immutable Prefix Route commit 只允许包含：

1. formal-entry/capability separation；
2. source-origin 与 registration identity；
3. live fairness 和 checkpoint tensor-state provenance；
4.真实 append-only emission ledger；
5. sealed bootstrap/terminal 与 clean-process formal runner；
6. exact execution import closure；
7. B2 exact tie-break 和 stream state machine；
8. 对应 hostile RED/closure tests；
9. 同步协议、manifest 和研究记忆。

完成后先跑零 GPU focused checks，再发布 exact commit 供新的零信任
Protocol V2 closure review。只有该审查明确返回
`PASS_PREFIX_PROTOCOL_FOR_ROUTE_EVIDENCE_DESIGN`，才讨论下一个被授权的
窄范围；它也不自动授权 CT、collection、profile、training 或 GPU。

## 当前状态

- Prefix Protocol：`revise_required`
- Gate A：`failed`
- B2 local contract：`partially_closed_not_scientifically_bound`
- B4：`designed_candidate_only`
- ChronoTransport strict-online adaptation：`not_reached`
- Streaming CT kill-test：`not_designed_not_authorized`
- experiment：`not_running`
- scientific result：`none`
- paper-ready claim：`none`

最终吸收 token：

`REVISE_PREFIX_PROTOCOL_BEFORE_ANY_CT_WORK`
