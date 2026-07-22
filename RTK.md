# RTK: Online/Causal TAD Clean Repo

## 研究第一优先级（最高规则）

- 本项目的第一目标是提出**性能更好、创新性更强的 On-TAD 模型算法**，不是建设一个更复杂、更通用或形式上完美的工程框架。发生冲突时，模型创新、有效学习和可测性能增益永远优先。
- 每一项代码工作都必须直接服务于至少一个明确的模型假设、训练问题、对照实验或核心指标；不能说明预期科学信息增益的工程工作默认不做。
- 工程实现遵循“最小可验证”原则：只做到能忠实运行官方基线、公平训练候选模型、可靠评测和复现实验。不要为了接口统一、抽象完整、覆盖所有未来场景或追求零瑕疵框架而延迟模型实验。
- 优先复用和忠实参考前沿官方方法，只在能够形成新学习机制或解决明确性能瓶颈的核心位置修改；避免大规模重写训练框架、数据层和无关基础设施。
- 时间和算力优先投入到模型诊断、损失与分配机制、表示与记忆设计、直接父方法对照、关键消融和并行候选实验。没有独立增益的模块应尽快删除，不因已经投入工程成本而保留。
- 只有当问题会使科学结论无效或阻断实验时，才优先修工程，包括：无法训练/评测、基线不忠实、数据或因果泄漏、指标错误、结果不可复现。修复到恢复可信实验即可，不继续扩展为通用工程项目。
- 任何阶段都必须区分“框架跑通”和“模型学得更好”。技术完整性不能替代 mAP、Recall、错误配对率、起止延时、并发鲁棒性和资源效率等性能证据。
- 默认决策口令：**模型优先，工程够用，快速证伪，性能与创新说话。**

## 项目规则

- 本库只保留 OpenTAD 代码库主体、在线/因果 TAD 路线 `CausalTAD`、对应配置、训练评测入口和轻量合约测试。
- 不放历史 `logs/`、`figures/`、实验压缩包、checkpoint、数据集、特征文件或远端同步缓存。
- 这里的“在线 TAD”按当前代码实际实现落到 `CausalTAD`：核心是 causal attention / causal Mamba，只使用过去或受限方向的时序上下文，服务于流式或在线部署形态。
- 验证/测试阶段严禁使用 GT、teacher cache、raw prediction shortcut 或隐藏缓存决策。`inference.load_from_raw_predictions` 必须保持 `False`，除非明确标记为诊断。
- 新增配置必须说明数据形态：预提取特征、视频帧、音频特征或多模态特征；不能把特征缓存、数据集或 checkpoint 放入仓库。
- 正式训练必须走 Slurm 或受控远端队列，不能在登录节点直接训练。

## 关键代码面

- `configs/causaltad/`: 在线/因果 TAD 配置入口。
- `configs/_base_/models/causaltad.py`: CausalTAD 模型基配置。
- `opentad/models/projections/causal_proj.py`: `CausalProj`，包含 causal Mamba 与 causal attention。
- `opentad/models/detectors/mamba.py`: `VideoMambaSuite` detector。
- `opentad/models/dense_heads/actionformer_head.py`: 默认检测 head。
- `opentad/datasets/`: 特征级数据集、滑窗和截断管线。

## 依赖要点

基础依赖见 `requirements.txt`。`CausalProj` 额外需要 Linux/CUDA 环境下的：

```bash
pip install causal-conv1d mamba-ssm flash-attn
```

这些包和 CUDA/PyTorch 版本绑定较强，优先在目标 GPU 服务器的 conda 环境里安装，不建议在 Windows 本地强行安装。

## 本地检查

```powershell
cd E:\DeskTop\TAD\OpenTAD_OnlineTADClean_20260702
python -m py_compile tools/train.py tools/test.py opentad/models/projections/causal_proj.py
python -m pytest tests/test_causaltad_config_contracts.py -q
```

## 远端边界

- Windows 本地不要用 WSL 做 SSH/SCP/rsync 或远端控制；用 PowerShell 和 `C:\Windows\System32\OpenSSH\ssh.exe` / `scp.exe`。
- N16R4 远端只在 `~/run/yuzibo` 或 `/data/run01/sczc063/yuzibo` 下放代码、环境、数据、日志和输出。
- 登录节点只做编辑、编译、轻量检查和提交 Slurm；训练必须用 `sbatch`。
- THUMOS14 在线 TAD 默认用预提取特征，推荐放在 `$BASE/thumos14/features/` 下，并保持 annotation/class map 在 `$BASE/thumos14/annotations/`。

## N16 学术下载出口（非模型配置）

- N16 官方 Google Drive 数据 staging 使用受控学术出口 `10.244.6.36:3128`，仅为下载官方
  数据、依赖或论文资料服务；它不是训练、评测或模型的一部分。
- 在启动下载进程的**受控会话**中同时注入 `http_proxy` 和 `https_proxy`。凭据必须来自受控
  环境变量或用户当次提供，示意命令为：

  ```bash
export http_proxy=u-MtfrT7:vH5orjDV@10.244.6.36:3128
export https_proxy=u-MtfrT7:vH5orjDV@10.244.6.36:3128
  ```

- 禁止把 `ACADEMIC_PROXY_CREDENTIAL`、完整代理 URL、shell history、下载日志或数据文件提交到
  Git/Wiki。运行记录只写入端点、变量名、官方源、文件哈希和非敏感的可达性/验证结果。
- 为避免登录目录配额污染，`gdown`/Python 下载任务必须将 `HOME`、`XDG_CACHE_HOME` 和
  `PIP_CACHE_DIR` 指向 `/data/run01/sczc063/yuzibo/<project-data-root>` 下的运行时目录；下载
  完成后必须验证官方归档、数据格式和 SHA-256，生成 ready sentinel 后才可提交 smoke。
