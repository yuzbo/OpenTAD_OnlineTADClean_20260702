# RTK: Online/Causal TAD Clean Repo

## 项目规则

- 本库只保留 OpenTAD 代码库主体、在线/因果 TAD 路线 `CausalTAD`、对应配置、训练评测入口和轻量合约测试。
- 不放历史 `research-wiki/`、`logs/`、`figures/`、实验压缩包、checkpoint、数据集、特征文件或远端同步缓存。
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
