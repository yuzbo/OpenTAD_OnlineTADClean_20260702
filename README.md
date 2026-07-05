# OpenTAD Online/Causal TAD Clean Route

Status: the raw-frame SigLIP/SigLIP2 and VideoMAE-adapter routes in this repo are validation candidate implementations under causal-audit validation. Streaming-safe emission-ledger evaluation is currently single-rank only, because DDP splits per-video online state across ranks. Do not claim paper-ready Online TAD results, DDP-auditable streaming evaluation, or visual-tower finetuning until emission-ledger evaluation, no-future tests, overfit checks, and real remote training results are attached.

这是从 `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_Back` 的受 Git 跟踪 `HEAD` 抽出的在线 TAD 干净代码库。当前代码中“在线 TAD”对应的是 `CausalTAD` 路线：用 causal attention 和 causal Mamba 限制时序信息流，面向流式或在线场景下的 temporal action detection。

本库只保留 OpenTAD 相关库、`configs/causaltad/`、CausalTAD 基配置、训练评测入口和轻量测试；历史 wiki、日志、图表、checkpoint、数据、特征、压缩包和同步缓存都不进入版本库。

## 当前目标

当前目标是建立一个可独立使用的在线/因果 TAD 起点：在特征级 THUMOS14、ActivityNet、HACS、EPIC、Ego4D 等配置上验证 CausalTAD 的因果时序建模能力，并为后续流式推理、低延迟检测、有限未来上下文、状态缓存和在线后处理实验提供干净基线。

路线边界：

- 模型核心是 `CausalProj`，依赖 causal Mamba 和 causal attention。
- 默认 detector 是 `VideoMambaSuite`，默认 head 是 `ActionFormerHead`。
- 默认输入是预提取特征，不包含视频原始数据或特征文件。
- 默认评测不读取 raw prediction shortcut，不使用验证/测试 GT 之外的任何隐藏决策缓存。

## 目录

- `opentad/`: OpenTAD 相关库和 CausalTAD 运行所需模块。
- `configs/causaltad/`: 在线/因果 TAD 配置。
- `configs/_base_/datasets/`: CausalTAD 配置依赖的数据集基配置。
- `configs/_base_/models/causaltad.py`: CausalTAD 模型基配置。
- `configs/causaltad/thumos_videomae_adapter_matr_ontad.py`: raw-frame VideoMAE adapter + MATR head 在线 TAD 实验配置骨架。
- `tools/train.py`, `tools/test.py`: 训练与评测入口。
- `tests/test_causaltad_config_contracts.py`: 最小配置和代码合约测试。
- `RTK.md`: 本仓库项目规则和远端边界。
- `end-to-end-ontad-research.md`: raw-frame 端到端在线 TAD/On-TAL 调研备忘。
- `online-action-models-2026-research.md`: 2026 在线动作检测、分割和流式理解模型调研备忘。

## 本地使用

```powershell
cd E:\DeskTop\TAD\OpenTAD_OnlineTADClean_20260702
pip install -r requirements.txt
python -m py_compile tools/train.py tools/test.py opentad/models/projections/causal_proj.py
python -m pytest tests/test_causaltad_config_contracts.py -q
```

注意：`mamba-ssm`、`causal-conv1d`、`flash-attn` 通常需要 Linux/CUDA 编译环境，本地 Windows 更适合做配置加载和语法检查。

## 推荐配置入口

- `configs/causaltad/thumos_i3d.py`: 官方 THUMOS14 I3D 特征配置。
- `configs/causaltad/thumos_internvideo2.py`: 官方 THUMOS14 InternVideo2 特征配置。
- `configs/causaltad/thumos_i3d_n16r4.py`: N16R4 THUMOS14 I3D 特征路径配置。
- `configs/causaltad/thumos_internvideo2_n16r4.py`: N16R4 THUMOS14 InternVideo2 特征路径配置。
- `configs/causaltad/thumos_videomae_adapter_matr_ontad.py`: 实验性 raw-frame On-TAD 模型入口，当前 raw-frame dataset/pipeline 已落地，但 backbone 仍是 contract-only stub；正式训练前必须接入真实 causal/streaming VideoMAE 并关闭 stub。

两个 `*_n16r4.py` 配置默认使用 `/data/run01/sczc063/yuzibo/thumos14`。如果远端数据根不同，直接改配置顶部的路径常量，不要在配置里引入 `import os`。

## N16R4 远端环境

本地登录用 PowerShell，不要从 WSL 发起远端控制：

```powershell
ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -o HostkeyAlgorithms=+ssh-rsa -i C:\Users\skywalker\.ssh\id_rsa -p 22 -l "sczc063@BSCC-N16R4" ssh.cn-zhongwei-1.paracloud.com
```

远端建议：

```bash
BASE=/data/run01/sczc063/yuzibo
cd "$BASE"
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
```

需要外网下载时，在登录节点设置代理：

```bash
export http_proxy='http://USER:PASSWORD@HOST:PORT'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

THUMOS14 特征路径约定：

```bash
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/thumos14/features/i3d_actionformer_stride4_thumos/
$BASE/thumos14/features/thumos14_6b/
```

正式训练必须用 Slurm，例如：

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J causaltad_thumos
#SBATCH -o logs/%x-%j.out

BASE=/data/run01/sczc063/yuzibo
cd "$BASE/OpenTAD_OnlineTADClean_20260702"
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
export OMP_NUM_THREADS=8

python tools/train.py configs/causaltad/thumos_i3d_n16r4.py --id 0
```

## 协议红线

验证/测试不能使用 GT、teacher cache、raw prediction shortcut 或旧运行缓存。所有结果解释必须区分完整训练、短 smoke、诊断 run 和正式评测；短 run 只能证明可启动、依赖完整、显存和配置健康，不能当作路线最终成败。
