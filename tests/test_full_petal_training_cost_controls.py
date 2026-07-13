import subprocess
import sys
from pathlib import Path

import pytest
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def _torch_or_skip():
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("torch import timed out")
    if probe.returncode != 0:
        pytest.skip("torch import is unavailable in this local environment")
    import torch

    return torch


def test_q2_training_registers_bf16_and_fixed_step_profile_contract():
    cfg = Config.fromfile(
        ROOT / "configs" / "causaltad" / "thumos_pes_q2_persist_fixed.py"
    )

    assert cfg.solver.amp is True
    assert cfg.solver.amp_dtype == "bf16"
    assert cfg.profile_contract.warmup_steps == 50
    assert cfg.profile_contract.measured_steps == 200


def test_precision_resolver_distinguishes_fp16_bf16_and_disabled():
    torch = _torch_or_skip()
    from opentad.cores.train_engine import resolve_amp_dtype

    assert resolve_amp_dtype(False, "bf16") is None
    assert resolve_amp_dtype(True, "fp16") is torch.float16
    assert resolve_amp_dtype(True, "float16") is torch.float16
    assert resolve_amp_dtype(True, "bf16") is torch.bfloat16
    assert resolve_amp_dtype(True, "bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="amp_dtype"):
        resolve_amp_dtype(True, "tf32")


def test_training_entrypoint_passes_registered_dtype_to_train_val_and_eval():
    source = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")

    assert "resolve_amp_dtype" in source
    assert "amp_dtype=amp_dtype" in source
    assert "GradScaler(enabled=amp_dtype is torch.float16)" in source


def test_engine_uses_amp_without_requiring_a_grad_scaler():
    source = (ROOT / "opentad" / "cores" / "train_engine.py").read_text(
        encoding="utf-8"
    )

    assert "use_amp = amp_dtype is not None" in source
    assert "dtype=amp_dtype" in source
    assert "use_amp = False if scaler is None else True" not in source
