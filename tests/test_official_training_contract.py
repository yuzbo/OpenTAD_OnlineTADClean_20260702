"""Scientific contracts for the MATR-parent experiment schedule.

These tests deliberately distinguish the official-parent schedule from the
retired FIXED/REMATCH screening budget.  An arm may change only the two
registered scientific factors; it may not silently change training budget,
data modality, seed, or the official MATR architecture defaults.
"""

from __future__ import annotations

import json
import importlib.util
import ast
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiment_configs" / "eventmatr_bxo_official_thumos14.json"
COMMON_WRAPPER = ROOT / "scripts" / "train_eventmatr_common.sh"
NATIVE_WRAPPER = ROOT / "scripts" / "train_native_matr.sh"
ENV_ACTIVATOR = ROOT / "scripts" / "activate_matr_env.sh"
LOCKED_TEST_WRAPPER = ROOT / "scripts" / "eval_locked_test_once.sh"

ARM_FACTORS = {
    "b0o0": ("matr_delayed", "fresh_rematch"),
    "b1o0": ("instant_transition", "fresh_rematch"),
    "b0o1": ("matr_delayed", "sticky_owner"),
    "b1o1": ("instant_transition", "sticky_owner"),
}


def _parse_args(monkeypatch: pytest.MonkeyPatch, *argv: str):
    from util.config import make_parser

    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    return make_parser()


def _load_experiment_config() -> dict:
    assert CONFIG_PATH.is_file(), f"missing frozen experiment config: {CONFIG_PATH}"
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_official_parser_defaults_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse_args(monkeypatch)

    # Native MATR training contract, not the retired 12-epoch screen.
    assert args.epochs == 100
    assert args.batch == 64
    assert args.random_seed == 52

    # Native THUMOS14 feature/model/optimizer contract.
    assert args.feat_dim == 4096
    assert args.num_frame == 64
    assert args.num_queries == 10
    assert args.max_memory_len == 7
    assert args.memory_sampler == "gap2"
    assert args.min_lr == pytest.approx(1e-8)
    assert args.max_lr == pytest.approx(1e-5)
    assert args.weight_decay == pytest.approx(1e-4)
    assert args.lr_Tup == 3
    assert args.lr_Tcycle == 10
    assert args.cls_threshold == pytest.approx(0.1)
    assert args.nms_threshold == pytest.approx(0.3)
    assert args.event_resource_limit == 0
    assert args.model_variant == "native_matr"
    assert args.event_birth_logit_threshold is None
    assert args.event_end_logit_threshold is None

    # No-flag invocation is the official parent path.
    assert args.birth_mode == "matr_delayed"
    assert args.ownership_mode == "fresh_rematch"


@pytest.mark.parametrize(
    ("birth_mode", "ownership_mode"),
    ARM_FACTORS.values(),
)
def test_only_registered_factor_values_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
    birth_mode: str,
    ownership_mode: str,
) -> None:
    args = _parse_args(
        monkeypatch,
        "--birth_mode",
        birth_mode,
        "--ownership_mode",
        ownership_mode,
    )
    assert args.birth_mode == birth_mode
    assert args.ownership_mode == ownership_mode


@pytest.mark.parametrize(
    "argv",
    [
        ("--birth_mode", "slot_bank"),
        ("--ownership_mode", "nearest_slot"),
    ],
)
def test_unregistered_slot_style_modes_are_rejected(
    monkeypatch: pytest.MonkeyPatch, argv: tuple[str, str]
) -> None:
    with pytest.raises(SystemExit):
        _parse_args(monkeypatch, *argv)


def test_frozen_json_uses_official_matr_settings() -> None:
    config = _load_experiment_config()
    common = config["common"]
    data = config["data_contract"]

    expected = {
        "epochs": 100,
        "batch": 64,
        "random_seed": 52,
        "rgb": True,
        "flow": True,
        "feat_dim": 4096,
        "num_frame": 64,
        "num_queries": 10,
        "max_memory_len": 7,
        "memory_sampler": "gap2",
        "optimizer": "Adam",
        "min_lr": 1e-8,
        "max_lr": 1e-5,
        "weight_decay": 1e-4,
        "lr_Tup": 3,
        "lr_Tcycle": 10,
        "use_focal": True,
        "cls_threshold": 0.1,
        "nms_threshold": 0.3,
    }
    for key, value in expected.items():
        assert key in common, f"official setting {key!r} is not frozen"
        if isinstance(value, float):
            assert common[key] == pytest.approx(value)
        else:
            assert common[key] == value

    assert data["feature_dim"] == 4096
    assert data["segment_frames"] == 64
    lanes = config["lanes"]
    assert lanes["native_matr"]["model_variant"] == "native_matr"
    assert "event_arm" not in lanes["native_matr"]
    for arm, (birth, owner) in ARM_FACTORS.items():
        assert lanes[arm]["model_variant"] == "eventmatr"
        assert lanes[arm]["event_arm"] == arm
        assert lanes[arm]["birth_mode"] == birth
        assert lanes[arm]["ownership_mode"] == owner

    training = config["training_protocol"]
    expected_training = {
        "source": "official_complete_validation_train",
        "epochs": 100,
        "checkpoint_policy": "terminal_epoch",
        "checkpoint_epoch": 100,
        "test_during_training": False,
        "locked_test_once_per_lane": True,
    }
    for key, value in expected_training.items():
        assert training[key] == value


@pytest.mark.parametrize("arm", ARM_FACTORS)
def test_arm_wrappers_change_only_birth_and_ownership(arm: str) -> None:
    wrapper = ROOT / "scripts" / f"train_{arm}.sh"
    assert wrapper.is_file(), f"missing arm wrapper: {wrapper}"
    text = wrapper.read_text(encoding="utf-8")
    birth_mode, ownership_mode = ARM_FACTORS[arm]

    invocation = re.search(
        rf"train_eventmatr_common\.sh\"?\s+{arm}\s+{birth_mode}\s+{ownership_mode}(?:\s|\")",
        text,
    )
    explicit_options = (
        re.search(rf"--birth_mode(?:=|\s+){re.escape(birth_mode)}(?:\s|\\|$)", text)
        and re.search(
            rf"--ownership_mode(?:=|\s+){re.escape(ownership_mode)}(?:\s|\\|$)",
            text,
        )
    )
    assert invocation or explicit_options
    assert "--model_variant eventmatr" in COMMON_WRAPPER.read_text(encoding="utf-8")

    forbidden_overrides = (
        "--epochs",
        "--batch",
        "--random_seed",
        "--feat_dim",
        "--num_frame",
        "--num_queries",
        "--max_memory_len",
        "--memory_sampler",
        "--min_lr",
        "--max_lr",
        "--weight_decay",
    )
    for option in forbidden_overrides:
        assert option not in text, f"{arm} illegally overrides common setting {option}"


def test_common_wrapper_explicitly_uses_official_budget_and_seed() -> None:
    assert COMMON_WRAPPER.is_file(), f"missing common launcher: {COMMON_WRAPPER}"
    text = COMMON_WRAPPER.read_text(encoding="utf-8")
    assert re.search(r"--epochs(?:=|\s+)100(?:\s|\\|$)", text)
    assert re.search(r"--batch(?:=|\s+)64(?:\s|\\|$)", text)
    assert re.search(r"--random_seed(?:=|\s+)52(?:\s|\\|$)", text)
    assert re.search(r"--model_variant(?:=|\s+)eventmatr(?:\s|\\|$)", text)
    assert re.search(r"--study_protocol(?:=|\s+)matched_study(?:\s|\\|$)", text)


def test_native_matr_is_an_independent_non_event_lane() -> None:
    assert NATIVE_WRAPPER.is_file(), f"missing native MATR lane: {NATIVE_WRAPPER}"
    text = NATIVE_WRAPPER.read_text(encoding="utf-8")
    assert re.search(r"--model_variant(?:=|\s+)native_matr(?:\s|\\|$)", text)
    assert "--event_arm" not in text
    assert "--birth_mode" not in text
    assert "--ownership_mode" not in text
    assert re.search(r"--epochs(?:=|\s+)100(?:\s|\\|$)", text)
    assert re.search(r"--random_seed(?:=|\s+)52(?:\s|\\|$)", text)
    assert re.search(r"--study_protocol(?:=|\s+)matched_study(?:\s|\\|$)", text)


def test_retired_fixed_rematch_budget_is_absent_from_new_experiment_files() -> None:
    config = _load_experiment_config()
    common = config["common"]
    assert common["epochs"] == 100
    assert common["random_seed"] == 52

    paths = [COMMON_WRAPPER, NATIVE_WRAPPER]
    paths.extend(ROOT / "scripts" / f"train_{arm}.sh" for arm in ARM_FACTORS)

    for path in paths:
        assert path.is_file(), f"missing experiment file: {path}"
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?:--epochs(?:=|\s+)|\"epochs\"\s*:\s*)12(?:\D|$)", text)
        assert not re.search(
            r"(?:--random_seed(?:=|\s+)|\"random_seed\"\s*:\s*)705(?:\D|$)",
            text,
        )


def test_legacy_feature_screen_capacity_and_backbone_are_forbidden() -> None:
    paths = [CONFIG_PATH, COMMON_WRAPPER, NATIVE_WRAPPER]
    paths.extend(ROOT / "scripts" / f"train_{arm}.sh" for arm in ARM_FACTORS)
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()

    assert "siglip" not in corpus
    assert "6-entry" not in corpus
    assert "six-entry" not in corpus
    assert not re.search(r"--event_resource_limit(?:=|\s+)6(?:\D|$)", corpus)
    manifest = _load_experiment_config()
    assert manifest["common"].get("event_resource_limit", 0) == 0


@pytest.mark.parametrize(
    ("key", "retired_value"),
    [("epochs", 12), ("random_seed", 705)],
)
def test_protocol_verifier_fails_closed_on_retired_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    retired_value: int,
) -> None:
    verifier_path = ROOT / "scripts" / "verify_official_protocol.py"
    spec = importlib.util.spec_from_file_location("eventmatr_protocol_verifier", verifier_path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    manifest = _load_experiment_config()
    manifest["common"][key] = retired_value
    drifted_path = tmp_path / "drifted_protocol.json"
    drifted_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(verifier, "MANIFEST", drifted_path)

    with pytest.raises(SystemExit, match="official protocol drift"):
        verifier.main()


def test_formal_event_arms_do_not_pass_fixed_birth_or_end_thresholds() -> None:
    common = COMMON_WRAPPER.read_text(encoding="utf-8").lower()
    wrappers = "\n".join(
        (ROOT / "scripts" / f"train_{arm}.sh").read_text(encoding="utf-8").lower()
        for arm in ARM_FACTORS
    )
    event_protocol = common + "\n" + wrappers

    # MATR's original segment-memory --flag_threshold 0.5 may remain for exact
    # native parity, but it is not an EventMATR lifecycle decision.  EventMATR
    # EventMATR birth/end decisions must be learned logits/calibration rather
    # than a newly frozen probability-0.5 rule in the formal B1 arms.
    assert "--event_birth_probability_threshold" not in event_protocol
    assert "--event_end_probability_threshold" not in event_protocol
    assert "--event_birth_logit_threshold" not in event_protocol
    assert "--event_end_logit_threshold" not in event_protocol


def _matched_training_source() -> str:
    path = ROOT / "on_tal_task.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "train_matched_study",
            "_train_matched_study",
        }:
            return ast.get_source_segment(source, node) or ""
    matched_blocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.get_source_segment(source, node.test) or ""
            if "matched_study" in test:
                matched_blocks.extend(
                    ast.get_source_segment(source, child) or "" for child in node.body
                )
    if matched_blocks:
        return "\n".join(matched_blocks)
    raise AssertionError("on_tal_task.py has no isolated matched-study training path")


def test_matched_study_is_train_only_and_saves_terminal_epoch_100_once() -> None:
    source = _matched_training_source()
    # The matched branch may explicitly poison the test loader with ``None``;
    # it must never construct or consume that loader while fitting.
    assert "test_loader = None" in source
    assert "DataLoader" not in source
    assert "test_one_epoch" not in source
    assert "subset='test'" not in source and 'subset="test"' not in source
    assert "torch.save" in source
    assert "terminal_epoch" in source
    assert "best_epoch" not in source

    common = COMMON_WRAPPER.read_text(encoding="utf-8").lower()
    assert "calib" not in common
    assert "matr_calib_feature" not in common
    # The upstream CLI requires a test-feature argument.  The matched wrapper
    # may pass a deliberately invalid sentinel, while the Python branch above
    # proves that no test dataset/loader is constructed from it.
    if "--video_feature_all_test" in common:
        assert "locked_test_sentinel" in common


def test_matched_training_dag_has_no_calibration_input_and_includes_all_lanes() -> None:
    paths = [
        CONFIG_PATH,
        COMMON_WRAPPER,
        ROOT / "scripts" / "submit_eventmatr_bxo_official_slurm.sh",
        ROOT / "scripts" / "slurm_eventmatr_train.sh",
        ROOT / "scripts" / "slurm_eventmatr_finalize.sh",
        ROOT / "scripts" / "verify_eventmatr_completion.py",
    ]
    corpus = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert not re.search(r"--[^\s\\]*calib[^\s\\]*", corpus)
    assert '"calibration_split": true' not in corpus
    assert '"calibration_file"' not in corpus
    assert '"calibration_path"' not in corpus

    # Mentioning a calibration variable is permitted only as a fail-closed
    # guard that rejects such input from the formal fitting DAG.
    if "matr_calib_feature" in corpus:
        launcher = (
            ROOT / "scripts" / "submit_eventmatr_bxo_official_slurm.sh"
        ).read_text(encoding="utf-8").lower()
        assert "matr_calib_feature" in launcher
        assert "forbids test/report/calibration" in launcher
        assert re.search(r"matr_calib_feature[\s\S]*?exit\s+2", launcher)

    launcher = (ROOT / "scripts" / "submit_eventmatr_bxo_official_slurm.sh").read_text(
        encoding="utf-8"
    )
    for lane in ("native_matr", *ARM_FACTORS):
        assert lane in launcher


def test_completion_contract_requires_one_terminal_checkpoint_per_lane() -> None:
    verifier = ROOT / "scripts" / "verify_eventmatr_completion.py"
    assert verifier.is_file()
    text = verifier.read_text(encoding="utf-8").lower()
    assert "eventmatr_bxo_completion.json" in text
    for lane in ("native_matr", *ARM_FACTORS):
        assert lane in text
    assert "terminal_epoch100.pth" in text
    assert "best_epoch" not in text
    assert "calib" not in text
    assert "matched_study" in text
    assert "locked_test" in text


def test_locked_test_is_reserved_and_consumed_once_per_lane() -> None:
    assert LOCKED_TEST_WRAPPER.is_file(), f"missing one-shot test wrapper: {LOCKED_TEST_WRAPPER}"
    text = LOCKED_TEST_WRAPPER.read_text(encoding="utf-8").lower()
    assert "locked_test_consumed" in text
    assert "native_matr" in text
    for arm in ARM_FACTORS:
        assert arm in text
    assert "test -e" in text or "-f" in text or "-d" in text
    assert "mkdir" in text or "touch" in text

    # Fitting/finalization must not consume the one-shot test reservation.
    training_launcher = (
        ROOT / "scripts" / "submit_eventmatr_bxo_official_slurm.sh"
    ).read_text(encoding="utf-8").lower()
    assert "slurm_locked_test_once.sh" not in training_launcher
    assert "eval_locked_test_once.sh" not in training_launcher


def test_environment_activation_has_one_explicit_path_and_no_hidden_fallback() -> None:
    assert ENV_ACTIVATOR.is_file(), f"missing environment activator: {ENV_ACTIVATOR}"
    text = ENV_ACTIVATOR.read_text(encoding="utf-8")
    assert "MATR_ENV_ACTIVATE" in text
    assert "MATR_CONDA_SH" in text
    assert "MATR_CONDA_ENV" in text
    assert re.search(r"if\s+\[\[.*MATR_ENV_ACTIVATE", text, re.DOTALL)
    assert "conda activate" in text
    assert ">&2" in text
    assert re.search(r"(?:return|exit)\s+2", text)

    for name in (
        "slurm_eventmatr_smoke.sh",
        "slurm_eventmatr_train.sh",
        "slurm_eventmatr_finalize.sh",
        "slurm_locked_test_once.sh",
    ):
        path = ROOT / "scripts" / name
        assert path.is_file(), f"missing formal Slurm stage: {path}"
        stage = path.read_text(encoding="utf-8")
        assert "activate_matr_env.sh" in stage
        assert "conda activate" not in stage
