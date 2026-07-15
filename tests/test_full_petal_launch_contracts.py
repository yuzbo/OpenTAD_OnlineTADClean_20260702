from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from mmengine.config import Config
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "causaltad"
TICKET_SCRIPT = ROOT / "tools" / "build_full_petal_launch_ticket.py"
TICKET_SPEC = importlib.util.spec_from_file_location(
    "build_full_petal_launch_ticket", TICKET_SCRIPT
)
TICKET_MODULE = importlib.util.module_from_spec(TICKET_SPEC)
TICKET_SPEC.loader.exec_module(TICKET_MODULE)


def _load(name):
    return Config.fromfile(CONFIG_ROOT / name)


def _without_changed_axis(cfg):
    data = deepcopy(dict(cfg))
    data["model"] = dict(data["model"])
    data["model"].pop("trajectory_binding_mode")
    data.pop("work_dir")
    return data


def test_q2_bridge_configs_change_only_post_birth_loss_binding():
    rematch = _load("thumos_pes_q2_persist_rematch.py")
    fixed = _load("thumos_pes_q2_persist_fixed.py")

    assert _without_changed_axis(rematch) == _without_changed_axis(fixed)
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"


def test_q2_bridge_freezes_scientific_and_cost_contracts():
    cfg = _load("thumos_pes_q2_persist_fixed.py")

    assert cfg.route_stage == "q2_persistent_binding_one_factor"
    assert cfg.formal_training_ready is False
    assert cfg.visual_training_allowed is False
    assert cfg.raw_video_finetuning is False
    assert cfg.gpu_authorization == "BLOCKED_UNTIL_B0_AND_PROFILE"
    assert cfg.profile_contract.warmup_steps == 50
    assert cfg.profile_contract.measured_steps == 200
    assert cfg.profile_contract.step_unit == "optimizer_event"
    assert cfg.profile_contract.world_size == 1
    assert cfg.profile_contract.b1_total_gpu_hour_cap == 2
    assert cfg.profile_contract.b2_total_gpu_hour_cap == 10
    assert cfg.optimizer.audit.fail_on_frozen is True
    assert cfg.launch_contract.schema_version == "full-petal-launch-contract-v4"
    assert cfg.launch_contract.required_reviewer_id == (
        "019f5abd-5104-79b3-882e-354ca796f2c1"
    )
    assert set(cfg.launch_contract.attestation_trust_roots) == {
        "b0",
        "formal",
        "g0",
        "review",
        "profile",
    }
    assert dict(cfg.launch_contract.evidence_trust_model) == {
        "schema_version": "full-petal-evidence-trust-model-v1",
        "purpose": "scientific_reproducibility",
        "trusted_computing_base": [
            "launch_validator",
            "train_engine",
            "runtime_evidence_session",
            "in_process_attestation_key_material",
            "g0_preregistration_and_audit_runner",
        ],
        "guarantees": [
            "fail_closed_lifecycle_wiring",
            "provenance_binding",
            "post_publication_tamper_evidence",
        ],
        "out_of_scope": [
            "arbitrary_code_execution_inside_tcb",
            "in_process_private_key_compromise",
        ],
        "key_compromise_action": "BLOCK_ROTATE_AND_RERUN",
    }

    assert cfg.experiment_contract.changed_axis == "post_birth_target_to_slot_loss_binding"
    assert cfg.experiment_contract.shared_first_crossing_birth is True
    assert cfg.experiment_contract.canonical_lifecycle_shared is True
    assert cfg.experiment_contract.runtime_state_contains_gt is False
    assert cfg.experiment_contract.rematch_is_supervision_control_only is True

    assert cfg.model.type == "PersistentTrajectoryOnlineDetector"
    assert cfg.model.birth_assignment_mode == "first_crossing_shared"
    assert cfg.model.canonical_supervision_lifecycle is True
    assert cfg.model.head.query_mode == "persistent"
    assert cfg.model.head.start_mode == "scalar"
    assert cfg.model.head.endpoint_mode == "binary"
    assert cfg.model.head.num_slots == 4
    assert "backbone" not in cfg.model

    assert cfg.dataset.train.subset_name == "training"
    assert cfg.dataset.val.subset_name == "training"
    assert cfg.dataset.test.subset_name == "training"
    assert cfg.dataset.train.allow_list.endswith("thumos_fit_core_160.txt")
    assert cfg.dataset.val.allow_list.endswith("thumos_calibration_40.txt")
    assert cfg.dataset.test.allow_list.endswith("thumos_calibration_40.txt")
    assert cfg.dataset.train.split_role == "fit_core"
    assert cfg.dataset.val.split_role == "calibration"
    assert cfg.dataset.test.split_role == "calibration"
    assert cfg.dataset.train.split_seed == 20260713
    assert cfg.dataset.train.split_manifest.endswith("thumos_development_split.json")
    assert cfg.evaluation.subset == "training"
    assert cfg.evaluation.allowed_videos.endswith("thumos_calibration_40.txt")
    assert cfg.reporting_contract.allow_during_training is False
    assert cfg.reporting_contract.locked_population.endswith(
        "thumos_reporting_locked_211.txt"
    )
    assert cfg.solver.train.streaming is True
    assert cfg.solver.train.batch_size == 1
    assert cfg.inference.load_from_raw_predictions is False
    assert "nms" not in cfg.post_processing
    assert cfg.post_processing.streaming_safe_emission is True
    assert cfg.post_processing.emission_ledger_filename.endswith(".jsonl")
    assert cfg.post_processing.emission_ledger_commitment_filename.endswith(
        ".commitment.json"
    )


def test_q2_configs_keep_pointer_hazard_and_raw_video_out_of_c1_gate():
    for name in (
        "thumos_pes_q2_persist_rematch.py",
        "thumos_pes_q2_persist_fixed.py",
    ):
        cfg = _load(name)
        assert cfg.dataset.train.type == "StreamingFeatureDataset"
        assert cfg.model.head.start_mode != "pointer"
        assert cfg.model.head.endpoint_mode != "hazard"
        assert cfg.experiment_contract.input == "fixed_cached_causal_features"
        assert cfg.experiment_contract.offline_nms is False


def test_streaming_evaluator_does_not_inject_rank_into_model_metadata():
    source = (ROOT / "opentad" / "cores" / "test_engine.py").read_text(
        encoding="utf-8"
    )

    assert 'meta["eval_rank"]' not in source


def _entrypoint_source(name):
    return (ROOT / "tools" / name).read_text(encoding="utf-8")


def test_full_petal_gate_runs_before_ddp_and_cuda_in_every_entrypoint():
    for name in ("train.py", "test.py"):
        source = _entrypoint_source(name)
        main_offset = source.index("def main()")
        gate_offset = source.index("validate_full_petal_launch(", main_offset)
        receipt_offset = source.index("persist_launch_receipt(", gate_offset)
        ddp_offset = source.index("dist.init_process_group(", main_offset)
        cuda_offset = source.index("torch.cuda.set_device(", main_offset)

        assert gate_offset < receipt_offset < ddp_offset
        assert gate_offset < receipt_offset < cuda_offset


def test_profile_path_exits_before_checkpoint_or_evaluation():
    source = _entrypoint_source("train.py")
    completion_offset = source.index('train_stats["fixed_step_profile_complete"]')
    artifact_offset = source.index('"fixed_step_profile.json"', completion_offset)
    return_offset = source.index("            return", artifact_offset)
    checkpoint_offset = source.index("        # save checkpoint", completion_offset)
    validation_offset = source.index("        # val for one epoch", completion_offset)

    assert completion_offset < artifact_offset < return_offset
    assert return_offset < checkpoint_offset < validation_offset
    assert "build_fixed_step_profile_artifact(" in source
    assert "optimizer_event_recorder.persist(trace_path, commitment_path)" in source
    assert "optimizer_event_trace_path=trace_path" in source
    assert "optimizer_event_commitment_path=commitment_path" in source
    assert "profiler_measurements = fixed_step_profiler.measurements()" in source
    assert "profiler_measurements=profiler_measurements" in source
    assert "fixed_step_profiler.workload_measurements(" in source


def test_test_entrypoint_uses_resolved_amp_dtype():
    source = _entrypoint_source("test.py")

    assert "amp_dtype = resolve_amp_dtype(" in source
    assert "amp_dtype=amp_dtype," in source


def test_legacy_smoke_cannot_launch_full_petal():
    source = _entrypoint_source("smoke_pes_stage1.py")

    assert "is_full_petal_route(cfg)" in source
    assert "cannot use the legacy Stage-1 smoke launcher" in source


def test_ticket_builder_and_slurm_launcher_use_the_locked_gate():
    builder = _entrypoint_source("build_full_petal_launch_ticket.py")
    reader = _entrypoint_source("read_full_petal_launch_ticket.py")
    launcher = (
        ROOT / "tools" / "remote" / "submit_full_petal_q2_n16r4.sh"
    ).read_text(encoding="utf-8")

    assert "build_launch_ticket(" in builder
    assert 'cfg_overrides["work_dir"] = str(output.parent / "work")' in builder
    assert 'set(overrides) != {"work_dir"}' in reader
    assert '--launch-mode "${MODE}"' in launcher
    assert '--launch-ticket "${TICKET}"' in launcher
    assert '--cfg-options work_dir="${WORK_DIR}"' in launcher
    assert "STAMP=" not in launcher
    assert "RUNS_ROOT=" not in launcher
    assert "--nproc_per_node=1" in launcher
    assert "ALLOW_FORMAL" in launcher


def test_ticket_builder_requires_external_non_overwritable_output(tmp_path):
    output = tmp_path / "ticket.json"
    assert TICKET_MODULE._external_output(output) == output.resolve()

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(TICKET_MODULE.FullPetalLaunchError, match="overwrite"):
        TICKET_MODULE._external_output(output)
    with pytest.raises(TICKET_MODULE.FullPetalLaunchError, match="outside"):
        TICKET_MODULE._external_output(ROOT / "ticket.json")


def test_isolated_torch_runner_bypasses_only_package_level_registration():
    source = (
        ROOT / "tools" / "testing" / "run_isolated_torch_pytest.py"
    ).read_text(encoding="utf-8")

    assert "_install_b0_opentad" in source
    assert 'Registry("full_petal_b0_models")' in source
    assert 'sys.path.insert(0, str(repo_root))' in source
    assert "pytest.main(args.pytest_args)" in source
    assert "_ensure_cpu_nms_module" in source
    assert 'if exc.name != "nms_1d_cpu"' in source
    assert "_install_cpu_nms_test_double(torch)" in source
    assert "mmcv" not in source


def test_b0_runner_emits_hashed_junit_logs_and_requires_clean_repo():
    source = _entrypoint_source("run_full_petal_b0.py")

    assert "Full PETAL B0 requires a clean committed checkout" in source
    assert '"junit_sha256"' in source
    assert '"log_sha256"' in source
    assert "B0 evidence must be written outside the repository" in source
    assert "DEFAULT_MANIFEST" in source
    assert "validate_manifest(payload, repository_root=ROOT)" in source
    assert 'evidence_manifest_path = output_dir / "b0-manifest.json"' in source
    assert '"manifest_path": evidence_manifest_path.name' in source
    assert '"test_report_path": test_report_path.name' in source
    assert '"audit_report_path": audit_report_path.name' in source
    assert "sign_b0_evidence(" in source
    assert "_sign_payload(" not in source


def test_b0_manifest_hash_locks_evidence_trust_model_sources():
    manifest = json.loads(
        (ROOT / "tools" / "testing" / "full_petal_b0_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    locked_paths = {item["path"] for item in manifest["runner_sources"]}

    assert "FULL_PETAL_TRUST_MODEL.md" in locked_paths
    assert "FULL_PETAL_EXECUTION_GATES.md" in locked_paths
    assert "configs/causaltad/thumos_pes_q2_base.py" in locked_paths
    assert "configs/causaltad/thumos_pes_q2_crs_eps_base.py" in locked_paths
    assert "tools/read_full_petal_launch_ticket.py" in locked_paths
    assert "tools/run_crs_eps_gold_audit.py" in locked_paths
    assert "tools/preregister_crs_eps_gold_audit.py" in locked_paths
    assert "opentad/utils/crs_eps_sampling.py" in locked_paths
    assert "opentad/utils/crs_eps_gold_gate.py" in locked_paths
    assert "opentad/utils/crs_eps_gold_evidence.py" in locked_paths
    assert "opentad/datasets/crs_eps_feature.py" in locked_paths
    assert "opentad/models/detectors/persistent_trajectory_ontad.py" in locked_paths
    assert "opentad/evaluations/full_petal_metrics.py" in locked_paths
