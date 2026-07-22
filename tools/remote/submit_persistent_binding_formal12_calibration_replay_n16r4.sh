#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_DIR=${BASE_DIR:-$REPO_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the replay code commit}
SOURCE_RUN=${SOURCE_RUN:?set SOURCE_RUN to the completed formal12 run}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the source training commit}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/persistent_binding}
SBATCH_EXCLUDE=${SBATCH_EXCLUDE:-g0063}
MAX_SUBMIT_JOBS=${MAX_SUBMIT_JOBS:-16}
SEED=705
FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_formal12_boundary_calibration_batched_fixed.py
REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_formal12_boundary_calibration_batched_rematch.py

COMMIT_SHA=$(git -C "$BASE_DIR" rev-parse HEAD)
if [[ "$COMMIT_SHA" != "$EXPECTED_COMMIT" ]]; then
    echo "Calibration replay checkout does not match EXPECTED_COMMIT" >&2
    exit 2
fi
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "SOURCE_COMMIT must be a full lowercase Git SHA" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Calibration replay checkout must be clean" >&2
    exit 2
fi
SOURCE_RUN=$(realpath "$SOURCE_RUN")
test -d "$SOURCE_RUN"
for config in "$FIXED_CONFIG" "$REMATCH_CONFIG"; do
    test -f "$BASE_DIR/$config"
done

python3 - "$SOURCE_RUN" "$SOURCE_COMMIT" "$COMMIT_SHA" "$BASE_DIR" \
    "$FIXED_CONFIG" "$REMATCH_CONFIG" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

source, source_commit, replay_commit, repo, fixed_config, rematch_config = sys.argv[1:]
source = Path(source).resolve()
repo = Path(repo).resolve()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

contract = load(source / "formal12_contract.json")
if contract.get("schema_version") != "persistent_binding_feature_multi_epoch_deployment.v1":
    raise SystemExit("source formal12 contract schema mismatch")
if contract.get("code_commit") != source_commit:
    raise SystemExit("source formal12 commit mismatch")
if contract.get("seed") != 705 or contract.get("epochs") != 12:
    raise SystemExit("source formal12 training identity mismatch")
if contract.get("checkpoint_epochs") != [3, 6, 9, 12]:
    raise SystemExit("source formal12 checkpoint schedule mismatch")
if contract.get("fixed_thresholds") != {"birth": 0.5, "alive": 0.5, "end": 0.5}:
    raise SystemExit("source formal12 thresholds changed")
for field in ("reporting_accessed", "threshold_search", "raw_rgb_authorized"):
    if contract.get(field) is not False:
        raise SystemExit(f"source formal12 changed {field}")

changed = subprocess.run(
    ["git", "-C", str(repo), "diff", "--name-only", f"{source_commit}..{replay_commit}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
allowed_exact = {
    "opentad/utils/online_protocol.py",
    "tools/verify_persistent_binding_calibration_replay.py",
    "tools/remote/submit_persistent_binding_formal12_calibration_replay_n16r4.sh",
}
unexpected = [
    path
    for path in changed
    if path not in allowed_exact
    and not path.startswith("tests/")
    and not path.startswith("research-wiki/")
]
if unexpected:
    raise SystemExit(f"calibration replay changed model/evaluation code: {unexpected}")
if "opentad/utils/online_protocol.py" not in changed:
    raise SystemExit("calibration replay is missing the ledger-order repair")

configs = {"fixed": fixed_config, "rematch": rematch_config}
for arm, config in configs.items():
    recovery = source / arm / "recovery"
    manifest = load(recovery / "recovery_manifest.json")
    audit = load(recovery / "training_audit.json")
    if manifest.get("schema_version") != "persistent_binding_training_recovery.v1":
        raise SystemExit(f"{arm} source recovery schema mismatch")
    if manifest.get("arm") != arm or manifest.get("code_commit") != source_commit:
        raise SystemExit(f"{arm} source recovery identity mismatch")
    if manifest.get("successful_updates") != 24120:
        raise SystemExit(f"{arm} source recovery update count mismatch")
    if manifest.get("reporting_accessed") is not False:
        raise SystemExit(f"{arm} source recovery accessed reporting")
    if sha(recovery / "training_audit.json") != manifest.get("training_audit_sha256"):
        raise SystemExit(f"{arm} source training audit hash mismatch")
    if sha(repo / config) != manifest.get("config_sha256"):
        raise SystemExit(f"{arm} current config differs from trained config")
    totals = audit.get("totals", {})
    for field in ("expected_updates", "successful_updates", "scheduler_steps"):
        if int(totals.get(field, -1)) != 24120:
            raise SystemExit(f"{arm} source audit {field} mismatch")
    for field in (
        "skipped_updates",
        "gt_supervision_exhaustions",
        "gt_birth_runtime_entry_free_collisions",
    ):
        if int(totals.get(field, -1)) != 0:
            raise SystemExit(f"{arm} source audit {field} is nonzero")
    rows = manifest.get("checkpoints", [])
    if [int(row.get("epoch", -1)) for row in rows] != [3, 6, 9, 12]:
        raise SystemExit(f"{arm} source recovery checkpoint epochs mismatch")
    for row in rows:
        path = recovery / row["path"]
        if sha(path) != row.get("sha256"):
            raise SystemExit(f"{arm} source checkpoint hash mismatch: {path}")
    artifacts = source / arm / "artifacts"
    for human in (3, 6, 9, 12):
        for name in (
            f"calibration_candidate_epoch_{human}.json",
            f"calibration_epoch_{human}_emissions.json",
        ):
            if not (artifacts / name).is_file():
                raise SystemExit(f"{arm} source calibration artifact missing: {name}")
    if not (artifacts / "calibration_receipt.json").is_file():
        raise SystemExit(f"{arm} source calibration receipt missing")
    if not (artifacts / "resource_report.json").is_file():
        raise SystemExit(f"{arm} source resource report missing")
PY

for name in pb_h2f12_replay_fixed pb_h2f12_replay_rematch; do
    if squeue -u "$USER" -h -n "$name" | grep -q .; then
        echo "Refusing duplicate active calibration replay job: $name" >&2
        exit 3
    fi
done
active_jobs=$(squeue -u "$USER" -h | wc -l)
if (( active_jobs + 3 > MAX_SUBMIT_JOBS )); then
    echo "Calibration replay needs three submit slots; active=$active_jobs cap=$MAX_SUBMIT_JOBS" >&2
    exit 3
fi

timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RUNS_ROOT/formal12_calibration_replay_seed705_$timestamp"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/fixed" "$RUN_DIR/rematch"
cp "$SOURCE_RUN/split_census.json" "$RUN_DIR/split_census.json"

python3 - "$SOURCE_RUN/formal12_contract.json" \
    "$RUN_DIR/formal12_contract.json" "$SOURCE_RUN" "$SOURCE_COMMIT" \
    "$COMMIT_SHA" "$RUN_DIR/split_census.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source_contract, output, source_run, source_commit, replay_commit, census = sys.argv[1:]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = json.loads(Path(source_contract).read_text(encoding="utf-8"))
payload.update(
    {
        "code_commit": replay_commit,
        "split_census_sha256": sha(census),
        "protocol_revision": "positive_duration_hard_reserve_ledger_tie_order_replay.v3",
        "source_training_run": str(Path(source_run).resolve()),
        "source_training_commit": source_commit,
        "source_formal_contract_sha256": sha(source_contract),
        "calibration_replay_only": True,
        "training_reused": True,
        "replay_reason": "tied_emit_frame_sequence_order_contract",
        "replay_event_payload_equality_required": True,
        "replay_silent_ledger_correction": False,
        "reporting_accessed": False,
        "threshold_search": False,
        "raw_rgb_authorized": False,
    }
)
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

submit_arm() {
    local arm=$1
    local config=$2
    local job_name=$3
    local arm_root="$RUN_DIR/$arm"
    local job_script="$arm_root/calibration_replay.sbatch.sh"
    cat > "$job_script" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=$job_name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --exclude=$SBATCH_EXCLUDE
#SBATCH --output=$arm_root/slurm.%j.out
#SBATCH --error=$arm_root/slurm.%j.err

set -euo pipefail
BASE_DIR='$BASE_DIR'
RUN_DIR='$RUN_DIR'
SOURCE_RUN='$SOURCE_RUN'
SOURCE_COMMIT='$SOURCE_COMMIT'
ARM='$arm'
CONFIG='$config'
EXPECTED_COMMIT='$COMMIT_SHA'
SEED='$SEED'
cd "\$BASE_DIR"
test "\$(git rev-parse HEAD)" = "\$EXPECTED_COMMIT"
test -z "\$(git status --porcelain)"
source tools/env/activate_n16r4_causaltad.sh
GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
STARTED=\$(date +%s)
LOCAL_ROOT=/tmp/\${USER}_ontad_formal12_replay_\${SLURM_JOB_ID}_\${ARM}
test ! -e "\$LOCAL_ROOT"
mkdir -p "\$LOCAL_ROOT"
SOURCE_RECOVERY="\$SOURCE_RUN/\$ARM/recovery"
SOURCE_ARTIFACTS="\$SOURCE_RUN/\$ARM/artifacts"
candidate_args=()
for spec in 3:2 6:5 9:8 12:11; do
    human=\${spec%%:*}
    zero=\${spec##*:}
    checkpoint="\$SOURCE_RECOVERY/checkpoint/epoch_\${zero}.pth"
    source_ledger="\$SOURCE_ARTIFACTS/calibration_epoch_\${human}_emissions.json"
    source_candidate="\$SOURCE_ARTIFACTS/calibration_candidate_epoch_\${human}.json"
    test -f "\$checkpoint"
    test -f "\$source_ledger"
    test -f "\$source_candidate"
    eval_root="\$LOCAL_ROOT/eval_epoch_\${human}"
    port_slot=\$(((SLURM_JOB_ID + human) % 5000))
    master_port=\$((20000 + port_slot * 8))
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
        --rdzv_endpoint "127.0.0.1:\${master_port}" \
        tools/test.py "\$CONFIG" \
        --evaluation-role calibration \
        --checkpoint "\$checkpoint" \
        --seed "\$SEED" \
        --id 0 \
        --cfg-options work_dir="\$eval_root"
    ledger="\$eval_root/gpu1_id0/calibration/persistent_binding_emissions.json"
    verification="\$LOCAL_ROOT/calibration_replay_verification_epoch_\${human}.json"
    candidate="\$LOCAL_ROOT/calibration_candidate_epoch_\${human}.json"
    test -f "\$ledger"
    python tools/verify_persistent_binding_calibration_replay.py \
        --source-ledger "\$source_ledger" \
        --replay-ledger "\$ledger" \
        --output "\$verification"
    python tools/build_persistent_binding_calibration_candidate.py \
        --repo "\$BASE_DIR" \
        --config "\$CONFIG" \
        --checkpoint "\$checkpoint" \
        --ledger "\$ledger" \
        --census "\$RUN_DIR/split_census.json" \
        --seed "\$SEED" \
        --arm "\$ARM" \
        --epoch "\$zero" \
        --output "\$candidate"
    python3 - "\$source_candidate" "\$candidate" "\$verification" <<'PY'
import json
import math
from pathlib import Path
import sys

source, replay, verification = (
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]
)
if verification.get("passed") is not True or verification.get("reorder_only") is not True:
    raise SystemExit("calibration replay verification did not pass")
if source.get("checkpoint_sha256") != replay.get("checkpoint_sha256"):
    raise SystemExit("calibration replay changed checkpoint identity")
if not math.isclose(
    float(source.get("metric_value")),
    float(replay.get("metric_value")),
    rel_tol=0.0,
    abs_tol=1e-15,
):
    raise SystemExit("calibration replay changed the selection metric")
PY
    candidate_args+=(--candidate "\$candidate")
done
RECEIPT="\$LOCAL_ROOT/calibration_receipt.json"
python tools/select_persistent_binding_checkpoint.py \
    "\${candidate_args[@]}" \
    --output "\$RECEIPT"
python3 - "\$SOURCE_ARTIFACTS/calibration_receipt.json" "\$RECEIPT" <<'PY'
import json
from pathlib import Path
import sys

source, replay = (
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]
)
for field in ("selected_epoch", "selected_checkpoint_sha256", "selected_metric_value"):
    if source.get(field) != replay.get(field):
        raise SystemExit(f"calibration replay changed receipt field: {field}")
PY
ENDED=\$(date +%s)
python tools/build_persistent_binding_resource_report.py \
    --repo "\$BASE_DIR" \
    --scope "\$ARM" \
    --started-unix "\$STARTED" \
    --ended-unix "\$ENDED" \
    --gpu-count 1 \
    --gpu-name "\$GPU_NAME" \
    --slurm-job-id "\${SLURM_JOB_ID}" \
    --output "\$LOCAL_ROOT/replay_resource_report.json"
python3 - "\$LOCAL_ROOT" "\$RUN_DIR/\$ARM" "\$CONFIG" "\$ARM" \
    "\$EXPECTED_COMMIT" "\$SOURCE_COMMIT" "\$SOURCE_RUN" \
    "\$SOURCE_RECOVERY" "\$SOURCE_ARTIFACTS" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

(
    local,
    output,
    config,
    arm,
    replay_commit,
    source_commit,
    source_run,
    source_recovery,
    source_artifacts,
) = sys.argv[1:]
local = Path(local).resolve()
output = Path(output).resolve()
config = Path(config).resolve()
source_run = Path(source_run).resolve()
source_recovery = Path(source_recovery).resolve()
source_artifacts = Path(source_artifacts).resolve()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

audit = load(source_recovery / "training_audit.json")
totals = audit["totals"]
recovery_manifest = load(source_recovery / "recovery_manifest.json")
if recovery_manifest.get("code_commit") != source_commit:
    raise SystemExit("source recovery commit changed before replay staging")

receipt = load(local / "calibration_receipt.json")
selected_zero = int(receipt["selected_epoch"])
if selected_zero not in (2, 5, 8, 11):
    raise SystemExit("replay selected checkpoint is outside epochs 3/6/9/12")

source_resource = load(source_artifacts / "resource_report.json")
replay_resource = load(local / "replay_resource_report.json")
if source_resource.get("code_commit") != source_commit:
    raise SystemExit("source resource report commit mismatch")
if replay_resource.get("code_commit") != replay_commit:
    raise SystemExit("replay resource report commit mismatch")
combined_resource = dict(replay_resource)
combined_resource.update(
    {
        "started_unix": int(source_resource["started_unix"]),
        "elapsed_seconds": int(source_resource["elapsed_seconds"])
        + int(replay_resource["elapsed_seconds"]),
        "allocated_gpu_hours": float(source_resource["allocated_gpu_hours"])
        + float(replay_resource["allocated_gpu_hours"]),
        "source_training_and_first_calibration_gpu_hours": float(
            source_resource["allocated_gpu_hours"]
        ),
        "calibration_replay_gpu_hours": float(
            replay_resource["allocated_gpu_hours"]
        ),
        "source_slurm_job_id": str(source_resource["slurm_job_id"]),
        "calibration_replay_slurm_job_id": str(replay_resource["slurm_job_id"]),
        "resource_accounting": "cumulative_source_plus_replay",
        "source_training_commit": source_commit,
        "code_commit": replay_commit,
    }
)

artifacts = output / "artifacts"
artifacts.mkdir(parents=True, exist_ok=False)
shutil.copy2(config, artifacts / "config.py")
shutil.copy2(source_recovery / "training_audit.json", artifacts / "training_audit.json")
shutil.copy2(
    source_recovery / "recovery_manifest.json",
    artifacts / "training_recovery_manifest.json",
)
(artifacts / "resource_report.json").write_text(
    json.dumps(combined_resource, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
shutil.copy2(local / "calibration_receipt.json", artifacts / "calibration_receipt.json")

curve = []
for human, zero in ((3, 2), (6, 5), (9, 8), (12, 11)):
    candidate_source = local / f"calibration_candidate_epoch_{human}.json"
    candidate_target = artifacts / f"calibration_candidate_epoch_{human}.json"
    shutil.copy2(candidate_source, candidate_target)
    candidate = load(candidate_source)
    ledger_source = (
        local
        / f"eval_epoch_{human}/gpu1_id0/calibration/persistent_binding_emissions.json"
    )
    ledger_target = artifacts / f"calibration_epoch_{human}_emissions.json"
    shutil.copy2(ledger_source, ledger_target)
    verification_source = local / f"calibration_replay_verification_epoch_{human}.json"
    verification_target = artifacts / f"calibration_replay_verification_epoch_{human}.json"
    shutil.copy2(verification_source, verification_target)
    verification = load(verification_target)
    if verification.get("passed") is not True:
        raise SystemExit(f"epoch {human} replay verification failed before staging")
    curve.append(
        {
            "epoch": human,
            "checkpoint_epoch": zero,
            "metric_name": candidate["metric_name"],
            "metric_value": candidate["metric_value"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "emission_ledger_sha256": sha(ledger_target),
            "replay_verification_sha256": sha(verification_target),
        }
    )

checkpoint_dir = artifacts / "checkpoint"
checkpoint_dir.mkdir()
final_source = source_recovery / "checkpoint/epoch_11.pth"
final_target = checkpoint_dir / "epoch_12_resume.pth"
shutil.copy2(final_source, final_target)
selected_source = source_recovery / "checkpoint" / f"epoch_{selected_zero}.pth"
if sha(selected_source) != receipt["selected_checkpoint_sha256"]:
    raise SystemExit("replay selected checkpoint hash changed before promotion")
if selected_zero == 11:
    selected_target = final_target
else:
    selected_target = checkpoint_dir / f"selected_epoch_{selected_zero + 1}.pth"
    shutil.copy2(selected_source, selected_target)

promotion = {
    "schema_version": "persistent_binding_calibration_promotion.v1",
    "arm": arm,
    "seed": 705,
    "code_commit": replay_commit,
    "source_training_commit": source_commit,
    "selected_epoch": selected_zero + 1,
    "selected_checkpoint_path": str(selected_target),
    "selected_checkpoint_sha256": sha(selected_target),
    "epoch12_resume_checkpoint_path": str(final_target),
    "epoch12_resume_checkpoint_sha256": sha(final_target),
    "calibration_replay_only": True,
    "reporting_accessed": False,
    "threshold_search": False,
    "raw_rgb_authorized": False,
}
(artifacts / "calibration_promotion.json").write_text(
    json.dumps(promotion, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

completion = {
    "schema_version": "persistent_binding_feature_multi_epoch_arm.v1",
    "arm": arm,
    "seed": 705,
    "epochs": 12,
    "checkpoint_epochs": [3, 6, 9, 12],
    "code_commit": replay_commit,
    "source_training_commit": source_commit,
    "source_training_run": str(source_run),
    "calibration_replay_only": True,
    "replay_event_payload_equality_required": True,
    "expected_updates": int(totals["expected_updates"]),
    "successful_updates": int(totals["successful_updates"]),
    "scheduler_steps": int(totals["scheduler_steps"]),
    "skipped_updates": int(totals["skipped_updates"]),
    "gt_supervision_exhaustions": int(totals["gt_supervision_exhaustions"]),
    "gt_birth_runtime_entry_free_collisions": int(
        totals["gt_birth_runtime_entry_free_collisions"]
    ),
    "calibration_curve": curve,
    "selected_epoch": selected_zero + 1,
    "reporting_accessed": False,
    "threshold_search": False,
    "raw_rgb_authorized": False,
}
(artifacts / "formal12_arm_completion.json").write_text(
    json.dumps(completion, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
manifest = []
for path in sorted(path for path in artifacts.rglob("*") if path.is_file()):
    manifest.append(
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
    )
(output / "artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(completion, sort_keys=True))
PY
case "\$LOCAL_ROOT" in
    /tmp/\${USER}_ontad_formal12_replay_\${SLURM_JOB_ID}_\${ARM})
        rm -rf -- "\$LOCAL_ROOT"
        ;;
    *)
        echo "Refusing unsafe replay cleanup path: \$LOCAL_ROOT" >&2
        exit 9
        ;;
esac
SBATCH
    bash -n "$job_script"
    local submit_output
    local job_id
    submit_output=$(sbatch --parsable "$job_script")
    job_id=${submit_output%%;*}
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Unable to parse calibration replay job id: $submit_output" >&2
        return 7
    fi
    printf '%s\n' "$job_id"
}

submit_finalize() {
    local fixed_job=$1
    local rematch_job=$2
    local job_script="$RUN_DIR/formal12_finalize.sbatch.sh"
    cat > "$job_script" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=pb_h2f12_replay_gate
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --exclude=$SBATCH_EXCLUDE
#SBATCH --dependency=afterok:$fixed_job:$rematch_job
#SBATCH --output=$RUN_DIR/formal12_finalize.%j.out
#SBATCH --error=$RUN_DIR/formal12_finalize.%j.err

set -euo pipefail
BASE_DIR='$BASE_DIR'
RUN_DIR='$RUN_DIR'
SOURCE_RUN='$SOURCE_RUN'
SOURCE_COMMIT='$SOURCE_COMMIT'
FIXED_CONFIG='$FIXED_CONFIG'
REMATCH_CONFIG='$REMATCH_CONFIG'
EXPECTED_COMMIT='$COMMIT_SHA'
FIXED_JOB='$fixed_job'
REMATCH_JOB='$rematch_job'
cd "\$BASE_DIR"
test "\$(git rev-parse HEAD)" = "\$EXPECTED_COMMIT"
test -z "\$(git status --porcelain)"
source tools/env/activate_n16r4_causaltad.sh
python tools/evaluate_persistent_binding_formal12.py \
    --repo "\$BASE_DIR" \
    --fixed-config "\$FIXED_CONFIG" \
    --rematch-config "\$REMATCH_CONFIG" \
    --fixed-root "\$RUN_DIR/fixed" \
    --rematch-root "\$RUN_DIR/rematch" \
    --census "\$RUN_DIR/split_census.json" \
    --formal-contract "\$RUN_DIR/formal12_contract.json" \
    --fixed-output "\$RUN_DIR/fixed/formal12_epoch12_result.json" \
    --rematch-output "\$RUN_DIR/rematch/formal12_epoch12_result.json" \
    --gate-output "\$RUN_DIR/formal12_gate.json"
python3 - "\$RUN_DIR" "\$EXPECTED_COMMIT" "\$SOURCE_COMMIT" \
    "\$SOURCE_RUN" "\$FIXED_JOB" "\$REMATCH_JOB" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
commit, source_commit, source_run, fixed_job, rematch_job = sys.argv[2:]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

gate_path = root / "formal12_gate.json"
gate = json.loads(gate_path.read_text(encoding="utf-8"))
completion = {
    "schema_version": "persistent_binding_feature_multi_epoch_pair.v1",
    "code_commit": commit,
    "source_training_commit": source_commit,
    "source_training_run": str(Path(source_run).resolve()),
    "calibration_replay_only": True,
    "seed": 705,
    "epochs": 12,
    "fixed_job_id": fixed_job,
    "rematch_job_id": rematch_job,
    "finalize_job_id": str(__import__("os").environ["SLURM_JOB_ID"]),
    "formal_fixed_threshold_pass": gate["formal_fixed_threshold_pass"],
    "reporting_accessed": False,
    "threshold_search": False,
    "raw_rgb_authorized": False,
}
(root / "formal12_pair_completion.json").write_text(
    json.dumps(completion, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
manifest = []
for path in sorted(path for path in root.rglob("*") if path.is_file()):
    if path.name == "formal12_artifact_manifest.json":
        continue
    if path.name.startswith("formal12_finalize."):
        continue
    manifest.append(
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
    )
(root / "formal12_artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(completion, sort_keys=True))
PY
SBATCH
    bash -n "$job_script"
    local submit_output
    local job_id
    submit_output=$(sbatch --parsable "$job_script")
    job_id=${submit_output%%;*}
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Unable to parse calibration replay finalizer id: $submit_output" >&2
        return 7
    fi
    printf '%s\n' "$job_id"
}

echo "PERSISTENT_BINDING_FORMAL12_REPLAY_RUN_DIR=$RUN_DIR"
echo "PERSISTENT_BINDING_FORMAL12_REPLAY_COMMIT=$COMMIT_SHA"
echo "PERSISTENT_BINDING_FORMAL12_SOURCE_RUN=$SOURCE_RUN"
echo "PERSISTENT_BINDING_FORMAL12_SOURCE_COMMIT=$SOURCE_COMMIT"
fixed_job=""
rematch_job=""
finalize_job=""
cancel_partial_pair() {
    if [[ -z "$finalize_job" ]]; then
        for job_id in "$fixed_job" "$rematch_job"; do
            if [[ "$job_id" =~ ^[0-9]+$ ]]; then
                scancel "$job_id" || true
            fi
        done
    fi
}
trap cancel_partial_pair ERR INT TERM
fixed_job=$(submit_arm fixed "$FIXED_CONFIG" pb_h2f12_replay_fixed)
rematch_job=$(submit_arm rematch "$REMATCH_CONFIG" pb_h2f12_replay_rematch)
finalize_job=$(submit_finalize "$fixed_job" "$rematch_job")
trap - ERR INT TERM
python3 - "$RUN_DIR/formal12_launch.json" "$COMMIT_SHA" "$SOURCE_COMMIT" \
    "$SOURCE_RUN" "$fixed_job" "$rematch_job" "$finalize_job" <<'PY'
import json
from pathlib import Path
import sys

output, commit, source_commit, source_run, fixed, rematch, finalize = sys.argv[1:]
payload = {
    "schema_version": "persistent_binding_feature_multi_epoch_launch.v1",
    "code_commit": commit,
    "source_training_commit": source_commit,
    "source_training_run": str(Path(source_run).resolve()),
    "calibration_replay_only": True,
    "seed": 705,
    "epochs": 12,
    "fixed_job_id": fixed,
    "rematch_job_id": rematch,
    "finalize_job_id": finalize,
    "paired_submission": True,
    "reporting_accessed": False,
    "threshold_search": False,
    "raw_rgb_authorized": False,
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
echo "PERSISTENT_BINDING_FORMAL12_REPLAY_FIXED_JOB=$fixed_job"
echo "PERSISTENT_BINDING_FORMAL12_REPLAY_REMATCH_JOB=$rematch_job"
echo "PERSISTENT_BINDING_FORMAL12_REPLAY_FINALIZE_JOB=$finalize_job"
