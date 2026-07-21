#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_DIR=${BASE_DIR:-$REPO_ROOT}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
PILOT_RUN=${PILOT_RUN:?set PILOT_RUN to the completed H2 epoch-1 run}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/persistent_binding}
SBATCH_EXCLUDE=${SBATCH_EXCLUDE:-g0063}
SEED=705
VARIANT=boundary_calibration_batched
FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_formal12_boundary_calibration_batched_fixed.py
REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_formal12_boundary_calibration_batched_rematch.py

COMMIT_SHA=$(git -C "$BASE_DIR" rev-parse HEAD)
if [[ -z "$EXPECTED_COMMIT" || "$COMMIT_SHA" != "$EXPECTED_COMMIT" ]]; then
    echo "Formal12 checkout does not match EXPECTED_COMMIT" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Formal12 checkout must be clean" >&2
    exit 2
fi
for config in "$FIXED_CONFIG" "$REMATCH_CONFIG"; do
    test -f "$BASE_DIR/$config"
done

python3 - "$PILOT_RUN" "$COMMIT_SHA" "$BASE_DIR" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
formal_commit = sys.argv[2]
repo = Path(sys.argv[3]).resolve()
required = (
    "pilot_contract.json",
    "split_census.json",
    "optimization_activation.json",
    "fixed_score_diagnosis.json",
    "rematch_score_diagnosis.json",
    "screen_gate.json",
)
for name in required:
    if not (root / name).is_file():
        raise SystemExit(f"pilot prerequisite missing: {name}")
screen = json.loads((root / "screen_gate.json").read_text(encoding="utf-8"))
if screen.get("schema_version") != "persistent_binding_seed705_screen_gate.v2":
    raise SystemExit("formal12 requires the epoch-1 layered gate v2")
if screen.get("learning_readiness_pass") is not True:
    raise SystemExit("H2 did not pass epoch-1 learning readiness")
if screen.get("reporting_accessed") is not False:
    raise SystemExit("pilot accessed reporting")
activation = json.loads(
    (root / "optimization_activation.json").read_text(encoding="utf-8")
)
if activation.get("variant") != "boundary_calibration_batched":
    raise SystemExit("pilot variant mismatch")
if activation.get("passed") is not True:
    raise SystemExit("pilot mechanism activation failed")
for arm in ("fixed", "rematch"):
    diagnosis = json.loads(
        (root / f"{arm}_score_diagnosis.json").read_text(encoding="utf-8")
    )
    calibration = diagnosis.get("lifecycle_calibration_invariance", {})
    boundary = diagnosis.get("boundary_factorization", {})
    pointer = boundary.get("endpoint_pointer", {})
    if calibration.get("passed") is not True:
        raise SystemExit(f"{arm} monotone calibration gate failed")
    if boundary.get("end_transition_mode") != "causal_delta_mlp":
        raise SystemExit(f"{arm} transition-end mode failed")
    if boundary.get("endpoint_start_mode") != "past_pointer":
        raise SystemExit(f"{arm} endpoint-start mode failed")
    if pointer.get("past_only") is not True:
        raise SystemExit(f"{arm} pointer accessed future memory")
    if boundary.get("runtime_state_contains_gt") is not False:
        raise SystemExit(f"{arm} runtime state contains GT")
pilot = json.loads((root / "pilot_contract.json").read_text(encoding="utf-8"))
pilot_commit = pilot.get("code_commit")
if not isinstance(pilot_commit, str) or len(pilot_commit) != 40:
    raise SystemExit("pilot commit is invalid")
changed = subprocess.run(
    [
        "git",
        "-C",
        str(repo),
        "diff",
        "--name-only",
        f"{pilot_commit}..{formal_commit}",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
allowed_prefixes = (
    "configs/causaltad/thumos_persistent_binding_formal12_",
    "research-wiki/",
    "tests/",
    "tools/evaluate_persistent_binding_formal12.py",
    "tools/remote/submit_persistent_binding_formal12_n16r4.sh",
)
unexpected = [
    path for path in changed
    if not any(path.startswith(prefix) for prefix in allowed_prefixes)
]
if unexpected:
    raise SystemExit(f"formal12 changed model/runtime code after pilot: {unexpected}")
PY

for name in pb_h2f12_fixed pb_h2f12_rematch; do
    if squeue -u "$USER" -h -n "$name" | grep -q .; then
        echo "Refusing duplicate active formal12 job: $name" >&2
        exit 3
    fi
done

timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RUNS_ROOT/formal12_boundary_calibration_batched_seed705_$timestamp"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/fixed" "$RUN_DIR/rematch"
cp "$PILOT_RUN/split_census.json" "$RUN_DIR/split_census.json"

python3 - \
    "$RUN_DIR/formal12_contract.json" \
    "$COMMIT_SHA" \
    "$PILOT_RUN" \
    "$FIXED_CONFIG" \
    "$REMATCH_CONFIG" \
    "$RUN_DIR/split_census.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output, commit, pilot, fixed, rematch, census = sys.argv[1:]
pilot_root = Path(pilot).resolve()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema_version": "persistent_binding_feature_multi_epoch_deployment.v1",
    "variant": "boundary_calibration_batched",
    "code_commit": commit,
    "pilot_run": str(pilot_root),
    "pilot_gate_sha256": sha(pilot_root / "screen_gate.json"),
    "pilot_activation_sha256": sha(
        pilot_root / "optimization_activation.json"
    ),
    "pilot_fixed_diagnosis_sha256": sha(
        pilot_root / "fixed_score_diagnosis.json"
    ),
    "pilot_rematch_diagnosis_sha256": sha(
        pilot_root / "rematch_score_diagnosis.json"
    ),
    "split_census_sha256": sha(census),
    "seed": 705,
    "epochs": 12,
    "checkpoint_epochs": [3, 6, 9, 12],
    "initialization": "seed_initialization_not_pilot_resume",
    "fixed_config": fixed,
    "rematch_config": rematch,
    "input": "fixed_cached_causal_features",
    "fit_only": True,
    "checkpoint_selection_split": "calibration_only",
    "reporting_accessed": False,
    "threshold_search": False,
    "fixed_thresholds": {"birth": 0.5, "alive": 0.5, "end": 0.5},
    "formal_fixed_threshold_gate_epoch": 12,
    "raw_rgb_authorized": False,
    "paired_gpu_hour_cap": 16.0,
    "submit_via_slurm_only": True,
}
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
    local job_script="$arm_root/formal12.sbatch.sh"
    cat > "$job_script" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=$job_name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=$SBATCH_EXCLUDE
#SBATCH --output=$arm_root/slurm.%j.out
#SBATCH --error=$arm_root/slurm.%j.err

set -euo pipefail
BASE_DIR='$BASE_DIR'
RUN_DIR='$RUN_DIR'
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
LOCAL_ROOT=/tmp/\${USER}_ontad_formal12_\${SLURM_JOB_ID}_\${ARM}
test ! -e "\$LOCAL_ROOT"
mkdir -p "\$LOCAL_ROOT/train"
python3 - "\$CONFIG" "\$ARM" <<'PY'
from mmengine.config import Config
import sys
cfg = Config.fromfile(sys.argv[1])
arm = sys.argv[2]
expected = {"fixed": "fixed_birth_slot", "rematch": "prefix_rematch_active_pool"}[arm]
assert cfg.formal_training_ready is True
assert cfg.screening_only is False
assert cfg.model.trajectory_binding_mode == expected
assert cfg.workflow.fit_only is True
assert cfg.workflow.end_epoch == 12
assert cfg.workflow.checkpoint_interval == 3
assert cfg.workflow.val_eval_interval == -1
assert cfg.multi_epoch_training_contract.reporting_accessed is False
assert cfg.multi_epoch_training_contract.threshold_search is False
assert cfg.multi_epoch_training_contract.raw_rgb_authorized is False
assert tuple(cfg.multi_epoch_training_contract.checkpoint_epochs) == (3, 6, 9, 12)
assert cfg.model.head.birth_threshold == 0.5
assert cfg.model.head.alive_threshold == 0.5
assert cfg.model.head.end_threshold == 0.5
PY
PORT_SLOT=\$((SLURM_JOB_ID % 5000))
MASTER_PORT=\$((20000 + PORT_SLOT * 8))
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
    --rdzv_endpoint "127.0.0.1:\${MASTER_PORT}" \
    tools/train.py "\$CONFIG" \
    --seed "\$SEED" \
    --id 0 \
    --cfg-options work_dir="\$LOCAL_ROOT/train"
WORK="\$LOCAL_ROOT/train/gpu1_id0"
AUDIT="\$WORK/training_audit.json"
test -f "\$AUDIT"
candidate_args=()
for spec in 3:2 6:5 9:8 12:11; do
    human=\${spec%%:*}
    zero=\${spec##*:}
    checkpoint="\$WORK/checkpoint/epoch_\${zero}.pth"
    test -f "\$checkpoint"
    eval_root="\$LOCAL_ROOT/eval_epoch_\${human}"
    MASTER_PORT=\$((MASTER_PORT + 1))
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
        --rdzv_endpoint "127.0.0.1:\${MASTER_PORT}" \
        tools/test.py "\$CONFIG" \
        --evaluation-role calibration \
        --checkpoint "\$checkpoint" \
        --seed "\$SEED" \
        --id 0 \
        --cfg-options work_dir="\$eval_root"
    ledger="\$eval_root/gpu1_id0/calibration/persistent_binding_emissions.json"
    candidate="\$LOCAL_ROOT/calibration_candidate_epoch_\${human}.json"
    test -f "\$ledger"
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
    candidate_args+=(--candidate "\$candidate")
done
RECEIPT="\$LOCAL_ROOT/calibration_receipt.json"
python tools/select_persistent_binding_checkpoint.py \
    "\${candidate_args[@]}" \
    --output "\$RECEIPT"
ENDED=\$(date +%s)
python tools/build_persistent_binding_resource_report.py \
    --repo "\$BASE_DIR" \
    --scope "\$ARM" \
    --started-unix "\$STARTED" \
    --ended-unix "\$ENDED" \
    --gpu-count 1 \
    --gpu-name "\$GPU_NAME" \
    --slurm-job-id "\${SLURM_JOB_ID}" \
    --output "\$LOCAL_ROOT/resource_report.json"
python3 - "\$LOCAL_ROOT" "\$RUN_DIR/\$ARM" "\$CONFIG" "\$ARM" "\$EXPECTED_COMMIT" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

local = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
config = Path(sys.argv[3]).resolve()
arm = sys.argv[4]
commit = sys.argv[5]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

audit = load(local / "train/gpu1_id0/training_audit.json")
if audit.get("schema_version") != "persistent_binding_training_audit.v1":
    raise SystemExit("unexpected training audit schema")
if audit.get("fit_only") is not True or audit.get("screening_only") is not False:
    raise SystemExit("formal12 audit route flags are invalid")
if audit.get("route_stage") != "persistent_binding_feature_multi_epoch_calibration":
    raise SystemExit("formal12 audit route stage mismatch")
epochs = audit.get("epochs", [])
if [int(row.get("epoch", -1)) for row in epochs] != list(range(12)):
    raise SystemExit("formal12 audit does not contain epochs 0..11")
totals = audit.get("totals", {})
expected = int(totals.get("expected_updates", -1))
if expected != 12 * 2010:
    raise SystemExit(f"formal12 expected updates mismatch: {expected}")
for field in ("successful_updates", "scheduler_steps"):
    if int(totals.get(field, -1)) != expected:
        raise SystemExit(f"formal12 {field} mismatch")
for field in (
    "skipped_updates",
    "gt_supervision_exhaustions",
    "gt_birth_runtime_entry_free_collisions",
):
    if int(totals.get(field, -1)) != 0:
        raise SystemExit(f"formal12 {field} is nonzero")

receipt = load(local / "calibration_receipt.json")
if receipt.get("candidate_count") != 4:
    raise SystemExit("formal12 calibration must compare four checkpoints")
if receipt.get("reporting_accessed") is not False:
    raise SystemExit("formal12 calibration accessed reporting")
selected_zero = int(receipt["selected_epoch"])
if selected_zero not in (2, 5, 8, 11):
    raise SystemExit("selected checkpoint is outside epochs 3/6/9/12")

artifacts = output / "artifacts"
artifacts.mkdir(parents=True, exist_ok=False)
shutil.copy2(config, artifacts / "config.py")
shutil.copy2(
    local / "train/gpu1_id0/training_audit.json",
    artifacts / "training_audit.json",
)
shutil.copy2(local / "resource_report.json", artifacts / "resource_report.json")
shutil.copy2(
    local / "calibration_receipt.json",
    artifacts / "calibration_receipt.json",
)

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
    curve.append(
        {
            "epoch": human,
            "checkpoint_epoch": zero,
            "metric_name": candidate["metric_name"],
            "metric_value": candidate["metric_value"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "emission_ledger_sha256": sha(ledger_target),
        }
    )

checkpoint_dir = artifacts / "checkpoint"
checkpoint_dir.mkdir()
final_source = local / "train/gpu1_id0/checkpoint/epoch_11.pth"
final_target = checkpoint_dir / "epoch_12_resume.pth"
shutil.copy2(final_source, final_target)
selected_source = Path(receipt["selected_checkpoint_path"]).resolve()
if sha(selected_source) != receipt["selected_checkpoint_sha256"]:
    raise SystemExit("selected checkpoint hash changed before promotion")
if selected_zero == 11:
    selected_target = final_target
else:
    selected_target = checkpoint_dir / f"selected_epoch_{selected_zero + 1}.pth"
    shutil.copy2(selected_source, selected_target)
if sha(selected_target) != receipt["selected_checkpoint_sha256"]:
    raise SystemExit("promoted selected checkpoint hash mismatch")

promotion = {
    "schema_version": "persistent_binding_calibration_promotion.v1",
    "arm": arm,
    "seed": 705,
    "code_commit": commit,
    "selected_epoch": selected_zero + 1,
    "selected_checkpoint_path": str(selected_target),
    "selected_checkpoint_sha256": sha(selected_target),
    "epoch12_resume_checkpoint_path": str(final_target),
    "epoch12_resume_checkpoint_sha256": sha(final_target),
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
    "code_commit": commit,
    "expected_updates": expected,
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
for path in sorted(p for p in artifacts.rglob("*") if p.is_file()):
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
    /tmp/\${USER}_ontad_formal12_\${SLURM_JOB_ID}_\${ARM})
        rm -rf -- "\$LOCAL_ROOT"
        ;;
    *)
        echo "Refusing unsafe local cleanup path: \$LOCAL_ROOT" >&2
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
        echo "Unable to parse formal12 Slurm job id: $submit_output" >&2
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
#SBATCH --job-name=pb_h2f12_gate
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --exclude=$SBATCH_EXCLUDE
#SBATCH --dependency=afterok:$fixed_job:$rematch_job
#SBATCH --output=$RUN_DIR/formal12_finalize.%j.out
#SBATCH --error=$RUN_DIR/formal12_finalize.%j.err

set -euo pipefail
BASE_DIR='$BASE_DIR'
RUN_DIR='$RUN_DIR'
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
python3 - "\$RUN_DIR" "\$EXPECTED_COMMIT" "\$FIXED_JOB" "\$REMATCH_JOB" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
fixed_job = sys.argv[3]
rematch_job = sys.argv[4]

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
for path in sorted(p for p in root.rglob("*") if p.is_file()):
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
        echo "Unable to parse formal12 finalizer job id: $submit_output" >&2
        return 7
    fi
    printf '%s\n' "$job_id"
}

echo "PERSISTENT_BINDING_FORMAL12_RUN_DIR=$RUN_DIR"
echo "PERSISTENT_BINDING_FORMAL12_COMMIT=$COMMIT_SHA"
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
fixed_job=$(submit_arm fixed "$FIXED_CONFIG" pb_h2f12_fixed)
rematch_job=$(submit_arm rematch "$REMATCH_CONFIG" pb_h2f12_rematch)
finalize_job=$(submit_finalize "$fixed_job" "$rematch_job")
trap - ERR INT TERM
python3 - \
    "$RUN_DIR/formal12_launch.json" \
    "$COMMIT_SHA" \
    "$fixed_job" \
    "$rematch_job" \
    "$finalize_job" <<'PY'
import json
from pathlib import Path
import sys

output, commit, fixed, rematch, finalize = sys.argv[1:]
payload = {
    "schema_version": "persistent_binding_feature_multi_epoch_launch.v1",
    "code_commit": commit,
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
echo "PERSISTENT_BINDING_FORMAL12_FIXED_JOB=$fixed_job"
echo "PERSISTENT_BINDING_FORMAL12_REMATCH_JOB=$rematch_job"
echo "PERSISTENT_BINDING_FORMAL12_FINALIZE_JOB=$finalize_job"
