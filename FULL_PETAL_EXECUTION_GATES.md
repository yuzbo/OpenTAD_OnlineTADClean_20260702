# Full PETAL Execution Gates

This route is fail-closed. The order below is mandatory.

## Gate Order

1. Freeze a clean git commit containing the P0, lifecycle, data/metric, optimizer,
   and launch-gate implementation.
2. Run the complete CPU B0 matrix with `tools/run_full_petal_b0.py`. Store the
   output outside the repository.
3. Give the exact commit and the generated `b0.json` to reviewer
   `019f5abd-5104-79b3-882e-354ca796f2c1` for the ordered scope recorded in
   `launch_contract.required_review_scope`.
4. Continue only when that reviewer returns `PASS` with zero blocking findings
   and zero protocol violations.
5. Build a profile ticket with `tools/build_full_petal_launch_ticket.py`.
6. Submit exactly one fixed-step profile through
   `tools/remote/submit_full_petal_q2_n16r4.sh profile ...`.
7. Do not save checkpoints, evaluate results, or continue training from the
   profile process. Its only accepted output is `fixed_step_profile.json`.
8. Formal training requires a later authorization-only commit, a fresh B0 and
   same-reviewer PASS for that commit, and the hash-bound profile artifact.

## B0 Example

```powershell
& C:\path\to\control-python.exe tools/run_full_petal_b0.py `
  --output-dir C:\path\outside\repo\full-petal-b0 `
  --torch-python C:\path\to\working-torch-python.exe `
  --attestation-private-key C:\path\outside\repo\b0-runner.pem `
  --attestation-key-id full-petal-b0-20260713
```

The B0 runner refuses a dirty checkout and an output directory inside the
repository. It executes every discovered test through the locked Torch runner,
records JUnit and raw logs, hashes every test and runner source, checks every
tracked Python source, runs `git diff --check`, and verifies that the checkout
remains clean.

Every accepted formal result must bind separate training and evaluation launch
tickets. Each entrypoint writes a non-overwritable `.receipt.json` beside its
ticket before CUDA/DDP initialization. The signed run manifest must include both
tickets and both receipts; the evaluation ticket must bind the exact checkpoint
path, size, and SHA-256. Build tickets and all evidence outside the repository.

Before building a formal run manifest, export the fully merged config used by
the run:

```powershell
python tools/export_full_petal_resolved_config.py `
  --config configs/causaltad/thumos_pes_q2_persist_fixed.py `
  --output C:\path\to\run-bundle\resolved-config.json
```

The formal manifest binds both `config` (the exact source config hashed by each
launch ticket) and `resolved_config` (the standalone JSON snapshot checked
against the ticket's resolved and scientific config digests). It must also bind
`fit_core` and `feature_cache_manifest`; training-order identity is derived from
their already verified bytes rather than reopening uncommitted data inputs.

## Prohibited Shortcuts

- No legacy Stage-1 smoke launcher for either Q2 Full PETAL config.
- No direct `tools/train.py` or `tools/test.py` launch without a valid ticket.
- No ticket reuse or overwrite; every launch produces one immutable receipt.
- No non-Slurm GPU launch.
- No scientific `--cfg-options`; only `work_dir` is allowed.
- No profile before B0 and the locked independent review reach PASS.
- No formal training before a passing fixed-step profile and explicit formal
  authorization.
