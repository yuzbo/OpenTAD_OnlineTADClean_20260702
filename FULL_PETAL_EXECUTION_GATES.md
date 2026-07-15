# Full PETAL Execution Gates

This route is fail-closed. The order below is mandatory.

## Gate Order

1. Freeze a clean git commit containing the P0, lifecycle, data/metric, optimizer,
   and launch-gate implementation.
2. On the clean target Linux/N16R4 checkout, run
   `tools/run_full_petal_posix_b0.py` and store its signed leaf outside the
   repository. The leaf must bind the same commit and locked B0 manifest.
3. Run the complete local CPU B0 matrix with `tools/run_full_petal_b0.py`, pass
   it the signed Linux leaf with `--posix-leaf`, and store the root bundle
   outside the repository.
4. Give the exact commit and the generated `b0.json` to reviewer
   `019f5abd-5104-79b3-882e-354ca796f2c1` for the ordered scope recorded in
   `launch_contract.required_review_scope`.
5. Continue only when that reviewer returns `PASS` with zero blocking findings
   and zero protocol violations.
6. For CRS-EPS, run the preregistered CPU-only four-arm G0 audit with
   `tools/run_crs_eps_gold_audit.py`. The signed selection and margins must be
   created first by `tools/preregister_crs_eps_gold_audit.py`; both bind the
   exact commit, resolved/scientific config, data identity, sampling population,
   episode manifest, and one another.
7. Build a profile ticket with `tools/build_full_petal_launch_ticket.py`.
8. Submit exactly one fixed-step profile through
   `tools/remote/submit_full_petal_q2_n16r4.sh profile ...`.
9. Do not save checkpoints, evaluate results, or continue training from the
   profile process. Its only accepted output is `fixed_step_profile.json`.
10. Formal training requires a later authorization-only commit, a fresh B0 and
   same-reviewer PASS for that commit, and the hash-bound profile artifact.

## B0 Example

First issue the Linux leaf from the exact clean target checkout:

```bash
python tools/run_full_petal_posix_b0.py \
  --output-dir /path/outside/repo/full-petal-posix-b0 \
  --torch-python /path/to/working-torch-python \
  --attestation-private-key /path/outside/repo/b0-runner.pem \
  --attestation-key-id full-petal-b0-20260713
```

Then build the signed root locally:

```powershell
& C:\path\to\control-python.exe tools/run_full_petal_b0.py `
  --output-dir C:\path\outside\repo\full-petal-b0 `
  --torch-python C:\path\to\working-torch-python.exe `
  --posix-leaf C:\path\to\copied-posix-b0\posix-b0.json `
  --attestation-private-key C:\path\outside\repo\b0-runner.pem `
  --attestation-key-id full-petal-b0-20260713
```

The B0 runner refuses a dirty checkout and an output directory inside the
repository. It first verifies and embeds the signed Linux leaf, then executes
every discovered test through the locked Torch runner,
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

## CRS-EPS G0 Inputs

`tools/run_crs_eps_gold_audit.py` is deliberately CPU-only. It requires a clean
commit, a CRS-EPS config, a frozen checkpoint, an immutable episode manifest,
and signed preregistration artifacts with schemas
`full-petal-crs-eps-g0-selection-v3` and
`full-petal-crs-eps-g0-margins-v3`. Selection preregistration signs the exact
checkpoint relative path, SHA-256, byte size, state key, and deterministic
initialization identity before any replay outcome. The preregistration tool
reproduces the model state from the manifest seed and exact config. Before
model construction the audit runner re-verifies those checkpoint bytes and
rebuilds the loaded dataset identity, checking the annotation, feature-cache
manifest, split files, sampling population, config, and commit against the
manifest. It compares each selected draw against video-start gold using
`dynamic_birth`, `fixed_192`, and `reset` arms under a restored RNG snapshot.
The final `full-petal-crs-eps-g0-audit-v3` artifact is signed and always
terminal: exactly `PASS` or `KILL`.

The profile ticket schema requires the signed G0 audit for every CRS-EPS route.
At launch, the validator re-verifies all three signatures, reloads the manifest,
recursively rehashes the checkpoint bytes, recomputes the gate from the trace
rows and preregistered margins, and requires the selected sample set and
checkpoint identity to match exactly. A missing, killed, substituted, or
cross-commit G0 artifact blocks before CUDA/DDP initialization.

The configured `draws_per_video=4` remains
`PROVISIONAL_IMPLEMENTATION_PROBE_UNTIL_G0`. It must not silently become the
formal training value.

## Prohibited Shortcuts

- No legacy Stage-1 smoke launcher for either Q2 Full PETAL config.
- No direct `tools/train.py` or `tools/test.py` launch without a valid ticket.
- No ticket reuse or overwrite; every launch produces one immutable receipt.
- No non-Slurm GPU launch.
- No scientific `--cfg-options`; only `work_dir` is allowed.
- No profile before B0 and the locked independent review reach PASS.
- No formal training before a passing fixed-step profile and explicit formal
  authorization.
