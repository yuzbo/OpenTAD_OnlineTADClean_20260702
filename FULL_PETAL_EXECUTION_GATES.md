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
python tools/run_full_petal_b0.py `
  --output-dir C:\path\outside\repo\full-petal-b0 `
  --torch-python C:\path\to\working-torch-python.exe `
  --dependency-site C:\path\to\pytest-site-packages
```

The B0 runner refuses a dirty checkout, refuses an output directory inside the
repository, executes the non-Torch and focused Torch contract suites, records
JUnit and raw logs, hashes every leaf artifact, checks every tracked Python
source, runs `git diff --check`, and verifies that the checkout remains clean.

## Prohibited Shortcuts

- No legacy Stage-1 smoke launcher for either Q2 Full PETAL config.
- No direct `tools/train.py` or `tools/test.py` launch without a valid ticket.
- No non-Slurm GPU launch.
- No scientific `--cfg-options`; only `work_dir` is allowed.
- No profile before B0 and the locked independent review reach PASS.
- No formal training before a passing fixed-step profile and explicit formal
  authorization.
