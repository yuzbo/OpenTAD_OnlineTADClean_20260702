# Persistent-Binding Slurm Smoke Record

Date: 2026-07-20
Status: passed
Scope: execution and causal-protocol verification, not an effectiveness result

## Frozen Run

- Git commit: `097bc72a0ca5840b09050a3af4ae35ea704d965b`
- Slurm job: `1176737`
- State: `COMPLETED`
- Elapsed: `00:06:08`
- GPU: one RTX 4090
- Remote run directory:
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260720_200015`
- Gate SHA-256:
  `58540459982bb5177d1602abad78d81c09bc42e3202897a6e6fed03f152d2f90`

The run directory, checkpoint, ledgers, and logs remain outside the repository.

## Real Data Selection

The smoke-only selector chose the shortest cached video with at least one
annotation from each frozen canonical split:

- train: `video_validation_0000190`, 34 tokens, 2 annotations;
- calibration: `video_validation_0000689`, 77 tokens, 1 annotation;
- reporting: `video_test_0000062`, 56 tokens, 2 annotations.

This selection minimizes smoke cost. It is not used for model selection or
scientific reporting.

## Passed Checks

- 64 focused tests passed in the allocated GPU job.
- FIXED and REMATCH each completed a real-feature backward and optimizer update.
- Both direct binding reports had zero dropped GT birth targets and zero runtime
  capacity exhaustions on the selected train stream.
- The standard OpenTAD runner completed FP32 forward, backward, checkpoint,
  streaming inference, immutable-ledger serialization, and evaluation.
- The checkpoint contained 29 optimizer-state entries.
- 29 state tensors changed relative to a freshly initialized model with the
  same config and seed; aggregate L2 change was `0.6226291824`.
- Reloading the checkpoint reproduced the original 45 final emissions exactly.
- The train and reload ledgers had the same SHA-256:
  `1cfb499157000035782b176e014c77e863621a5eead8060a71fe7f13ca65886f`.
- The ledger had zero future-end, future-source, negative-latency, and
  non-monotonic-emission violations.
- The evaluator consumed 2 real GT instances and all 45 immutable emissions.

## Failures Found Before the Passing Run

Three earlier smoke attempts were intentionally rejected:

1. The generic optimizer dereferenced `backbone.freeze_backbone` when the
   feature-only detector had `backbone=None`.
2. The inherited FP16 AMP route produced a non-finite first-step gradient and
   the generic loop skipped the update.
3. The one-step smoke scheduler initially used learning rate zero, leaving the
   checkpoint identical to initialization.

The final route supports feature-only optimization, uses FP32, makes non-finite
training a hard failure, gives the one-step smoke a nonzero learning rate, and
requires both optimizer state and measured parameter change.

## Scientific Interpretation

The one-video, one-epoch model emitted 45 intervals but matched no GT at the
frozen OnlineAP settings. This is not a failed method result because the smoke
does not train an effectiveness model. It confirms only that the complete
feature-level route executes, updates parameters, reloads deterministically,
emits final causal intervals, and reaches the evaluator.

`formal_training_ready` remains `False`. The next gate is one matched seed
(`705`) for FIXED and REMATCH on the frozen fit/calibration/reporting protocol.
Raw-RGB training remains blocked.
