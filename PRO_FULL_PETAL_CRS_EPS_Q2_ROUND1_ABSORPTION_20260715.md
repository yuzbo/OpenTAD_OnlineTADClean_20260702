# Full PETAL / CRS-EPS / Q2 Round-1 Review Absorption, 2026-07-15

## Source

- Archived response: `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_REVIEW_20260715.md`
- Source attachment: `c238c5ad-5124-4b4c-bd3b-0511d2778956/pasted-text.txt`
- Source SHA-256: `BCAC3DA1DB07B9F103D50ABAF35D8550383263FC2E7797B73E89221420DE1680`
- Prompt publication HEAD observed by Pro: `c56857b1e3bbbd445aa86e9e31a1e1ddd07ffe51`
- Immutable implementation anchor reviewed: `f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb`
- Round: 1 of 2
- Pro verdict: `REVISE-BEFORE-IMPLEMENTATION`

The review was archived byte-identically. It did not report experimental
effectiveness. It audited the public code, reconstructed the current training
and cost path, and reduced the remaining author decisions to Q1-Q12.

## Independent Verdict

**Accept the Round-1 central verdict and code findings. Do not profile or
train.**

The most important correction is now explicit:

> The current implementation is complete-video chronological cached-feature
> training with one optimizer event per video and chunk-detached state. It is
> not CRS-EPS, event-centric sampling, prefix-episode training, or raw-video
> end-to-end learning.

No substantive code claim in Round 1 requires rejection. The provisional
hybrid recommendation is reasonable but must remain unfrozen until the author
answers Q1-Q6 and Pro completes Round 2.

## Accepted Findings

### Current objective and compute

- `StreamingFeatureDataset` enumerates every selected cache token in contiguous
  chronological chunks.
- Every valid token enters the sequential Python detector scan and loss path.
- Nonterminal chunks backpropagate but do not step the optimizer.
- One complete video creates one optimizer/scheduler event.
- Chunk losses are weighted by valid-token count and the accumulated gradient
  is normalized by total video tokens at the video boundary.
- The resulting empirical target is a video-uniform mean of per-video token
  means, not a corpus-token-uniform objective.
- Numerical state crosses chunk boundaries, but its tensors are detached;
  endpoint losses cannot assign credit through earlier chunks.

### CRS-EPS status

Current code contains none of the defining CRS-EPS mechanisms:

- no event-centered anchor mixture;
- no immutable sampled-episode manifest;
- no bounded burn-in plus short supervised suffix;
- no active-at-entry reconstruction contract;
- no inclusion probabilities, IPW/SNIPW, or ESS;
- no sampled/full loss, gradient, or state audit.

The existing route is therefore a reference for its exact cached-token,
per-video-normalized, chunk-detached objective. It is not an unbiased
full-sequence gold objective without those qualifications.

### Q2 identifiability

The fixed/rematch treatment is locally clean: config, source features, model,
birth assignment, canonical lifecycle, inference, masks in covered tests, and
capacity are shared. Existing synthetic probes show the intended loss-binding
and gradient intervention. The experiment remains only partially identifiable
because a complete paired execution trace is absent and no scientific result
exists.

### Profile and launch

- `P0-LAUNCH-WORKDIR` is independently reconfirmed.
- The current 50-warmup plus 200-measured optimizer-event profile is meaningful
  only within the same complete-video route and matched event order.
- It is invalid as the primary comparison between a complete-video event and a
  roughly 196-200-token CRS-EPS event.
- A cross-protocol profile must report multiple physical and statistical
  denominators rather than only events per second.

### Scope

- Detector-level cached-token causal execution is supported.
- Raw-video causality remains unproven because extractor provenance and raw
  support intervals are unbound.
- Low wall-clock latency is unsupported because source time is recorded, while
  packet availability and completion time are not.
- Full PETAL remains a provisional reconstruction risk; fixed binding remains
  an unproven marginal candidate.

## Independent Author-Decision Recommendations

These recommendations are drafted for the author response and are not
experimental facts.

1. Preserve the current reference target: sample videos uniformly, then sample
   decision times uniformly within each video.
2. Use CRS-EPS as the candidate main trainer, a preregistered tiny full-stream
   subset for state/loss/gradient audit, and complete chronological validation
   and testing. Do not add periodic full-stream training or post-hoc finetuning
   to the primary Q2 route.
3. Replay gold-subset state from the video start. Never initialize active slots
   from GT.
4. In sampled episodes, extend context to the earliest prefix-observable birth
   of every instance active in the supervised suffix. Fall back to video-start
   replay when needed; use left-censor masks only for true dataset censoring,
   not artificial context truncation.
5. Cap all future pre-Stage-2 GPU work at 10 GPU-hours, including any cache
   rebuild, profile, gold audit, Q2 training, and evaluation.
6. Replace cross-protocol optimizer-event comparison with GPU-hours, wall time,
   temporal forward/backward tokens, supervised bins, frames, ESS, and
   time-to-fixed-weighted-supervision accounting.
7. Use standard average temporal mAP over tIoU 0.3:0.7 as the primary quality
   endpoint on full chronological immutable emissions. Treat identity-linked
   errors as Q2 mechanism endpoints and recall/FN plus completion delay as
   safety endpoints.
8. Kill Full PETAL and raw-video Stage 2 if fixed binding does not beat rematch
   by a preregistered meaningful effect, if its gain is absent from
   identity-linked errors, or if recall/delay materially regresses.
9. Rebuild the feature cache before any effectiveness run if exact extractor
   provenance cannot be recovered. An old cache may support a clearly labeled
   compute-only profile after P0 closes.
10. Keep 211-versus-213 unknown until a source-derived manifest diff names the
    two videos and reasons. Use the exact same population for every baseline.
11. Separate training chunks from serving packets. Use one-token primary
    serving/evaluation cadence and record source, availability, and completion
    clocks.
12. Require a dense/overlap second dataset for any retained general paper
    claim; it is not required for the first THUMOS mechanism kill test.

## Required Gate Order

1. Send the Q1-Q12 author response and obtain Round-2 protocol adjudication.
2. In parallel only with discussion, fix `P0-LAUNCH-WORKDIR` and add the real
   deterministic fake-`sbatch` integration test; do not start GPU work.
3. Freeze the Round-2-selected target risk, state replay, manifest, weighting,
   gold audit, profile, metrics, and kill contracts.
4. Implement CRS-EPS only after that protocol is fixed.
5. Freeze a clean commit, regenerate full B0, and obtain a new independent
   `PASS / PROFILE=ALLOW`.
6. Run a bounded multi-denominator profile.
7. Only after profile and a separate formal authorization may the paired Q2
   mechanism kill test begin.

## Final Disposition

```text
INDEPENDENT_VERDICT=REVISE_BEFORE_IMPLEMENTATION
ACCEPT_ROUND1_CODE_FINDINGS=YES
CURRENT_ROUTE=FULL_CHRONOLOGICAL_CACHED_VIDEO_EPISODE_TRUNCATED_BPTT
CRS_EPS_IMPLEMENTED=NO
RECOMMENDED_PROTOCOL=HYBRID_PROVISIONAL
Q2_IDENTIFIABILITY=PARTIALLY_IDENTIFIABLE
PROFILE_CONTRACT=REPLACE
FULL_PETAL_NOVELTY=RECONSTRUCTION_PROVISIONAL
RAW_VIDEO_STAGE2=BLOCKED
PROFILE=BLOCKED
FORMAL_TRAINING=BLOCKED
NEXT_DISCUSSION=ANSWER_Q1_TO_Q12_AND_RUN_ROUND2
NEXT_ENGINEERING=FIX_P0_LAUNCH_WORKDIR_WITH_FAKE_SBATCH_TEST
```
