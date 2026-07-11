---
type: experiment
node_id: exp:pceh-smoke-20260710
title: "PCEH smoke run 2026-07-10"
status: completed
verdict: partial
updated: 2026-07-11
---

# PCEH Smoke 2026-07-10

## Summary

Remote smoke run `pceh_smoke_17b9bb3_20260710_162823` completed and passed gate summary:

- packet audit;
- update audit;
- causal replay;
- frozen/adaptor route update checks.

## What It Supports

- The skeleton can run.
- Some prefix-causal smoke checks passed.
- Adapter/projection/head update path can be audited.

## What It Does Not Support

- formal paper result;
- multi-seed performance;
- valid PCEH risk-set objective;
- CESR mutable hypothesis protocol;
- LoRA or full visual tower finetuning;
- DDP streaming.

## Use

Protocol smoke only.
