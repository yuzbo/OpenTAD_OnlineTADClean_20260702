---
type: paper
node_id: paper:thinkingqwen2026-evidence-timing
title: "Progressive Online Video Understanding with Evidence-Aligned Timing and Transparent Decisions"
authors: ["Kecheng Zhang", "Zongxin Yang", "Mingfei Han", "Haihong Hao", "Yunzhi Zhuge", "Changlin Li", "Junhan Zhao", "Zhihui Li", "Xiaojun Chang"]
year: 2026
venue: "ICLR"
external_ids:
  arxiv: null
  doi: null
tags: ["streaming-video", "response-timing", "hypothesis-revision", "causal-state", "transparent-decision"]
added: 2026-07-11
---

# Thinking-QwenVL / ATDM 2026

## One-line thesis

An online video model should progressively update a compact causal state, revise hypotheses as evidence arrives, expose progress/confidence, and respond at the first sufficient-evidence time.

## Method

- Active Thinking Decision Maker for evidence-aligned response timing.
- Hierarchical Progressive Semantic Integration for compact cross-clip state.
- Observable progress and confidence variables for transparent decisions.

## Limitations / Failure Modes

The task is query answering and streaming VLM interaction, not action-instance localization with temporal IoU and immutable detection records.

## Relevance to This Project

This is the strongest adjacent threat to broad language such as "progressive online understanding," "causal semantic state maintenance," "hypothesis revision," and "decide when to respond." CESR must define a task-specific delta: identity-preserving temporal span beliefs and utility-based detection commit.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
