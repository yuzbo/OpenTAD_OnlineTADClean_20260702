---
type: experiment
node_id: exp:cancelled-pilots-20260710
title: "Cancelled PCEH/endpoint pilots 2026-07-10"
status: incomplete
verdict: negative-for-cost
updated: 2026-07-11
---

# Cancelled Pilots 2026-07-10

## Summary

Several PCEH/endpoint pilot jobs started but were cancelled after short runtime. Logs showed training began, losses changed, and some non-finite gradient skips occurred, but no short pilot completed.

## Main Lesson

The full-packet training route is too slow and fragile to use as the default formal protocol.

## What It Supports

- Need for CRS-EPS.
- Need for feature cache.
- Need for throughput benchmark before long jobs.

## What It Does Not Support

- Any model effectiveness claim.
- PCEH vs endpoint-only comparison.
- Formal training readiness.
