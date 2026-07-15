---
type: paper
node_id: paper:oat2022-online-tal
title: "A Sliding Window Scheme for Online Temporal Action Localization"
authors: ["Young Hwi Kim", "Hyolim Kang", "Seon Joo Kim"]
year: 2022
venue: "ECCV"
external_ids:
  arxiv: null
  doi: "10.1007/978-3-031-19830-4_37"
tags: ["On-TAL", "early-proposal", "boundary-refinement", "baseline"]
added: 2026-07-11
updated: 2026-07-15
verification_status: primary-verified
primary_url: "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf"
---

# OAT / Sliding Window Scheme for Online Temporal Action Localization

## One-line thesis

Online Anchor Transformer applies anchor-based sliding-window localization to
On-TAL, emits proposals before action completion, refines interval boundaries,
and uses online suppression to control repeated proposals without future video
context.

## Verification Record

Independent exact-title search on 2026-07-15 resolved the official ECCV 2022
paper and DOI. This corrects the Round-2 review's erroneous
`OAT_PRIMARY_ARTIFACT=UNRESOLVED` statement.

```text
OAT_PRIMARY_ARTIFACT=VERIFIED
TITLE=A Sliding Window Scheme for Online Temporal Action Localization
DOI=10.1007/978-3-031-19830-4_37
```

Primary source: [official ECVA PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf).

## Relevance to This Project

This is a direct verified threat to naive early-hypothesis, anchor-window,
boundary-refinement, responsiveness, and repeated-proposal-suppression claims.

## Overlap

- online TAL setting;
- early action proposals;
- boundary/offset refinement;
- online metric emphasis.

## Key Difference We Need

CESR must emphasize:

- mutable hypothesis lifecycle;
- timestamped revision history;
- explicit commit policy;
- state-transition objectives, not just proposal offsets;
- revision metrics in addition to final AP.

## Reviewer Warning

Never claim early proposal, sliding-window anchors, online boundary refinement,
or learned online proposal suppression alone as novelty.
