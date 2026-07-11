@RTK.md

# Agent Entry

This is a clean Online/Causal TAD OpenTAD route repository. Follow `RTK.md` first.

Keep the repository focused on CausalTAD code, route configs, required OpenTAD support code, launch/check helpers, and concise operating notes. Do not reintroduce old logs, checkpoints, feature dumps, sync bundles, or generated result archives.

When multiple agents must share Rosetta Chrome port `9223`, follow `ROSETTA_CHROME_9223_RULES.md`: acquire `.codex/chrome-9223.lock`, bind operations to an explicit `targetId`/page id, heartbeat the lock, and release it after use.
