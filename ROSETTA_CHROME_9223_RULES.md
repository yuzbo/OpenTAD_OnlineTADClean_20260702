# Rosetta Chrome 9223 Shared-Port Rules

These rules apply only when multiple agents must share the same Rosetta Chrome endpoint:
`http://127.0.0.1:9223`.

Multiple agents may connect to the same DevTools port, but they must not share simultaneous page control.
Chrome `9223` is only the DevTools entrypoint; the pages below it are separate targets with their own
`targetId` and `webSocketDebuggerUrl`.

Prefer one Chrome port and one profile per agent whenever possible. Shared `9223` mode is a fallback coordination mode.

## Shared Chrome

- Start only one shared Rosetta Chrome for this mode:
  ```powershell
  chrome.exe --remote-debugging-port=9223 --user-data-dir=C:\path\to\shared-profile
  ```
- Do not start a second Chrome process on port `9223`.
- Do not clear the shared profile or kill shared Chrome unless holding the maintenance lock described below.

## Lock

- Shared port `9223` is a serialized resource.
- Only one agent may operate pages through `9223` at a time.
- Before any page, tab, target, cookie, storage, navigation, input, screenshot, or DevTools operation, acquire:
  `.codex/chrome-9223.lock`
- Acquire the lock atomically. In PowerShell, create the file with `System.IO.FileMode.CreateNew`; do not overwrite an existing lock.
- The lock file must record:
  `owner`, `pid`, `startedAt`, `expiresAt`, `port=9223`, `targetId`, `pageId`, `webSocketDebuggerUrl`, `url`, and intended task.
- Keep the lock alive with a heartbeat by extending `expiresAt` during long operations.
- Default TTL is 5 minutes unless a shorter task-specific TTL is known.

Example lock content:

```json
{
  "owner": "agent-a",
  "pid": 12345,
  "port": 9223,
  "targetId": "ABCDEF...",
  "pageId": "ABCDEF...",
  "webSocketDebuggerUrl": "ws://127.0.0.1:9223/devtools/page/ABCDEF...",
  "url": "https://chatgpt.com/...",
  "task": "oracle-review",
  "startedAt": "2026-07-05T12:00:00+08:00",
  "expiresAt": "2026-07-05T12:05:00+08:00"
}
```

## Target Binding

- After acquiring the lock, query `http://127.0.0.1:9223/json/list`.
- Create or select exactly one target/page for the agent.
- Record `targetId`, `pageId`, `webSocketDebuggerUrl`, and current `url` in the lock before acting.
- All browser operations must address the recorded `targetId` or `pageId` explicitly.
- Do not operate by active tab, tab index, or newest page guess.
- Do not temporarily switch to another agent's target while holding the lock.

## Release

- Release the lock immediately after the browser operation finishes.
- Use a `finally`/cleanup path so failures do not leave the lock behind.
- If a lock exists, other agents must wait or report contention.
- Other agents must not steal focus, close tabs, clear cookies, kill Chrome, or delete the profile while the lock is held.

## Stale Lock

- A lock is stale only when `expiresAt` has passed and the heartbeat has not been updated.
- Before removing a stale lock, verify both:
  the recorded `pid` no longer exists, and the recorded `targetId` is gone from `/json/list`.
- Do not remove a stale-looking lock by timestamp alone if the owner process and target are still alive.

## Maintenance

- Maintenance actions such as killing Chrome, clearing cache/cookies/history, or deleting `.rosetta/chrome-profile-*` require the same lock in exclusive maintenance mode.
- If the DevTools endpoint is unreachable, or the recorded target disappears, stop browser work, release only your own lock, and report the failure.

## Stability

- Shared `9223` mode is less stable than isolated ports/profiles.
- Do not use shared `9223` mode for long autonomous multi-agent runs unless a higher-level scheduler enforces these rules.
