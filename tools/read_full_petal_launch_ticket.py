#!/usr/bin/env python3
"""Read the immutable runtime fields consumed by the Full PETAL Slurm launcher."""

import argparse
import json
from pathlib import Path
import sys


FIELDS = (
    "mode",
    "entrypoint",
    "seed",
    "run_id",
    "deterministic",
    "not_eval",
    "resume_checkpoint",
    "work_dir",
)


class TicketReadError(ValueError):
    pass


def _plain_scalar(value, label):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value and not any(char in value for char in "\x00\r\n"):
        return value
    raise TicketReadError(f"ticket field {label} is not a safe scalar")


def read_launcher_fields(path):
    ticket_path = Path(path).expanduser().resolve(strict=True)
    try:
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TicketReadError(f"cannot read launch ticket: {exc}") from exc
    if not isinstance(ticket, dict):
        raise TicketReadError("launch ticket must be a JSON object")
    runtime = ticket.get("runtime_identity")
    if not isinstance(runtime, dict):
        raise TicketReadError("launch ticket lacks runtime_identity")
    overrides = runtime.get("cfg_overrides")
    if not isinstance(overrides, dict) or set(overrides) != {"work_dir"}:
        raise TicketReadError("runtime cfg_overrides must contain only work_dir")
    work_dir = overrides["work_dir"]
    if not isinstance(work_dir, str) or not work_dir:
        raise TicketReadError("runtime work_dir must be a non-empty string")
    expected_work_dir = ticket_path.parent / "work"
    if Path(work_dir).expanduser().resolve() != expected_work_dir:
        raise TicketReadError(
            "runtime work_dir must equal <ticket artifact root>/work"
        )
    values = {
        "mode": ticket.get("mode"),
        "entrypoint": runtime.get("entrypoint"),
        "seed": runtime.get("seed"),
        "run_id": runtime.get("run_id"),
        "deterministic": runtime.get("deterministic"),
        "not_eval": runtime.get("not_eval"),
        "resume_checkpoint": runtime.get("resume_checkpoint"),
        "work_dir": str(expected_work_dir),
    }
    if values["resume_checkpoint"] is not None:
        raise TicketReadError("this launcher does not accept resume checkpoints")
    return {name: _plain_scalar(values[name], name) for name in FIELDS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticket", type=Path)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        fields = read_launcher_fields(args.ticket)
    except TicketReadError as exc:
        parser.error(str(exc))
    output = sys.stdout.buffer
    for name in FIELDS:
        output.write(name.encode("ascii") + b"\0")
        output.write(fields[name].encode("utf-8") + b"\0")
    output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
