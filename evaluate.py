#!/usr/bin/env python3
"""Main entry point: run the stages of an experiment config.

  python evaluate.py --config configs/quickstart.yaml                 # every stage in the config
  python evaluate.py --config configs/main.yaml --stages commute      # selected stages (comma-separated)
  python evaluate.py --config configs/main.yaml --dry-run             # print the commands, run nothing
  python evaluate.py --config configs/main.yaml --list                # list the stages

A config names a study and lists its stages in order. Each stage runs one experiment module
(`module`) once per entry of `runs` (or once with `args`). Arguments map to the module's
command-line flags: `llm_impact: x.csv` becomes `--llm-impact x.csv`, a list becomes several
values, `true` becomes a bare flag, and `positional` gives positional arguments.
Settings shared by several stages are written once with YAML anchors (`&name`, `<<: *name`).
Outputs go to results/.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def to_argv(args: dict) -> list[str]:
    """Command-line arguments for one run: positional values first, then --flags."""
    args = dict(args or {})
    argv = [str(v) for v in args.pop("positional", [])]
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


def commands(stage: dict) -> list[list[str]]:
    """One command per run of a stage; `args` applies to every run, `runs` adds to it."""
    base = stage.get("args") or {}
    runs = stage.get("runs") or [{}]
    return [[sys.executable, "-m", stage["module"], *to_argv({**base, **run})] for run in runs]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stages", default=None, help="comma-separated stage names (default: all)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--list", action="store_true", help="list the config's stages and exit")
    args = p.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stages = config["stages"]
    if args.list:
        print(f"{config['name']}: {config.get('description', '').strip()}\n")
        for name, stage in stages.items():
            print(f"  {name:<22} {stage.get('description', '').strip()}")
        return 0
    selected = args.stages.split(",") if args.stages else list(stages)
    unknown = [s for s in selected if s not in stages]
    if unknown:
        p.error(f"unknown stage(s) {unknown}; available: {list(stages)}")

    for name in selected:
        for cmd in commands(stages[name]):
            print(f"[{config['name']}:{name}] " + shlex.join(["python", *cmd[1:]]), flush=True)
            if not args.dry_run:
                done = subprocess.run(cmd, cwd=ROOT)
                if done.returncode != 0:
                    print(f"stage {name!r} failed (exit code {done.returncode})", file=sys.stderr)
                    return done.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
