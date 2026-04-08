#!/usr/bin/env python3
"""Launch a script through torchrun with sensible single-node defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import List


def detect_nproc_per_node() -> int:
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    if count <= 0:
        raise RuntimeError(
            "Unable to detect any visible CUDA devices. Set --nproc-per-node explicitly or expose GPUs."
        )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    nproc_per_node = args.nproc_per_node or detect_nproc_per_node()
    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        str(args.nnodes),
        "--node_rank",
        str(args.node_rank),
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        str(args.master_port),
        "--max_restarts",
        str(args.max_restarts),
        args.script,
    ]
    command.extend(script_args)
    return command


def main() -> int:
    args = parse_args()
    command = build_command(args)
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
