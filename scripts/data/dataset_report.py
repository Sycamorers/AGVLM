#!/usr/bin/env python3
"""Print a compact manifest report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agri_vlm.data.manifest_io import read_manifest, summarize_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to a normalized or merged JSONL manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    rows = read_manifest(manifest_path)
    print(json.dumps(summarize_manifest(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
