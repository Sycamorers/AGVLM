#!/usr/bin/env python3
"""Verify local environment prerequisites for agri-vlm-v1."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = [
    "yaml",
    "pydantic",
    "PIL",
]

OPTIONAL_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
]


def try_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return "missing (%s)" % exc.__class__.__name__
    return getattr(module, "__version__", "ok")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print("repo_root=%s" % repo_root)
    print("python=%s" % sys.version.replace("\n", " "))
    print("platform=%s" % platform.platform())

    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ is required for the documented training stack.")
        return 1

    failures = []
    for package in REQUIRED_PACKAGES:
        version = try_import(package)
        print("required[%s]=%s" % (package, version))
        if version.startswith("missing"):
            failures.append(package)

    for package in OPTIONAL_PACKAGES:
        version = try_import(package)
        print("optional[%s]=%s" % (package, version))

    if failures:
        print("ERROR: missing required packages: %s" % ", ".join(failures))
        return 1

    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
