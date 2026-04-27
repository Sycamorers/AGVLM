#!/usr/bin/env python3
"""Report benchmark preparation and evaluation readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.paths import build_template_context, compute_subset_tag, get_data_root
from agri_vlm.utils.io import ensure_dir, load_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/benchmarks.yaml")
    parser.add_argument("--download-mode", choices=["partial", "full"], default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--subset-tag", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-json", default="outputs/artifacts/reports/benchmark_status.json")
    parser.add_argument("--output-markdown", default="outputs/artifacts/reports/benchmark_status.md")
    return parser.parse_args()


def _resolve_path(template: str, context: Dict[str, str]) -> Path:
    path = Path(str(template).format(**context)).expanduser()
    if not path.is_absolute():
        path = Path(context["repo_root"]) / path
    return path.resolve()


def _path_status(path_spec: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
    path = _resolve_path(path_spec["path"], context)
    return {
        "path": str(path),
        "kind": path_spec.get("kind", "path"),
        "required": bool(path_spec.get("required", True)),
        "exists": path.exists(),
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    payload = load_yaml(repo_root / args.config)
    defaults = payload.get("defaults", {})
    download_mode = args.download_mode or defaults.get("download_mode", "full")
    sample_fraction = float(args.fraction if args.fraction is not None else defaults.get("sample_fraction", 1.0))
    subset_tag = args.subset_tag or compute_subset_tag(download_mode, sample_fraction)
    data_root = get_data_root(repo_root, args.data_root)
    context = build_template_context(
        repo_root=repo_root,
        data_root=data_root,
        subset_tag=subset_tag,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
    )

    benchmarks: List[Dict[str, Any]] = []
    for benchmark in payload.get("benchmarks", []):
        readiness = [_path_status(path_spec, context) for path_spec in benchmark.get("readiness_paths", [])]
        missing_required = [item for item in readiness if item["required"] and not item["exists"]]
        implementation_status = benchmark.get("implementation_status", "planned")
        ready_for_eval = implementation_status == "implemented" and not missing_required
        status = "ready" if ready_for_eval else "missing_data" if implementation_status == "implemented" else "planned_or_blocked"
        benchmarks.append(
            {
                "name": benchmark["name"],
                "role": benchmark.get("role", ""),
                "paper_use": benchmark.get("paper_use", ""),
                "implementation_status": implementation_status,
                "readiness_status": status,
                "ready_for_eval": ready_for_eval,
                "missing_required_paths": missing_required,
                "readiness_paths": readiness,
                "access": benchmark.get("access", ""),
                "preparation_mode": benchmark.get("preparation_mode", ""),
                "licensing_notes": benchmark.get("licensing_notes", ""),
                "prepare_commands": benchmark.get("prepare_commands", []),
                "eval_commands": benchmark.get("eval_commands", []),
            }
        )

    return {
        "subset_tag": subset_tag,
        "download_mode": download_mode,
        "sample_fraction": sample_fraction,
        "data_root": str(data_root),
        "benchmarks": benchmarks,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Benchmark Status",
        "",
        "Subset tag: `%s`" % report["subset_tag"],
        "Download mode: `%s`" % report["download_mode"],
        "Data root: `%s`" % report["data_root"],
        "",
        "| Benchmark | Role | Implementation | Readiness | Missing required paths |",
        "| --- | --- | --- | --- | --- |",
    ]
    for benchmark in report["benchmarks"]:
        missing = "<br>".join(item["path"] for item in benchmark["missing_required_paths"]) or "none"
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                benchmark["name"],
                benchmark["role"],
                benchmark["implementation_status"],
                benchmark["readiness_status"],
                missing,
            )
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for benchmark in report["benchmarks"]:
        lines.append("### %s" % benchmark["name"])
        lines.append("")
        lines.append("Preparation:")
        for command in benchmark["prepare_commands"]:
            lines.append("- `%s`" % command)
        lines.append("")
        lines.append("Evaluation:")
        for command in benchmark["eval_commands"]:
            lines.append("- `%s`" % command)
        lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(args)
    write_json(Path(args.output_json), report)
    write_markdown(report, Path(args.output_markdown))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
