import json
import os
import subprocess
import sys
from pathlib import Path


def run(repo_root: Path, env: dict, *args: str) -> None:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr


def test_data_prep_pipeline_generates_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["AGRI_VLM_DATA_ROOT"] = str(tmp_path / "data")

    run(
        repo_root,
        env,
        "scripts/data/prepare_manual_dataset_slots.py",
        "--with-smoke-data",
        "--download-mode",
        "partial",
        "--fraction",
        "0.1",
    )
    run(
        repo_root,
        env,
        "scripts/data/normalize_all.py",
        "--download-mode",
        "partial",
        "--fraction",
        "0.1",
    )
    run(
        repo_root,
        env,
        "scripts/data/build_sft_manifest.py",
        "--download-mode",
        "partial",
        "--fraction",
        "0.1",
    )
    run(
        repo_root,
        env,
        "scripts/data/build_rl_manifest.py",
        "--download-mode",
        "partial",
        "--fraction",
        "0.1",
    )
    run(
        repo_root,
        env,
        "scripts/data/build_eval_manifest.py",
        "--download-mode",
        "partial",
        "--fraction",
        "0.1",
    )
    run(
        repo_root,
        env,
        "scripts/data/dataset_report.py",
        "--download-mode",
        "partial",
        "--fraction",
        "0.1",
    )

    report_path = tmp_path / "data" / "manifests" / "partial_10pct" / "dataset_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["subset_tag"] == "partial_10pct"
    assert report["datasets"]["agrobench"]["status"] == "normalized"
    assert report["datasets"]["ip102"]["normalized_rows"] >= 1
