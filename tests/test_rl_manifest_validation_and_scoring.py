import json
import os
import copy
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = REPO_ROOT / "tests/fixtures/rl/valid_rl_manifest.jsonl"


def _env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing else "src:%s" % existing
    return env


def _read_fixture_rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_validate_rl_manifest_accepts_fixture(tmp_path: Path) -> None:
    output_json = tmp_path / "validation.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_rl_manifest.py",
            "--manifest",
            str(FIXTURE_MANIFEST),
            "--output-json",
            str(output_json),
            "--check-image-open",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["row_count"] == 2
    assert report["issue_count"] == 0


def test_validate_rl_manifest_reports_invalid_cases(tmp_path: Path) -> None:
    rows = _read_fixture_rows()
    bad = copy.deepcopy(rows[0])
    bad["sample_id"] = "bad-label"
    bad["images"] = ["tests/fixtures/rl/images/missing.ppm"]
    bad["target"]["canonical_label"] = ""
    bad["verifier"]["accepted_labels"] = [""]
    bad["reward_meta"]["weights"]["management_coverage"] = 1.0
    duplicate = copy.deepcopy(rows[0])
    duplicate["sample_id"] = "bad-label"
    duplicate["split"] = "validation"
    manifest_path = tmp_path / "bad.jsonl"
    _write_jsonl(manifest_path, [bad, duplicate])
    output_json = tmp_path / "validation_bad.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_rl_manifest.py",
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads(output_json.read_text(encoding="utf-8"))
    issues = report["issues"]
    assert issues["image_path_missing"]["count"] == 1
    assert issues["empty_or_invalid_classification_label"]["count"] >= 1
    assert issues["invalid_accepted_labels"]["count"] >= 1
    assert issues["management_keywords_missing_when_enabled"]["count"] >= 1
    assert issues["duplicate_sample_id"]["count"] == 1
    assert issues["duplicate_sample_across_splits"]["count"] == 1


def test_score_rl_manifest_on_fixture(tmp_path: Path) -> None:
    output_path = tmp_path / "reward_report.jsonl"
    summary_path = tmp_path / "reward_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_rl_manifest.py",
            "--manifest",
            str(FIXTURE_MANIFEST),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert all("by_module" in row for row in rows)
    assert summary["scored_rows"] == 2
    assert summary["total_reward"]["negative_count"] == 0
    assert "structured_format" in summary["module_rewards"]


def test_prepare_pairwise_preference_data_fixture(tmp_path: Path) -> None:
    output_path = tmp_path / "pairs.jsonl"
    summary_path = tmp_path / "pairs_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/prepare_pairwise_preference_data.py",
            "--manifest",
            str(FIXTURE_MANIFEST),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    pairs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(pairs) == 1
    assert pairs[0]["chosen"]
    assert pairs[0]["rejected"]
    assert summary["trains_reward_model"] is False
