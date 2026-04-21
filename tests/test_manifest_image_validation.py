import json
import os
import subprocess
import sys
from pathlib import Path

from agri_vlm.utils.image import save_solid_image
from agri_vlm.utils.io import write_jsonl


def _row(sample_id: str, image_path: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_dataset": "agbase",
        "task_type": "consultation",
        "split": "train",
        "images": [image_path],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an agricultural assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Diagnose the issue."},
                ],
            },
        ],
        "target": {"answer_text": "leaf spot", "canonical_label": "leaf spot"},
        "metadata": {"source_image_id": image_path},
        "verifier": {"mode": "structured"},
        "reward_meta": {"weights": {"management_coverage": 1.0}},
    }


def test_manifest_image_validation_reports_invalid_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "repo"
    valid_image = data_root / "data/raw/valid.png"
    invalid_image = data_root / "data/raw/not_an_image.jpg"
    save_solid_image(valid_image, [40, 120, 70])
    invalid_image.parent.mkdir(parents=True, exist_ok=True)
    invalid_image.write_text("%PDF-1.7\nnot an image\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.jsonl"
    valid_output = tmp_path / "valid.jsonl"
    invalid_output = tmp_path / "invalid.jsonl"
    summary_output = tmp_path / "summary.json"
    write_jsonl(
        manifest_path,
        [
            _row("valid", "data/raw/valid.png"),
            _row("invalid", "data/raw/not_an_image.jpg"),
        ],
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/data/validate_manifest_images.py",
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(data_root),
            "--valid-output",
            str(valid_output),
            "--invalid-output",
            str(invalid_output),
            "--summary-output",
            str(summary_output),
            "--allow-invalid-with-report",
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert len(valid_output.read_text(encoding="utf-8").splitlines()) == 1
    invalid_rows = [json.loads(line) for line in invalid_output.read_text(encoding="utf-8").splitlines()]
    assert invalid_rows[0]["sample_id"] == "invalid"
    assert "UnidentifiedImageError" in invalid_rows[0]["image_errors"][0]["error"]
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["invalid_rows"] == 1
