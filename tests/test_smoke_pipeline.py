import os
import subprocess
from pathlib import Path


def test_smoke_pipeline_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        ["bash", "scripts/run_smoke_test.sh"],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
