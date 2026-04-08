import os
import subprocess
import sys
from pathlib import Path

from agri_vlm.utils.distributed import get_distributed_context


def test_distributed_context_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    context = get_distributed_context(set_device=False)
    assert context.rank == 3
    assert context.local_rank == 1
    assert context.world_size == 8
    assert context.is_distributed is True


def test_launch_torchrun_dry_run_builds_expected_command() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/launch_torchrun.py",
            "--dry-run",
            "--nproc-per-node",
            "4",
            "--master-port",
            "29601",
            "scripts/verify_environment.py",
        ],
        cwd=repo_root,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "torch.distributed.run" in completed.stdout
    assert "--nproc_per_node 4" in completed.stdout
    assert "scripts/verify_environment.py" in completed.stdout
