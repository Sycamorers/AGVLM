"""Training callbacks."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - exercised only when transformers is unavailable
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - fallback for dry-run imports
    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base class when transformers is unavailable."""

        pass


class JsonlMetricsCallback(TrainerCallback):
    """Persist trainer logs as JSONL."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        if not logs:
            return
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        payload = dict(logs)
        payload["global_step"] = state.global_step
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
