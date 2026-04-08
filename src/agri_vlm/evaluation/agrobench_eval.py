"""AgroBench-oriented evaluation."""

from typing import Any, Dict

from agri_vlm.evaluation.agmmu_eval import run_agmmu_eval


def run_agrobench_eval(model_config: Any, eval_config: Any) -> Dict[str, Any]:
    return run_agmmu_eval(model_config=model_config, eval_config=eval_config)
