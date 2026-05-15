"""Composite reward routing."""

from collections import Counter
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from agri_vlm.rewards.clarify_decision import clarify_vs_respond_reward
from agri_vlm.rewards.classification import normalized_label_reward
from agri_vlm.rewards.exact_match import exact_match_reward
from agri_vlm.rewards.hallucination_penalty import hallucination_penalty
from agri_vlm.rewards.management_coverage import management_coverage_reward
from agri_vlm.rewards.preference import preference_proxy_reward
from agri_vlm.rewards.structure import structured_format_reward
from agri_vlm.rewards.synonym_match import synonym_match_reward
from agri_vlm.rewards.uncertainty import uncertainty_calibration_reward
from agri_vlm.schemas.reward_schema import RewardBreakdown, RewardInput


REWARD_REGISTRY = {
    "exact_match": exact_match_reward,
    "normalized_label": normalized_label_reward,
    "synonym_match": synonym_match_reward,
    "structured_format": structured_format_reward,
    "uncertainty_calibration": uncertainty_calibration_reward,
    "clarify_vs_respond": clarify_vs_respond_reward,
    "management_coverage": management_coverage_reward,
    "hallucination_penalty": hallucination_penalty,
    "preference_proxy": preference_proxy_reward,
}


def _loads_json_object(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Reward JSON payload must decode to an object.")
    return payload


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value)]
    return []


def _merged_list(*payloads: Dict[str, Any], key: str) -> List[str]:
    values: List[str] = []
    seen = set()
    for payload in payloads:
        for item in _as_list(payload.get(key)):
            if item not in seen:
                values.append(item)
                seen.add(item)
    return values


def _first_value(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _preference_payload_fields(preference: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = [
        "preference_score",
        "preference_rationale",
        "chosen_response",
        "rejected_response",
        "expert_quality_score",
        "agronomic_correctness_score",
        "management_usefulness_score",
        "uncertainty_calibration_score",
        "safety_score",
    ]
    return {key: preference.get(key) for key in allowed_keys if key in preference}


def build_reward_input(
    prediction: str,
    task_type: str,
    target_json: str,
    verifier_json: str,
    reward_meta_json: str,
    metadata_json: Optional[str] = None,
    preference_json: Optional[str] = None,
) -> RewardInput:
    target = _loads_json_object(target_json)
    verifier = _loads_json_object(verifier_json)
    reward_meta = _loads_json_object(reward_meta_json)
    metadata = _loads_json_object(metadata_json)
    preference = _loads_json_object(preference_json)
    expected_uncertainty = verifier.get("expected_uncertainty")
    if expected_uncertainty is None:
        expected_uncertainty = metadata.get("expected_uncertainty")
    return RewardInput(
        prediction=prediction,
        task_type=task_type,
        target_text=target.get("answer_text"),
        target_label=target.get("canonical_label"),
        target_labels=target.get("canonical_labels") or [],
        expected_decision=target.get("decision") or verifier.get("expected_decision"),
        required_sections=verifier.get("required_sections") or [],
        management_keywords=verifier.get("management_keywords") or [],
        forbidden_claims=verifier.get("forbidden_claims") or [],
        acceptable_answers=(target.get("acceptable_answers") or []) + (verifier.get("accepted_answers") or []),
        accepted_labels=verifier.get("accepted_labels") or [],
        synonym_groups=verifier.get("synonyms") or {},
        uncertainty_required=bool(verifier.get("uncertainty_required") or expected_uncertainty is True),
        expected_uncertainty=expected_uncertainty if isinstance(expected_uncertainty, bool) else None,
        crop=_first_value(verifier.get("crop"), metadata.get("crop")),
        disease=_first_value(verifier.get("disease"), metadata.get("disease")),
        known_facts=_merged_list(verifier, metadata, key="known_facts"),
        allowed_claims=_merged_list(verifier, metadata, key="allowed_claims"),
        visual_evidence=_merged_list(verifier, metadata, key="visual_evidence"),
        unsafe_recommendations=_merged_list(verifier, metadata, key="unsafe_recommendations"),
        **_preference_payload_fields(preference),
        weights=reward_meta.get("weights") or {},
    )


def compute_composite_reward(
    reward_input: RewardInput,
    reward_modules: Iterable[str],
    reward_weights: Dict[str, float],
) -> RewardBreakdown:
    by_module: Dict[str, float] = {}
    total = 0.0
    for module_name in reward_modules:
        reward_fn = REWARD_REGISTRY[module_name]
        raw_value = float(reward_fn(reward_input))
        weight = float(reward_weights.get(module_name, reward_input.weights.get(module_name, 1.0)))
        value = raw_value * weight
        by_module[module_name] = value
        total += value
    return RewardBreakdown(total=total, by_module=by_module, notes=[])


def reward_histogram(values: Iterable[float]) -> Dict[str, int]:
    """Bucket reward totals into stable diagnostic bins."""
    buckets = Counter()
    for value in values:
        if value < -1.0:
            buckets["lt_-1"] += 1
        elif value < 0.0:
            buckets["-1_to_0"] += 1
        elif value == 0.0:
            buckets["zero"] += 1
        elif value <= 1.0:
            buckets["0_to_1"] += 1
        elif value <= 2.0:
            buckets["1_to_2"] += 1
        else:
            buckets["gt_2"] += 1
    return {key: int(buckets.get(key, 0)) for key in ["lt_-1", "-1_to_0", "zero", "0_to_1", "1_to_2", "gt_2"]}


def summarize_reward_breakdowns(breakdowns: Iterable[RewardBreakdown]) -> Dict[str, Any]:
    """Aggregate per-sample reward diagnostics for logs and reward-only scoring."""
    rows = list(breakdowns)
    totals = [float(row.total) for row in rows]
    module_names = sorted({name for row in rows for name in row.by_module})
    module_summary: Dict[str, Dict[str, Any]] = {}
    for module_name in module_names:
        values = [float(row.by_module.get(module_name, 0.0)) for row in rows]
        nonzero = [value for value in values if value != 0.0]
        positive = [value for value in values if value > 0.0]
        negative = [value for value in values if value < 0.0]
        module_summary[module_name] = {
            "count": len(values),
            "nonzero_count": len(nonzero),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "sum": float(sum(values)),
            "mean": float(sum(values) / len(values)) if values else 0.0,
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }
    return {
        "sample_count": len(rows),
        "total_reward": {
            "sum": float(sum(totals)),
            "mean": float(sum(totals) / len(totals)) if totals else 0.0,
            "min": float(min(totals)) if totals else 0.0,
            "max": float(max(totals)) if totals else 0.0,
            "zero_count": sum(1 for value in totals if value == 0.0),
            "negative_count": sum(1 for value in totals if value < 0.0),
            "histogram": reward_histogram(totals),
        },
        "module_rewards": module_summary,
        "hallucination_penalty_count": sum(
            1 for row in rows if float(row.by_module.get("hallucination_penalty", 0.0)) < 0.0
        ),
        "uncertainty_reward_count": sum(
            1 for row in rows if float(row.by_module.get("uncertainty_calibration", 0.0)) > 0.0
        ),
    }


def _append_reward_diagnostics(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        content = completion[-1].get("content")
        if isinstance(content, str):
            return content
    return str(completion)


def make_trl_reward_function(
    reward_modules: List[str],
    reward_weights: Dict[str, float],
) -> Callable[..., List[float]]:
    """Build a TRL-compatible reward function."""
    logger = logging.getLogger("agri_vlm.rewards")
    call_count = 0
    log_every = int(os.environ.get("AGRI_VLM_REWARD_LOG_EVERY", "0") or "0")
    diagnostics_path = os.environ.get("AGRI_VLM_REWARD_DIAGNOSTICS_JSONL")

    def reward_fn(
        prompts: List[str],
        completions: List[Any],
        task_type: List[str],
        target_json: List[str],
        verifier_json: List[str],
        reward_meta_json: List[str],
        metadata_json: Optional[List[str]] = None,
        preference_json: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[float]:
        nonlocal call_count
        call_count += 1
        rewards: List[float] = []
        breakdowns: List[RewardBreakdown] = []
        diagnostic_rows: List[Dict[str, Any]] = []
        sample_ids = kwargs.get("sample_id") if isinstance(kwargs.get("sample_id"), list) else []
        for index in range(len(completions)):
            reward_input = build_reward_input(
                prediction=_completion_to_text(completions[index]),
                task_type=task_type[index],
                target_json=target_json[index],
                verifier_json=verifier_json[index],
                reward_meta_json=reward_meta_json[index],
                metadata_json=metadata_json[index] if metadata_json else None,
                preference_json=preference_json[index] if preference_json else None,
            )
            breakdown = compute_composite_reward(
                reward_input=reward_input,
                reward_modules=reward_modules,
                reward_weights=reward_weights,
            )
            if not math.isfinite(breakdown.total):
                raise FloatingPointError("Composite reward is non-finite for batch index %s." % index)
            breakdowns.append(breakdown)
            rewards.append(breakdown.total)
            diagnostic_rows.append(
                {
                    "call": call_count,
                    "batch_index": index,
                    "sample_id": str(sample_ids[index]) if index < len(sample_ids) else "",
                    "task_type": task_type[index],
                    "total": breakdown.total,
                    "by_module": breakdown.by_module,
                }
            )
        summary = summarize_reward_breakdowns(breakdowns)
        if diagnostics_path:
            _append_reward_diagnostics(Path(diagnostics_path), diagnostic_rows)
        if log_every > 0 and call_count % log_every == 0:
            logger.info(
                "reward_diagnostics call=%s samples=%s total_mean=%.4f zero=%s negative=%s "
                "hallucination_penalty=%s uncertainty_reward=%s module_nonzero=%s",
                call_count,
                summary["sample_count"],
                summary["total_reward"]["mean"],
                summary["total_reward"]["zero_count"],
                summary["total_reward"]["negative_count"],
                summary["hallucination_penalty_count"],
                summary["uncertainty_reward_count"],
                {
                    name: int(payload["nonzero_count"])
                    for name, payload in summary["module_rewards"].items()
                },
            )
        return rewards

    return reward_fn
