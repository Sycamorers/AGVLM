"""Composite reward routing."""

import json
from typing import Any, Callable, Dict, Iterable, List

from agri_vlm.rewards.clarify_decision import clarify_vs_respond_reward
from agri_vlm.rewards.classification import normalized_label_reward
from agri_vlm.rewards.exact_match import exact_match_reward
from agri_vlm.rewards.hallucination_penalty import hallucination_penalty
from agri_vlm.rewards.management_coverage import management_coverage_reward
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
}


def build_reward_input(
    prediction: str,
    task_type: str,
    target_json: str,
    verifier_json: str,
    reward_meta_json: str,
) -> RewardInput:
    target = json.loads(target_json)
    verifier = json.loads(verifier_json)
    reward_meta = json.loads(reward_meta_json)
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
        synonym_groups=verifier.get("synonyms") or {},
        uncertainty_required=bool(verifier.get("uncertainty_required")),
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

    def reward_fn(
        prompts: List[str],
        completions: List[Any],
        task_type: List[str],
        target_json: List[str],
        verifier_json: List[str],
        reward_meta_json: List[str],
        **kwargs: Any
    ) -> List[float]:
        rewards: List[float] = []
        for index in range(len(completions)):
            reward_input = build_reward_input(
                prediction=_completion_to_text(completions[index]),
                task_type=task_type[index],
                target_json=target_json[index],
                verifier_json=verifier_json[index],
                reward_meta_json=reward_meta_json[index],
            )
            rewards.append(
                compute_composite_reward(
                    reward_input=reward_input,
                    reward_modules=reward_modules,
                    reward_weights=reward_weights,
                ).total
            )
        return rewards

    return reward_fn
