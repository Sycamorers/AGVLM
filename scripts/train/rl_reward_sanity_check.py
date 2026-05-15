#!/usr/bin/env python3
"""Score synthetic candidate completions with the configured RL rewards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Tuple

from agri_vlm.rewards.composite import make_trl_reward_function
from agri_vlm.rewards.parsing import extract_structured_sections
from agri_vlm.schemas.config_schema import RLTrainConfigSchema, load_config
from agri_vlm.utils.io import ensure_dir, read_jsonl, write_json


DEFAULT_REWARD_MODULES = [
    "exact_match",
    "normalized_label",
    "synonym_match",
    "structured_format",
    "uncertainty_calibration",
    "clarify_vs_respond",
    "management_coverage",
    "hallucination_penalty",
]
DEFAULT_REWARD_WEIGHTS = {
    "exact_match": 1.0,
    "normalized_label": 1.0,
    "synonym_match": 0.5,
    "structured_format": 0.5,
    "uncertainty_calibration": 0.5,
    "clarify_vs_respond": 1.0,
    "management_coverage": 0.5,
    "hallucination_penalty": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--config", default=None)
    parser.add_argument("--reward-modules", default=None)
    parser.add_argument("--reward-weights-json", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _sample_rows(path: Path, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path), start=1):
        if max_samples <= 0:
            reservoir.append(row)
            continue
        if len(reservoir) < max_samples:
            reservoir.append(row)
            continue
        replacement = rng.randint(1, index)
        if replacement <= max_samples:
            reservoir[replacement - 1] = row
    return reservoir


def _load_reward_settings(
    *,
    repo_root: Path,
    config_path: str | None,
    reward_modules_override: str | None,
    reward_weights_override: str | None,
) -> Tuple[List[str], Dict[str, float]]:
    reward_modules = list(DEFAULT_REWARD_MODULES)
    reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
    if config_path:
        config = load_config(repo_root / config_path, RLTrainConfigSchema)
        reward_modules = list(config.reward_modules)
        reward_weights = dict(config.reward_weights)
    if reward_modules_override:
        reward_modules = [item.strip() for item in reward_modules_override.split(",") if item.strip()]
    if reward_weights_override:
        weights_path = Path(reward_weights_override)
        if weights_path.exists():
            payload = json.loads(weights_path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(reward_weights_override)
        reward_weights = {str(key): float(value) for key, value in payload.items()}
    return reward_modules, reward_weights


def _target_answer(row: Dict[str, Any]) -> str:
    target = row.get("target") or {}
    if target.get("answer_text"):
        return "Answer: %s" % str(target["answer_text"])
    if target.get("canonical_label"):
        return "Answer: %s" % str(target["canonical_label"])
    if target.get("canonical_labels"):
        return "Answer: %s" % str(target["canonical_labels"][0])
    if target.get("acceptable_answers"):
        return "Answer: %s" % str(target["acceptable_answers"][0])
    if target.get("decision"):
        return "Decision: %s\nClarifying question: Which crop and field conditions should I consider?" % target["decision"]
    if target.get("structured"):
        return json.dumps(target["structured"], ensure_ascii=False, sort_keys=True)
    return ""


def _structured_consultation(row: Dict[str, Any]) -> str:
    target = row.get("target") or {}
    verifier = row.get("verifier") or {}
    label = target.get("canonical_label") or target.get("answer_text") or "uncertain agricultural issue"
    management_keywords = verifier.get("management_keywords") or []
    management = "; ".join(str(item) for item in management_keywords[:4]) or "monitor the plant and use locally appropriate management."
    return "\n".join(
        [
            "Diagnosis: %s" % label,
            "Evidence: Visible symptoms should be compared with crop context before acting.",
            "Uncertainty: Moderate confidence; confirm with clearer images or local extension guidance if needed.",
            "Management: %s" % management,
            "Follow-up: Recheck new growth and document whether symptoms spread.",
        ]
    )


def _known_good(row: Dict[str, Any]) -> str:
    task_type = str(row.get("task_type") or "")
    verifier = row.get("verifier") or {}
    target = row.get("target") or {}
    expected_decision = target.get("decision") or verifier.get("expected_decision")
    if task_type == "consultation" or verifier.get("required_sections"):
        return _structured_consultation(row)
    if task_type == "clarify_or_respond" and expected_decision == "clarify":
        return "Decision: clarify\nClarifying question: Which crop, plant part, and symptom timeline should I consider?"
    if task_type == "clarify_or_respond" and expected_decision == "respond":
        answer = target.get("answer_text") or (target.get("acceptable_answers") or ["monitor symptoms"])[0]
        return "Decision: respond\nAnswer: %s" % answer
    return _target_answer(row)


def _known_bad(row: Dict[str, Any]) -> str:
    task_type = str(row.get("task_type") or "")
    verifier = row.get("verifier") or {}
    target = row.get("target") or {}
    expected_decision = target.get("decision") or verifier.get("expected_decision")
    if task_type == "consultation" or verifier.get("required_sections"):
        return "This is definitely fine. No follow-up needed. Apply pesticide immediately."
    if task_type == "clarify_or_respond" and expected_decision == "clarify":
        return "Decision: respond\nAnswer: This is definitely a disease. Apply pesticide immediately."
    if task_type == "clarify_or_respond" and expected_decision == "respond":
        return "Decision: clarify\nClarifying question: Can you upload another image?"
    return "Answer: unrelated wrong answer"


def candidate_predictions(row: Dict[str, Any]) -> Dict[str, str]:
    good = _known_good(row)
    bad = _known_bad(row)
    return {
        "empty": "",
        "target_answer": good,
        "known_good": good,
        "known_bad": bad,
        "generic_uncertain": "I am uncertain from the available image evidence and need clearer crop context before making a firm diagnosis.",
        "generic_clarify": "Could you provide a clearer close-up image and the crop name before I answer?",
        "generic_overconfident": "Answer: early blight\nThis is definitely the only possible cause and a guaranteed cure is available. No follow-up needed.",
        "structured_consultation": _structured_consultation(row),
    }


def _distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "min": float(min(ordered)),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": float(max(ordered)),
    }


def _average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def _row_reward(
    reward_fn: Any,
    row: Dict[str, Any],
    completion: str,
) -> float:
    rewards = reward_fn(
        prompts=[row.get("messages") or ""],
        completions=[completion],
        task_type=[str(row.get("task_type") or "")],
        target_json=[json.dumps(row.get("target") or {}, ensure_ascii=False)],
        verifier_json=[json.dumps(row.get("verifier") or {}, ensure_ascii=False)],
        reward_meta_json=[json.dumps(row.get("reward_meta") or {}, ensure_ascii=False)],
        metadata_json=[json.dumps(row.get("metadata") or {}, ensure_ascii=False)],
        preference_json=[json.dumps(row.get("preference") or {}, ensure_ascii=False)],
    )
    return float(rewards[0])


def _add_example(
    examples: Dict[str, List[Dict[str, Any]]],
    key: str,
    row: Dict[str, Any],
    scores: Dict[str, float],
    max_examples: int = 20,
) -> None:
    if len(examples[key]) >= max_examples:
        return
    examples[key].append(
        {
            "sample_id": row.get("sample_id"),
            "task_type": row.get("task_type"),
            "verifier_mode": (row.get("verifier") or {}).get("mode"),
            "scores": scores,
        }
    )


def run_sanity_check(
    *,
    manifest_path: Path,
    reward_modules: List[str],
    reward_weights: Dict[str, float],
    max_samples: int,
    seed: int,
) -> Dict[str, Any]:
    rows = _sample_rows(manifest_path, max_samples=max_samples, seed=seed)
    reward_fn = make_trl_reward_function(reward_modules=reward_modules, reward_weights=reward_weights)
    by_candidate: Dict[str, List[float]] = defaultdict(list)
    by_task_type: Dict[str, List[float]] = defaultdict(list)
    by_verifier_mode: Dict[str, List[float]] = defaultdict(list)
    by_task_type_candidate: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_scores: List[float] = []
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    assertion_counts: Counter[str] = Counter()
    assertion_failures: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_failure(name: str, row: Dict[str, Any], scores: Dict[str, float], reason: str) -> None:
        assertion_counts[name] += 1
        if len(assertion_failures[name]) >= 20:
            return
        assertion_failures[name].append(
            {
                "sample_id": row.get("sample_id"),
                "task_type": row.get("task_type"),
                "verifier_mode": (row.get("verifier") or {}).get("mode"),
                "reason": reason,
                "scores": scores,
            }
        )

    for row in rows:
        predictions = candidate_predictions(row)
        scores = {name: _row_reward(reward_fn, row, text) for name, text in predictions.items()}
        task_type = str(row.get("task_type") or "")
        verifier = row.get("verifier") or {}
        verifier_mode = str(verifier.get("mode") or "")
        for candidate_name, score in scores.items():
            by_candidate[candidate_name].append(score)
            by_task_type[task_type].append(score)
            by_verifier_mode[verifier_mode].append(score)
            by_task_type_candidate[task_type][candidate_name].append(score)
            all_scores.append(score)

        target = row.get("target") or {}
        expected_decision = target.get("decision") or verifier.get("expected_decision")
        if scores["empty"] > 0.25:
            _add_example(examples, "empty_output_high_reward", row, scores)
        if scores["target_answer"] <= scores["empty"]:
            _add_example(examples, "target_answer_not_above_empty", row, scores)
            add_failure("good_not_above_empty", row, scores, "known-good target answer did not beat empty output")
        if scores["known_good"] <= scores["known_bad"]:
            add_failure("good_not_above_bad", row, scores, "known-good completion did not beat known-bad completion")
        if expected_decision == "respond" and scores["generic_clarify"] > scores["target_answer"]:
            _add_example(examples, "generic_clarify_beats_target_on_respond", row, scores)
            add_failure("clarify_beats_respond", row, scores, "generic clarification beat a respond target")
        if verifier.get("uncertainty_required") and scores["generic_overconfident"] >= 0.0:
            _add_example(examples, "overconfident_not_penalized_when_uncertainty_required", row, scores)
            add_failure("overconfident_not_penalized", row, scores, "unsafe overconfident answer was not negative")
        if (row.get("task_type") == "consultation" or verifier.get("required_sections")) and scores["structured_consultation"] == 0.0:
            _add_example(examples, "structured_consultation_reward_zero", row, scores)
            add_failure("structured_sections_not_rewarded", row, scores, "structured completion got zero reward")
        if row.get("task_type") == "consultation" and len(extract_structured_sections(predictions["structured_consultation"])) < 5:
            add_failure("structured_parser_failed", row, scores, "structured section parser found fewer than five sections")
        if all(value == 0.0 for value in scores.values()):
            _add_example(examples, "all_candidates_zero", row, scores)
            add_failure("all_candidates_zero", row, scores, "all synthetic completions scored zero")

    return {
        "manifest_path": str(manifest_path),
        "sampled_rows": len(rows),
        "reward_modules": reward_modules,
        "reward_weights": reward_weights,
        "average_reward_by_candidate": {
            name: _average(values) for name, values in sorted(by_candidate.items())
        },
        "average_reward_by_task_type": {
            name: _average(values) for name, values in sorted(by_task_type.items())
        },
        "average_reward_by_verifier_mode": {
            name: _average(values) for name, values in sorted(by_verifier_mode.items())
        },
        "average_reward_by_task_type_candidate": {
            task_type: {
                candidate: _average(values)
                for candidate, values in sorted(candidate_values.items())
            }
            for task_type, candidate_values in sorted(by_task_type_candidate.items())
        },
        "reward_distribution": _distribution(all_scores),
        "examples": dict(examples),
        "assertion_failures": dict(assertion_failures),
        "assertion_failure_count": int(sum(assertion_counts.values())),
    }


def write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# RL Reward Sanity Check",
        "",
        "- Manifest: `%s`" % report["manifest_path"],
        "- Sampled rows: `%s`" % report["sampled_rows"],
        "- Reward modules: `%s`" % ", ".join(report["reward_modules"]),
        "- Assertion failure count: `%s`" % report.get("assertion_failure_count", 0),
        "",
        "## Average Reward By Candidate",
        "",
        "| Candidate | Average Reward |",
        "| --- | ---: |",
    ]
    for name, value in report["average_reward_by_candidate"].items():
        lines.append("| %s | %.4f |" % (name, value))
    lines.extend(["", "## Average Reward By Task Type", "", "| Task Type | Average Reward |", "| --- | ---: |"])
    for name, value in report["average_reward_by_task_type"].items():
        lines.append("| %s | %.4f |" % (name, value))
    lines.extend(["", "## Average Reward By Verifier Mode", "", "| Verifier Mode | Average Reward |", "| --- | ---: |"])
    for name, value in report["average_reward_by_verifier_mode"].items():
        lines.append("| %s | %.4f |" % (name, value))
    lines.extend(["", "## Reward Distribution", "", "| Statistic | Reward |", "| --- | ---: |"])
    for name, value in report["reward_distribution"].items():
        lines.append("| %s | %.4f |" % (name, value))
    lines.extend(["", "## Examples", ""])
    for name, examples in sorted(report["examples"].items()):
        lines.append("### %s" % name)
        lines.append("")
    lines.extend(["## Assertion Failures", ""])
    if not report.get("assertion_failures"):
        lines.append("No assertion failures.")
    for name, examples in sorted(report.get("assertion_failures", {}).items()):
        lines.append("### %s" % name)
        lines.append("")
        lines.append("- Count shown: `%s`" % len(examples))
        for example in examples:
            lines.append(
                "- `%s` task=`%s` verifier=`%s` reason=`%s` scores=`%s`"
                % (
                    example.get("sample_id"),
                    example.get("task_type"),
                    example.get("verifier_mode"),
                    example.get("reason"),
                    json.dumps(example.get("scores"), sort_keys=True),
                )
            )
        lines.append("")
        lines.append("- Count shown: `%s`" % len(examples))
        for example in examples:
            lines.append(
                "- `%s` task=`%s` verifier=`%s` scores=`%s`"
                % (
                    example.get("sample_id"),
                    example.get("task_type"),
                    example.get("verifier_mode"),
                    json.dumps(example.get("scores"), sort_keys=True),
                )
            )
        lines.append("")
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    reward_modules, reward_weights = _load_reward_settings(
        repo_root=repo_root,
        config_path=args.config,
        reward_modules_override=args.reward_modules,
        reward_weights_override=args.reward_weights_json,
    )
    report = run_sanity_check(
        manifest_path=Path(args.manifest_path),
        reward_modules=reward_modules,
        reward_weights=reward_weights,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    write_json(Path(args.output_json), report)
    write_markdown_report(report, Path(args.output_md))
    print("rl_reward_sanity=%s sampled_rows=%s" % (args.output_json, report["sampled_rows"]))
    return 0 if report["sampled_rows"] and report.get("assertion_failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
