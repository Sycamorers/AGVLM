# Reward Design

## Default RL objective

The repo uses `trl.GRPOTrainer` with modular reward functions.

The current objective is still rule-based / verifier-based. There is no
pretrained learned reward model yet, and the default GRPO path does not require
expert preference data.

Default reward modules:
- `exact_match`
- `normalized_label`
- `synonym_match`
- `structured_format`
- `uncertainty_calibration`
- `clarify_vs_respond`
- `management_coverage`
- `hallucination_penalty`

## Module intent

`exact_match`:
- short-answer VQA
- MCQ answers
- deterministic answer slots

`normalized_label`:
- disease labels
- pest labels
- crop-condition labels after normalization

`synonym_match`:
- alternate disease or pest names
- dataset-specific label variants

`structured_format`:
- consultation outputs with required sections
- management-oriented answers that must remain parseable
- requires meaningful content under each required section
- penalizes repeated boilerplate headings and empty sections

`uncertainty_calibration`:
- rewards explicit uncertainty when evidence is incomplete
- discourages unjustified certainty in ambiguous cases
- does not treat `high confidence` as uncertainty
- requires uncertainty statements to be linked to image ambiguity, missing evidence, field context, or clarification needs

`clarify_vs_respond`:
- rewards the correct high-level action choice
- especially important for MIRAGE-MMMT-style tasks

`management_coverage`:
- rewards coverage of expected management actions
- useful for semi-structured consultation tasks
- uses unique keyword matches and caps the score
- requires management keywords to appear in meaningful answer context
- applies a mild penalty to keyword stuffing, excessive repetition, and very long completions

`hallucination_penalty`:
- penalizes forbidden claims
- penalizes overconfident phrasing when uncertainty is required
- treats `high confidence` as overconfidence
- penalizes contradictions with accepted labels, unsupported definitive answers when clarification is expected, unsupported chemical/dosage/safety claims, fabricated visual evidence, unsafe recommendations, and crop/disease mismatches when metadata is available

`preference_proxy`:
- optional scaffold for future expert pairwise data
- rewards exact matches to `preference.chosen_response` and penalizes exact matches to `preference.rejected_response`
- not enabled by default and not a learned reward model

## Composite strategy

The composite reward:
- builds a normalized `RewardInput`
- applies configured modules
- multiplies by per-module weights
- sums the result into one scalar reward

This design keeps V1 deterministic by default and isolates any future judge-model reward behind an optional config path rather than making it mandatory.

## Diagnostics

Use reward-only scoring before GRPO:

```bash
PYTHONPATH=src python3 scripts/score_rl_manifest.py \
  --manifest data/manifests/full/rl_manifest.jsonl \
  --output reports/rl_reward_report.jsonl \
  --summary-output reports/rl_reward_summary.json \
  --max-samples 200
```

The summary includes per-module nonzero/positive/negative counts, total reward
statistics, zero and negative total reward counts, hallucination penalty counts,
uncertainty reward counts, and a small histogram. During GRPO smoke runs, set
`AGRI_VLM_REWARD_LOG_EVERY=1` and `AGRI_VLM_REWARD_DIAGNOSTICS_JSONL=...` to
emit batch-level reward diagnostics.

## Preference Pathway

Future expert preference data is represented as an optional top-level
`preference` object in each manifest row. See
`docs/preference_reward_data.md` for the format and export command. These fields
are backward-compatible and are ignored by the default rule-based reward unless
the optional `preference_proxy` module is explicitly enabled.
