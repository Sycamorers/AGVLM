# Reward Design

## Default RL objective

The repo uses `trl.GRPOTrainer` with modular reward functions.

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

`uncertainty_calibration`:
- rewards explicit uncertainty when evidence is incomplete
- discourages unjustified certainty in ambiguous cases

`clarify_vs_respond`:
- rewards the correct high-level action choice
- especially important for MIRAGE-MMMT-style tasks

`management_coverage`:
- rewards coverage of expected management actions
- useful for semi-structured consultation tasks

`hallucination_penalty`:
- penalizes forbidden claims
- penalizes overconfident phrasing when uncertainty is required

## Composite strategy

The composite reward:
- builds a normalized `RewardInput`
- applies configured modules
- multiplies by per-module weights
- sums the result into one scalar reward

This design keeps V1 deterministic by default and isolates any future judge-model reward behind an optional config path rather than making it mandatory.
