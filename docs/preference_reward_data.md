# Future Expert Preference Data

The current GRPO reward is rule-based / verifier-based. There is no pretrained
learned reward model in this repository yet. Preference data is optional and
backward-compatible so future reward-model work can be added without changing
the current default GRPO path.

## Manifest Format

Add an optional top-level `preference` object to any RL manifest row:

```json
{
  "sample_id": "example-1",
  "messages": [],
  "target": {},
  "verifier": {},
  "reward_meta": {},
  "preference": {
    "preference_score": 0.9,
    "preference_rationale": "Chosen answer is agronomically safer and better calibrated.",
    "chosen_response": "Diagnosis: likely early blight...",
    "rejected_response": "This is definitely late blight. Apply pesticide immediately.",
    "expert_quality_score": 0.9,
    "agronomic_correctness_score": 0.9,
    "management_usefulness_score": 0.8,
    "uncertainty_calibration_score": 0.8,
    "safety_score": 1.0
  }
}
```

All `preference` fields are optional. `chosen_response` and
`rejected_response` are required only when exporting pairwise preference rows.

## Export Pairwise Rows

```bash
PYTHONPATH=src python3 scripts/data/prepare_pairwise_preference_data.py \
  --manifest data/manifests/full/rl_manifest.jsonl \
  --output data/interim/rl_pairwise_preferences.jsonl \
  --summary-output reports/rl_pairwise_preferences_summary.json \
  --allow-empty
```

The script validates the source manifest schema and writes rows with messages,
images, target/verifier metadata, `chosen`, `rejected`, scores, and rationale.
It does not train a reward model.

## Optional Public Data Scaffold

For additional public agricultural QA/VQA/classification data, use the local
conversion scaffold only after license and source metadata are known:

```bash
PYTHONPATH=src python3 scripts/data/ingest_public_agri_qa_manifest.py \
  --input-jsonl /path/to/local/source_rows.jsonl \
  --image-root /path/to/local/images \
  --source-name source_name \
  --source-license "license name or terms" \
  --source-url "documented source URL" \
  --default-task-type vqa \
  --output data/interim/source_name_rl_manifest.jsonl
```

The scaffold does not download data and does not hard-code URLs. It requires
local inputs, source/license metadata, image existence checks, label
normalization, and UnifiedSample schema validation.

## Before Learned Reward Training

- Collect enough expert pairwise rows with source, license, crop/disease, visual
  evidence, and safety metadata.
- Validate manifests with `scripts/validate_rl_manifest.py`.
- Score rule-based rewards with `scripts/score_rl_manifest.py` to find reward
  conflicts before training a separate reward model.
- Add a learned reward backend behind `PreferenceRewardProvider`; do not replace
  the default rule-based GRPO path until it has separate validation coverage.
