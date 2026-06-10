# Stage6 Multiple-Choice Classification Probe

Date: 2026-06-05

## Rationale

Stage5 classification remained source-collapsed even after label-space repair. The benchmark predictions were parseable but dominated by one label per source, so a controlled formatting probe is reasonable before another broad SFT round.

This probe keeps the Stage6 classification-only slice at the same seed as the plain-label probe and changes only the classification format:

- Prompt includes five source-local options as `A.` through `E.`
- Supervised target includes `Choice: <letter>` and `Answer: <canonical label>`
- `Answer:` remains the canonical label so existing label parsers can still score generation outputs.

## Generated Files

- Data config: `configs/data/sft_classification_probe_stage6_mc_phi4_max3.yaml`
- Train manifest: `data/manifests/full/sft_classification_probe_stage6_mc_train.jsonl`
  - Rows: `280`
  - Source mix: `40` rows each for banana, DigiGreen, IP102, PlantDoc, PlantVillage, rice, and tea
- Eval manifest: `data/manifests/full/sft_classification_probe_stage6_mc_eval.jsonl`
  - Rows: `96`
  - Source mix: banana `14`, DigiGreen `5`, IP102 `14`, PlantDoc `9`, PlantVillage `14`, rice `20`, tea `20`
- Summary: `data/manifests/full/sft_classification_probe_stage6_mc_summary.json`

Every generated row has five unique options and a `classification_choice_answer` matching `target.canonical_label`.

## Training

Submitted Slurm job `33944810`:

- Job name: `agri-vlm-sft-stage6-cls-mc`
- Preflight config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_probe_stage6_mc_preflight.yaml`
- Full train config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_probe_stage6_mc.yaml`
- Checkpoint output: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-classification-probe-stage6-mc-b200-4gpu`
- TensorBoard port: `6016`

The original plain-label Stage6 job `33919126` remains pending as a format-control run.

## Validation

- Focused tests: `25 passed`
- Preflight dry-run: `280` train rows, classification-only
- Full dry-run: `280` train rows, `96` eval rows, no train/eval overlap
