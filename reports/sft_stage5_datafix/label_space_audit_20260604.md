# Stage5 Label-Space Audit And Stage6 Probe Setup

Date: 2026-06-04

## Findings

Stage5 classification benchmark errors were not only model collapse; two data issues were present.

1. `digigreen_crop_disease` eval rows included labels outside the attached source label space.
   - Eval examples included `coriander healthy`, `cucumber squash bug`, `medicinal plants potassium deficiency`, and `tomato potassium deficiency`.
   - The Stage5 closed-label eval manifest had derived allowed labels from the Stage5 train manifest, where those labels were absent.
   - This made those eval rows impossible to answer correctly under the attached closed-label contract.

2. `tea_sickness` used the label `gray light`.
   - The model predicted `gray blight` for all tea examples.
   - The source label is treated as a typo and now normalizes to `gray blight` through the shared classification label repair path.

## Fixes

- `src/agri_vlm/data/builders.py`
  - Added `gray light -> gray blight` classification label aliasing.
  - Added fail-fast validation for eval classification labels missing from their source label space.
  - Added reusable classification probe manifest builder.

- `scripts/data/build_classification_probe_manifests.py`
  - Added a thin CLI wrapper for classification-only probe manifest generation.

- `configs/data/sft_eval_stage5_closed_label_datafix_fullspace_phi4_max3.yaml`
  - Added a corrected Stage5 closed-label eval build that derives source label spaces from the full Stage5 source manifest.

- `configs/data/sft_classification_probe_stage6_phi4_max3.yaml`
  - Added the Stage6 classification-only probe data config.

## Generated Files

- `data/manifests/full/sft_eval_phi4_max3_stage5_closed_label_fullspace_stratified1024.jsonl`
  - Rows: `1024`
  - Classification rows: `670`
  - Label-space sizes: banana `7`, DigiGreen `253`, IP102 `102`, PlantDoc `28`, PlantVillage `38`, rice `21`, tea `8`

- `data/manifests/full/sft_classification_probe_stage6_train.jsonl`
  - Rows: `280`
  - Task mix: classification only
  - Source mix: `40` rows each for banana, DigiGreen, IP102, PlantDoc, PlantVillage, rice, and tea

- `data/manifests/full/sft_classification_probe_stage6_eval.jsonl`
  - Rows: `96`
  - Task mix: classification only
  - Source mix: banana `14`, DigiGreen `5`, IP102 `14`, PlantDoc `9`, PlantVillage `14`, rice `20`, tea `20`

Probe train/eval sample ID overlap is `0`, and every probe target label is present in its attached reduced source label space.

## Stage6 Probe

Submitted Slurm job `33919126`:

- Job name: `agri-vlm-sft-stage6-cls-probe`
- Data config: `configs/data/sft_classification_probe_stage6_phi4_max3.yaml`
- Preflight config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_probe_stage6_preflight.yaml`
- Full train config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_classification_probe_stage6.yaml`
- Checkpoint output: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-classification-probe-stage6-b200-4gpu`

Acceptance criterion for continuing to broader SFT: the probe should reach near-perfect same-source closed-label classification accuracy on the 96-example probe eval split. If it does not, the next investigation should stay on prompt/target formatting, adapter loading, and optimization rather than scaling data volume.
