# Stage4 Data-Fix Execution

Date: 2026-06-02

## Submitted Jobs

| Job | ID | Partition | Dependency | Purpose |
| --- | ---: | --- | --- | --- |
| `agri-vlm-sft-stage4-datafix` | `33727177` | `hpg-b200` | none | Runs Stage4 preflight first, then full 800-step SFT if preflight succeeds. |
| `agri-sft-stage4-bench` | `33727850` | `hpg-turin` | `afterok:33727177` | Benchmarks the completed Stage4 adapter on `splits_stage4_datafix`. |

## Training Configuration

- Model config: `configs/model/phi4_reasoning_vision_15b_b200.yaml`
- Preflight config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage4_datafix_preflight.yaml`
- Full config: `configs/train/sft_phi4_reasoning_vision_15b_b200_4gpu_stage4_datafix.yaml`
- Stage4 data prep config passed to the B200 wrapper: `configs/data/sft_stage4_closed_label_datafix_phi4_max3.yaml`
- Expected adapter path: `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/phi4-reasoning-vision-15b-stage4-datafix-b200-4gpu`

## Benchmark Configuration

- Model key: `agvlm_phi4_sft_stage4_datafix_b200_candidate`
- Output dir: `benchmarks/vlm_baselines/results/agvlm_stage4_datafix_benchmark_20260602`
- Split dir: `benchmarks/vlm_baselines/splits_stage4_datafix`
- Eval manifest: `data/manifests/full/sft_eval_phi4_max3_stage4_closed_label_stratified768.jsonl`
- Train manifest: `data/manifests/full/sft_train_phi4_max3_stage4_closed_label_datafix.jsonl`
- Split summary: `data/manifests/full/sft_train_eval_phi4_max3_stage4_datafix_summary.json`
- Quantization: `4bit`

## Status Check

Submitted state immediately after launch:

```text
33727177  hpg-b200   agri-vlm-sft-stage4-datafix  PENDING  (Priority)
33727850  hpg-turin  agri-sft-stage4-bench        PENDING  (Dependency)
```
