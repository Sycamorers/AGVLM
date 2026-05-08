# Inference-Only VLM Baselines

This directory is isolated from the repository training path. The code reads
normalized manifests, runs Hugging Face VLMs in `model.eval()` and
`torch.no_grad()` mode, and writes only under `benchmarks/vlm_baselines/`.
It does not call training scripts, trainer classes, optimizer/scheduler code,
or write checkpoints, tokenizer files, LoRA adapters, training logs, or training
output directories.

## Split Policy

The repository already has an active no-overlap held-out SFT evaluation
manifest:

- train: `data/manifests/full/sft_train_phi4_max3_no_eval_overlap.jsonl`
- held-out eval: `data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl`
- summary: `data/manifests/full/sft_train_eval_phi4_max3_summary.json`

`split_dataset.py` therefore reuses the existing held-out eval data:

- benchmark `test`: all 512 rows from `sft_eval_phi4_max3_stratified512.jsonl`
- benchmark `val`: the 120 rows from that manifest whose original split tag is
  `validation`

No source data is changed. The generated split manifests are:

- `benchmarks/vlm_baselines/splits/test_manifest.jsonl`
- `benchmarks/vlm_baselines/splits/val_manifest.jsonl`
- `benchmarks/vlm_baselines/splits/distribution_report.json`
- `benchmarks/vlm_baselines/splits/distribution_report.md`

If the active held-out eval manifest is missing, the splitter falls back to a
deterministic seed-42 group-aware split of the normalized SFT manifest using
70% train, 10% val, 20% test. Group keys are selected in this priority order:
`scene_id`, `dialogue_id`/`conversation_id`, `image_id`/`video_id`,
`subject_id`/`participant_id`, `source_image_id`, then source file stem.

Current generated distribution:

- test rows: 512
- val rows: 120
- unique sample ids across both manifests: 512
- task mix across generated manifests: classification 354, VQA 250,
  clarify-or-respond 28
- missing image samples: 0
- missing text prompts: 0

## GPU Plan

Baselines are intended to run sequentially. The default is one model per job on
one approximately 24GB GPU, batch size 1, no quantization unless an OOM fallback
is triggered or `--quantization 4bit` is explicitly requested.

| Model | GPUs | Default dtype | Default quantization | Fallback | Notes |
|---|---:|---|---|---|---|
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 1 | bf16 | none | 4bit | Smallest baseline. |
| `google/paligemma2-3b-mix-448` | 1 | bf16 | none | 4bit | Single-image model; multi-image samples use the first image and record that policy. |
| `microsoft/Phi-4-multimodal-instruct` | 1 | bf16 | none | 4bit | Uses remote-code Phi-4 multimodal adapter. |
| `allenai/Molmo2-4B` | 1 | bf16 | none | 4bit | Uses remote-code Molmo2 adapter. |
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | 1 | bf16 | none | 4bit | Largest baseline; keep `max_new_tokens` conservative. |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 1 | bf16 | none | 4bit | Uses `qwen-vl-utils` when available and caps pixel budget. |

## Setup

Use the repository environment first. Benchmark-only optional additions are
listed separately:

```bash
python -m pip install -r benchmarks/vlm_baselines/requirements-benchmark.txt
```

Some model repositories are gated or require accepting upstream licenses on
Hugging Face before download. The runner records dtype, quantization, model
revision when available, CUDA device info, command arguments, and generation
settings in `benchmarks/vlm_baselines/results/metadata/`.

## Commands

Create or refresh split manifests:

```bash
PYTHONPATH=benchmarks/vlm_baselines \
python benchmarks/vlm_baselines/split_dataset.py --force
```

Smoke test one model on 5 validation samples:

```bash
PYTHONPATH=benchmarks/vlm_baselines \
python benchmarks/vlm_baselines/run_baselines.py \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --split val \
  --batch-size 1 \
  --max-new-tokens 64 \
  --dtype bf16 \
  --max-samples 5 \
  --smoke-test \
  --device cuda:0 \
  --output-dir benchmarks/vlm_baselines/results_smoke
```

Run one full held-out benchmark:

```bash
PYTHONPATH=benchmarks/vlm_baselines \
python benchmarks/vlm_baselines/run_baselines.py \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --split test \
  --batch-size 1 \
  --max-new-tokens 128 \
  --dtype bf16 \
  --quantization none \
  --seed 42 \
  --device cuda:0 \
  --output-dir benchmarks/vlm_baselines/results
```

Run all six locally, sequentially:

```bash
for MODEL_NAME in \
  "HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
  "google/paligemma2-3b-mix-448" \
  "microsoft/Phi-4-multimodal-instruct" \
  "allenai/Molmo2-4B" \
  "llava-hf/llava-onevision-qwen2-7b-ov-hf" \
  "Qwen/Qwen2.5-VL-3B-Instruct"; do
  MODEL_NAME="$MODEL_NAME" SPLIT=test MAX_SAMPLES=0 \
    bash benchmarks/vlm_baselines/scripts/run_one_model.sh
done
```

Build or refresh the summary table after runs:

```bash
PYTHONPATH=benchmarks/vlm_baselines \
python benchmarks/vlm_baselines/evaluate_predictions.py \
  --output-dir benchmarks/vlm_baselines/results/metrics \
  --refresh-summary-only
```

The table is written to
`benchmarks/vlm_baselines/results/metrics/summary_table.csv`.

## Slurm

Smoke test:

```bash
sbatch --export=ALL,MODEL_NAME="HuggingFaceTB/SmolVLM2-2.2B-Instruct",SPLIT=val,MAX_SAMPLES=5,MAX_NEW_TOKENS=64 \
  benchmarks/vlm_baselines/slurm/run_vlm_baselines_24gb.sbatch
```

One model on the full benchmark:

```bash
sbatch --export=ALL,MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct",SPLIT=test,MAX_SAMPLES=0 \
  benchmarks/vlm_baselines/slurm/run_vlm_baselines_24gb.sbatch
```

All six models sequentially in one Slurm job:

```bash
sbatch --export=ALL,RUN_ALL=1,SPLIT=test,MAX_SAMPLES=0 \
  benchmarks/vlm_baselines/slurm/run_vlm_baselines_24gb.sbatch
```

The Slurm script requests one GPU and does not launch concurrent model jobs.

## Outputs

Predictions:

```text
benchmarks/vlm_baselines/results/predictions/{model_slug}_{split}.jsonl
```

Metrics:

```text
benchmarks/vlm_baselines/results/metrics/{model_slug}_{split}_metrics.json
benchmarks/vlm_baselines/results/metrics/summary_table.csv
```

Each prediction row includes the raw output, normalized prediction, ground
truth, references, sample id, source dataset, image paths, prompt, model name,
generation config, dtype, quantization, runtime, image policy, and any error
message.

Metrics include classification accuracy, macro-F1, weighted-F1, per-class
precision/recall/F1, confusion matrix, VQA exact match, relaxed accuracy,
token-F1, clarify-vs-respond metrics, invalid-output rate, and failure rate.

## Adapter References

The model adapters follow Hugging Face model card or Transformers examples for
SmolVLM2, PaliGemma 2, Phi-4 multimodal, Molmo2, LLaVA-OneVision, and
Qwen2.5-VL:

- https://huggingface.co/docs/transformers/model_doc/smolvlm
- https://huggingface.co/google/paligemma2-3b-mix-448
- https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- https://huggingface.co/allenai/Molmo2-4B
- https://huggingface.co/docs/transformers/model_doc/llava_onevision
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
