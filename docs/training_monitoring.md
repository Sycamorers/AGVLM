# Training Monitoring

## What Is Logged

SFT and GRPO write:

- TensorBoard events in `<run_dir>/tensorboard/`
- structured trainer logs in `<run_dir>/metrics/train_metrics.jsonl`
- a compatibility copy in `<run_dir>/metrics.jsonl`
- resolved config in `<run_dir>/resolved_config.yaml`
- run metadata in `<run_dir>/run_metadata.json`
- artifact pointers in `<run_dir>/artifact_manifest.json` after successful training

The trainer reports loss, eval loss, learning rate, and grad norm when Transformers emits them. GRPO also reports TRL reward scalars when available.

## Distributed Behavior

Only global rank zero writes run metadata and JSONL metric rows. TensorBoard logging is configured through Hugging Face `TrainingArguments` or TRL `GRPOConfig` with `report_to: tensorboard` and `logging_dir: <run_dir>/tensorboard`.

Existing configs set `report_to: tensorboard`. If a config omits `report_to`, the schema default still enables TensorBoard. Run `bash scripts/hpc/prepare_env.sh` once after pulling repo changes so the conda environment installs `tensorboard`.

## HiPerGator TensorBoard

From the repo root on the login node or an interactive job:

```bash
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agri-vlm-v1

tensorboard --logdir outputs --host 0.0.0.0 --port 6006
```

If using SSH port forwarding from a local machine:

```bash
ssh -L 6006:<hpg-host>:6006 <user>@<hpg-login-host>
```

Then open `http://localhost:6006`.

## Active B200 Max3 Monitoring

Probe launch:

```bash
sbatch \
  --export=ALL,TRAIN_CONFIG=configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml \
  scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm
```

Monitor the probe:

```bash
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.out
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.err
tail -n 40 outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced-probe/metrics.jsonl
ls -lh /orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced-probe/
```

Full-run monitor:

```bash
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.out
tail -f logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.err
tail -n 40 outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced/metrics.jsonl
```

Checkpoint and metric checks:

```bash
ls -lh outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced/
ls -lh /orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced/
tensorboard --logdir outputs/sft/llama4-scout-17b-16e-lora-full-max3-b200-4gpu-from-balanced/tensorboard --host 0.0.0.0 --port 6013
```

## May 6 Debug Note

The 4x B200 AGBASE-disjoint continuation was cancelled at step `500` after the step-500 loss eval completed but before a checkpoint was saved. The job was still using GPUs; the expensive path was inline distributed generation metrics.

Future large SFT runs should keep:

```yaml
eval_generation_metrics: false
prediction_loss_only: true
```

Run generation evaluation separately on selected checkpoints.

## Historical L4 Note

The earlier L4 Qwen SFT log `logs/slurm/agri-vlm-sft-full-l4-30580348.err` failed with CUDA OOM during fp32 loss conversion in the default model loss. The mitigation remains in the codebase:

- `loss_chunk_size`
- custom chunked causal LM loss in `src/agri_vlm/training/sft_trainer.py`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in Slurm scripts

This L4 rerun is no longer the active project milestone.
