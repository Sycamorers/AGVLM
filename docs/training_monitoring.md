# Training Monitoring

## What Is Logged

SFT and GRPO now write:

- TensorBoard events in `<run_dir>/tensorboard/`
- structured trainer logs in `<run_dir>/metrics/train_metrics.jsonl`
- a compatibility copy in `<run_dir>/metrics.jsonl`
- resolved config in `<run_dir>/resolved_config.yaml`
- run metadata in `<run_dir>/run_metadata.json`
- artifact pointers in `<run_dir>/artifact_manifest.json` after successful training

The trainer reports loss, eval loss, learning rate, and grad norm when Transformers emits them. GRPO also reports TRL reward scalars when available.

## Distributed Behavior

Only global rank zero writes run metadata and JSONL metric rows. TensorBoard logging is configured through Hugging Face `TrainingArguments` or TRL `GRPOConfig` with `report_to: tensorboard` and `logging_dir: <run_dir>/tensorboard`.

Existing configs set `report_to: tensorboard`. If a config omits `report_to`, the schema default still enables TensorBoard.
Run `bash scripts/hpc/prepare_env.sh` once after pulling these changes so the conda environment installs `tensorboard`.

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

## Current SFT Log Note

The latest visible L4 SFT log, `logs/slurm/agri-vlm-sft-full-l4-30580348.err`, failed with CUDA OOM during fp32 loss conversion in the default model loss. The current config/code mitigation is:

- `loss_chunk_size: 1024`
- custom chunked causal LM loss in `src/agri_vlm/training/sft_trainer.py`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the L4 Slurm script

After rerun, inspect:

```bash
tail -n 120 logs/slurm/agri-vlm-sft-full-l4-<job_id>.err
tail -n 120 logs/slurm/agri-vlm-sft-full-l4-<job_id>.out
ls -lh outputs/sft/qwen3-vl-4b-lora-full-l4/
```
