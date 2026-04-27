# Progress Tracker

Current active milestone: full Turin SFT is running on job `31095385` after a successful distributed preflight on job `31095166`.

| Workstream | Task | Purpose | Status | Current state | Dependency | Output / evidence | Next action | Paper section impacted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Data pipeline | Full manifests | Build SFT, RL, and eval inputs | completed | Full manifests and dataset reports exist | manual source staging | `data/manifests/full/` | keep reports with final run artifacts | data construction |
| Monitoring and logging | TensorBoard and JSONL metrics | Make long runs observable and reproducible | completed | live Turin SFT writes TensorBoard and JSONL metrics; bootstrap scripts pin `setuptools<81` so TensorBoard starts cleanly | project deps installed | `outputs/sft/qwen3-vl-4b-lora-full-turin/tensorboard/`, `outputs/sft/qwen3-vl-4b-lora-full-turin/metrics/train_metrics.jsonl` | keep watching live metrics and export final figures after completion | experimental settings |
| SFT training | Full Turin SFT | Produce Agri-SFT checkpoint | running | real run is active on job `31095385` after successful 4-GPU preflight job `31095166`; latest visible loss has dropped from `16.1963` at step `5` to `9.9954` at step `25` | Turin allocation and current env | `logs/slurm/agri-vlm-sft-full-l4-31095385.out`, `outputs/sft/qwen3-vl-4b-lora-full-turin/` | let the run finish, then confirm checkpoint and export artifacts | stage-1 SFT |
| SFT debugging | OOM mitigation | Keep L4-class run viable | completed | chunked loss plus Turin configs and fresh output dirs have been validated in distributed preflight | updated trainer/config/scripts | `src/agri_vlm/training/sft_trainer.py`, `configs/train/sft_lora_full_turin_*.yaml`, `scripts/hpc/run_sft_*.slurm` | reuse the same Turin path for future resumes or reruns if needed | methods and reproducibility |
| Post-SFT evaluation | Local + MIRAGE | Compare base vs Agri-SFT | blocked | waiting for SFT checkpoint | SFT success | `outputs/benchmarks/<sft_run>/summary.json` | run benchmark suite after checkpoint exists | main results |
| RL training | GRPO | Optimize clarify-vs-respond behavior | blocked | reward scaffold and configs exist | SFT checkpoint | `configs/train/rl_grpo_*.yaml` | start GRPO after SFT eval | stage-2 policy optimization |
| Benchmark setup | MIRAGE and local holdout | Primary and internal benchmarks | completed | implemented in benchmark wrapper | prepared manifests | `scripts/eval/run_benchmark.py` | run full model matrix | experiments |
| Benchmark setup | AgMMU and AgroBench | Knowledge and breadth benchmarks | planned | registry entries document missing raw/eval pieces | access verification | `configs/benchmarks/benchmarks.yaml` | verify official sources and add normalizers | experiments |
| Artifact generation | Curves and tables | Reusable paper figures and tables | completed | export scripts added | metric JSONL and benchmark summaries | `scripts/artifacts/` | export after each run | figures and tables |
| Ablations | Reward, LoRA, freeze, data mixture | Support method claims | optional stretch | documented; configs to be expanded | main result stable | `docs/experiment_roadmap.md` | create focused configs after final model path works | ablations |

## After SFT Result Arrives

If SFT succeeds:

1. Run `scripts/artifacts/export_training_artifacts.py` on the SFT run directory.
2. Run `scripts/eval/run_benchmark.py` for local holdout and MIRAGE.
3. Run `scripts/artifacts/export_benchmark_tables.py` for base vs SFT.
4. Start GRPO from the SFT checkpoint.

If SFT fails:

1. Confirm whether the failure is still the fp32 loss OOM.
2. If yes, lower image pixels in model config or move to B200.
3. If not, record the new failure mode here with log path and traceback.
4. Keep failed outputs; do not overwrite logs without recording the job ID.
