# Progress Tracker

Current active milestone: submit the 100-step Llama 4 Scout full max3 B200 probe from the retained balanced adapter after HPG maintenance. If the probe is healthy, launch the full max3 B200 SFT run.

| Workstream | Task | Purpose | Status | Current state | Dependency | Output / evidence | Next action | Paper section impacted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Data pipeline | Full manifests | Build SFT, RL, and eval inputs | completed | Full manifests and dataset reports exist; max3 Llama 4 train/eval split config is ready | manual source staging | `data/manifests/full/`, `configs/data/sft_train_eval_llama4_max3.yaml` | rebuild max3 split at job launch and keep summary with artifacts | data construction |
| Storage cleanup | Outputs and Orange checkpoints | Keep only next-stage artifacts | completed | local `logs/` and `outputs/` were cleaned except skeleton dirs; Orange SFT storage keeps only the retained balanced adapter | user-approved cleanup | `/orange/hmedeiros/qinruoyao/agvlm/outputs/sft/llama4-scout-17b-16e-lora-balanced-continuation-b200-4gpu-from-step500-peft` | avoid deleting this adapter; future runs should write fresh dirs | reproducibility |
| SFT training | Balanced Llama 4 adapter | Provide next-stage initialization | completed | best completed adapter retained on Orange; step `2500`, eval loss `0.2343`, average reward `0.7480` | previous B200 balanced continuation | retained Orange adapter root | use as `sft_checkpoint_path` for max3 run | stage-1 SFT |
| SFT debugging | AGBASE-disjoint continuation | Diagnose frozen step-500 job | completed | job `31951103` was cancelled at `global_step=500`; no `checkpoint-500`; inline distributed generation eval was the bottleneck; degraded step-450 path was deleted | B200 disjoint run | May 6 session notes, removed failed output dirs | do not resume disjoint run; review AGBASE targets before future AGBASE-only work | methods and limitations |
| SFT training | Full max3 B200 probe | Validate memory, step time, and checkpoint writes | pending | 100-step config and Slurm wrapper are ready | HPG availability and retained adapter | `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced_probe.yaml` | submit after maintenance | experimental settings |
| SFT training | Full max3 B200 run | Produce publishable Agri-SFT checkpoint | pending | full config is ready and disables inline generation metrics | successful probe | `configs/train/sft_lora_b200_4gpu_llama4_scout_full_max3_from_balanced.yaml` | launch after probe passes | stage-1 SFT |
| Monitoring and logging | TensorBoard and JSONL metrics | Make long runs observable and reproducible | ready | trainer writes TensorBoard and JSONL metrics; artifact export script can regenerate plots | project deps installed | `<run_dir>/tensorboard/`, `<run_dir>/metrics/train_metrics.jsonl` | inspect probe metrics and checkpoint files before full launch | experimental settings |
| Post-SFT evaluation | Local + MIRAGE | Compare base vs Agri-SFT | blocked | waiting for completed full max3 checkpoint | SFT success | `outputs/benchmarks/<sft_run>/summary.json` | run benchmark suite after checkpoint exists | main results |
| Generation evaluation | Qualitative and reward predictions | Measure output behavior without freezing training | blocked | must run as a separate job on selected checkpoints | SFT checkpoints | `validation_predictions/*.jsonl` or benchmark prediction JSONL | schedule separately after probe/full checkpoints | main results and error analysis |
| RL training | GRPO | Optimize clarify-vs-respond behavior | blocked | reward scaffold and configs exist | SFT checkpoint and post-SFT eval | `configs/train/rl_grpo_*.yaml` | start GRPO after SFT eval | stage-2 policy optimization |
| Benchmark setup | MIRAGE and local holdout | Primary and internal benchmarks | completed | implemented in benchmark wrapper | prepared manifests | `scripts/eval/run_benchmark.py` | run full model matrix after SFT | experiments |
| Benchmark setup | AgMMU and AgroBench | Knowledge and breadth benchmarks | planned | registry entries document missing raw/eval pieces | access verification | `configs/benchmarks/benchmarks.yaml` | verify official sources and add normalizers | experiments |
| Artifact generation | Curves and tables | Reusable paper figures and tables | ready | export scripts exist; local old outputs were cleaned | metric JSONL and benchmark summaries | `scripts/artifacts/` | export after each successful run | figures and tables |
| Ablations | Reward, LoRA, freeze, data mixture | Support method claims | optional stretch | documented; configs to be expanded | main result stable | `docs/experiment_roadmap.md` | create focused configs after final model path works | ablations |

## After The Probe

If the 100-step probe succeeds:

1. Confirm checkpoint files exist in the probe Orange output directory.
2. Inspect step time, GPU memory, loss trend, and any Slurm warnings.
3. Launch `scripts/hpc/run_sft_b200_4gpu_llama4_scout_full_max3_from_balanced.slurm`.

If the probe fails:

1. Check `logs/slurm/agri-vlm-sft-full-max3-b200-<job_id>.err` and `.out`.
2. If it is memory-bound, lower `max_pixels`, reduce dataloader workers, or reduce `loss_chunk_size` before changing the data plan.
3. If checkpoint writing fails, inspect Orange permissions and available quota.
4. Record the new failure mode here with the exact job ID and traceback.

## After The Full Run

1. Export training curves from the full SFT run directory.
2. Run local holdout and MIRAGE benchmarks against the completed checkpoint or adapter.
3. Export benchmark tables for base vs SFT.
4. Decide whether the checkpoint is strong enough to seed GRPO.
