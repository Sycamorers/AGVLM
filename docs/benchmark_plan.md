# Benchmark Plan

After the Phi-4 SFT run completes, benchmark the selected checkpoint directory
against:

- local holdout
- MIRAGE MMST
- MIRAGE MMMT

Use:

```bash
PYTHONPATH=src python scripts/eval/run_benchmark.py \
  --model-config configs/model/phi4_reasoning_vision_15b_turin_24g.yaml \
  --checkpoint-path <checkpoint_or_run_dir> \
  --tasks local_holdout mirage_mmst mirage_mmmt \
  --prediction-mode model \
  --output-dir outputs/benchmarks/phi4-reasoning-vision-15b-full-max3
```
