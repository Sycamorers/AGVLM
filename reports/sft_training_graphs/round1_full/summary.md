# Training Metrics Graphs

## Run

- Title: `SFT Round 1 Full Max3 Turin16 Batch1`
- Metrics: `outputs/sft/phi4-reasoning-vision-15b-full-max3-turin-16gpu-batch1/metrics.jsonl`
- Rows: `4627`
- Step range: `1` to `4571`
- Epoch range: `0.0002187944426211574` to `1.0`

## Key Numbers

| Metric | Count | First | Last | Min | Max | Mean | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 4622 | 14.6309 | 2.97 | 2.2408 | 14.8638 | 3.50622 | 3.27545 |
| eval_loss | 4 | 3.29551 | 2.63084 | 2.63084 | 3.29551 | 2.89802 | 2.83286 |
| grad_norm | 4622 | 19.3417 | 13.1421 | 3.7535 | 23.9803 | 11.057 | 11.0202 |
| learning_rate | 4622 | 0 | 1.1279e-09 | 0 | 5e-06 | 2.49663e-06 | 2.4718e-06 |

## Loss Trend

- First 100-step mean: `10.503674`
- Last 100-step mean: `3.049953`
- Delta: `-7.453721`

## Final Train Summary

- `epoch`: `1.0`
- `global_step`: `4571`
- `total_flos`: `693226633134080.0`
- `train_loss`: `1.3351669137834121`
- `train_runtime`: `194699.9586`
- `train_samples_per_second`: `1.502`
- `train_steps_per_second`: `0.023`

## Plots

- `overview.png`
- `loss_curve.png`
- `grad_norm_curve.png`
- `learning_rate_curve.png`
- `loss_histogram.png`
- `eval_loss_curve.png`
