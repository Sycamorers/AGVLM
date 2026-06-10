# Training Metrics Graphs

## Run

- Title: `Stage6 MC SFT Training`
- Metrics: `outputs/sft/phi4-reasoning-vision-15b-classification-probe-stage6-mc-b200-4gpu/metrics.jsonl`
- Rows: `169`
- Step range: `1` to `160`
- Epoch range: `0.22857142857142856` to `32.0`

## Key Numbers

| Metric | Count | First | Last | Min | Max | Mean | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 160 | 7.8635 | 1.2811 | 1.1122 | 7.8635 | 2.68215 | 2.0748 |
| eval_loss | 8 | 4.35512 | 1.49939 | 1.49939 | 4.35512 | 2.36696 | 1.96725 |
| grad_norm | 160 | 65.4277 | 24.4237 | 11.8481 | 65.7239 | 19.8975 | 16.9284 |
| learning_rate | 160 | 0 | 6.41026e-09 | 0 | 1e-06 | 5e-07 | 5e-07 |

## Loss Trend

- First 100-step mean: `3.429947`
- Last 100-step mean: `1.715667`
- Delta: `-1.714280`

## Final Train Summary

- `epoch`: `32.0`
- `global_step`: `160`
- `total_flos`: `24939158986752.0`
- `train_loss`: `2.6821530029177665`
- `train_runtime`: `4073.6263`
- `train_samples_per_second`: `2.514`
- `train_steps_per_second`: `0.039`

## Plots

- `overview.png`
- `loss_curve.png`
- `grad_norm_curve.png`
- `learning_rate_curve.png`
- `loss_histogram.png`
- `eval_loss_curve.png`
