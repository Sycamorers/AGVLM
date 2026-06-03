# Benchmark Status Report

- phase: `sft`
- overall ok: `True`
- errors: `0`
- warnings: `2`

| Status | Severity | Check |
| --- | --- | --- |
| ok | error | rl_benchmark has no train/eval sample-id or group overlap. |
| ok | error | rl_benchmark split manifests have no duplicate sample IDs. |
| ok | error | sft_benchmark has no train/eval sample-id or group overlap. |
| ok | error | sft_benchmark split manifests have no duplicate sample IDs. |
| ok | error | External baseline model config parses. |
| ok | warning | AGVLM checkpoint config parses; placeholder paths are warnings until selected for a run. |
| ok | error | Prediction parser handles Answer, Decision, and structured sections. |
| ok | error | Metrics module can score synthetic benchmark predictions. |
| ok | error | Summary table can be refreshed. |
| ok | error | Required benchmark Slurm scripts exist. |
| fail | warning | Required benchmark/project docs exist. |
| ok | info | No SFT training files/configs/Slurm scripts are dirty. |

## Dirty SFT Guard

The status check reports dirty SFT training files if present, but it does not revert user work.

```text

```
