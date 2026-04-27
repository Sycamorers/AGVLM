# Paper Outline

## Abstract

Agricultural VLMs should not only answer correctly; they should know when the evidence is insufficient and ask for clarification. The paper introduces a clarify-vs-respond consultation task, constructs decision-aware agricultural multimodal data, applies two-stage post-training, and evaluates reliability-oriented behavior.

## Introduction

- Agricultural consultation is high consequence and context-dependent.
- General VLMs often answer prematurely from incomplete images or missing agronomic context.
- A useful agricultural assistant needs a decision policy: answer now or ask a targeted clarification.
- Contributions: task, data construction, two-stage post-training, reliability evaluation.

## Method

1. Task formulation
   - action space: respond or clarify
   - expected behavior by ambiguity level
   - clarification quality criteria
2. Decision-aware data construction
   - direct-answer examples
   - clarify-first examples
   - clarify-to-resolution examples
   - source provenance and manual dataset caveats
3. Stage-1 agricultural SFT
   - model, LoRA/freeze settings, data mixture
   - consultation answer style
4. Stage-2 policy optimization
   - GRPO setup
   - reward modules and weights
   - mapping from reward design to reliability goals

## Experiments

1. Experimental settings
   - base model, hardware, precision, checkpoints, configs
2. Benchmarks
   - MIRAGE
   - AgMMU
   - AgroBench
   - local holdout
3. Baselines
   - base model
   - generic SFT baseline if available
   - Agri-SFT
   - Agri-SFT + GRPO
4. Main results
   - answer metrics
   - decision metrics
   - benchmark tables
5. Ablations
   - no-RL
   - reward components
   - LoRA scope
   - freeze strategy
   - data mixtures
6. Behavioral analysis
   - clarify precision and recall
   - unnecessary clarification and premature answer rates
   - ambiguity-level breakdown
7. Error analysis
   - hallucinations
   - missing management advice
   - wrong clarification type

## Reproducibility Appendix

- dataset access matrix
- output directory convention
- run metadata fields
- TensorBoard and artifact regeneration commands
