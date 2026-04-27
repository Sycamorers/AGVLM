# Research Plan

## Task Formulation

The main action space is:

- `respond`: provide a direct agricultural answer.
- `clarify`: ask for missing information before committing to diagnosis or management advice.

The optional future extension is a defer or safety fallback action, but it is not the V1 mainline.

Each decision-aware example should make clear whether the current evidence supports an answer. Clarification questions should be high-value: crop identity, symptom distribution, timing, pest visibility, management history, or image quality.

## Data Construction

Decision-aware data should be organized into:

- Direct-answer examples: sufficient image/context and a complete answer target.
- Clarify-first examples: insufficient evidence and a target clarification.
- Clarify-to-resolution examples: first ask for missing information, then answer after the additional evidence arrives.

The manifest schema already supports target decisions through `target.decision` and reward metadata. Dataset-specific access limitations must remain explicit in dataset reports and benchmark docs.

## Training Method

Stage 1: agricultural SFT.

- Teaches crop disease, pest, symptom, VQA, and consultation response behavior.
- Default path is LoRA/QLoRA with the vision encoder frozen.
- The current full L4 SFT run is the active milestone.

Stage 2: GRPO policy optimization.

- Starts from the SFT checkpoint.
- Uses reward modules for exact or normalized labels, synonyms, structured format, uncertainty calibration, clarify-vs-respond, management coverage, and hallucination penalty.
- Reward component choices should map directly to the method section.

## Evaluation Agenda

Report metrics beyond generic answer correctness:

- answer accuracy or exact match
- clarify accuracy
- clarify precision
- clarify recall
- unnecessary clarification rate
- premature answer rate
- hallucination penalty or rate proxy
- management coverage
- average composite reward
- task/category breakdowns when sample metadata supports them

The evaluation code now emits the clarify precision, recall, unnecessary clarification, and premature answer metrics when decision labels are present.

## Model Comparison

Minimum publishable line:

- Base Qwen3-VL-4B-Instruct
- Agri-SFT
- Agri-SFT + GRPO

Extended baselines can be added without changing the benchmark result schema:

- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-32B-Instruct
- LLaVA-OneVision-Qwen2-7B
- LLaVA-v1.6-Mistral-7B
- optional Gemma-3-12B-it
- agricultural checkpoints such as AgroGPT or Agri-LLaVA if runnable checkpoints are available
- GPT-4o or GPT-4.1 as optional closed-source references
