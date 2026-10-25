# VRED implementation handoff

Give the coding agent this file and the three prompts below, in order. It should read `project_background.md` and `docs/coding_principles.md` first, then execute each prompt rather than only proposing a plan.

The goal is one reproducible VRED pipeline in this repository. Reuse `llavakd/` for tokenization, conversation formatting, model utilities, evaluation, and DeepSpeed conventions. Put new code and configs under `vred/`; keep legacy behavior unchanged unless a small adapter is necessary. Do not copy whole legacy modules.

The implementation uses cached SigLIP2 features, independent two-layer GELU connectors, Qwen2.5-3B teacher, and Qwen2.5-0.5B student. The stages are DPT (`CE + write`, K=3, connector only), SFT (`CE`, full student), and DFT (`CE + write + final forward-KL`, K=6, full student). The write loss must expose SmoothL1, signed-effect KL, and signed-effect JS.

Each prompt must leave working code, run its checks, and append a short entry to `vred/IMPLEMENTATION_STATUS.md` containing files changed, commands/results, and any genuine blocker. Do not launch long training automatically. End with the exact command for the next prompt.

## Order

1. `01_foundation.md` — audit assets, implement cached-feature data and vision-free teacher/student loading.
2. `02_vred_training.md` — implement local visual-message effects, probes/losses, and the three-stage trainer.
3. `03_verify_and_run.md` — add launch/evaluation/diagnostic commands, run smoke checks, and produce the final runbook.

Suggested opening message:

> Read `project_background.md`, `docs/coding_principles.md`, this README, and `implementation_prompts/01_foundation.md`. Implement the foundation now under `vred/`, reusing LLaVA-KD. Verify it with small CPU or single-GPU checks before continuing.
