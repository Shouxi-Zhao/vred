# 03 Verify, launch, and deliver

Make the pipeline runnable from a clean shell using the code from prompts 01–02.

1. Add concise `vred/` launch configs for one-node six-GPU DeepSpeed ZeRO-2 DPT, SFT, and DFT, with resume paths, seed, batch/accumulation, precision, loss coefficients, and output directories explicit. Log resolved dataset counts, cache coverage, tokenizer IDs, layer pairs, loss reductions, and trainable parameters.
2. Reuse LLaVA-KD evaluation code through an adapter. Add commands for the ten local benchmark directories and a result collector that distinguishes locally scored results from submission-required results. Do not fabricate scores.
3. Add a single diagnostic command covering the core ablations: CE-only versus write, final-KL versus write, K=3/6, and the three write-loss modes. It should use a bounded subset and write JSON/CSV metrics, not start a full sweep.
4. Run all cheap checks, one bounded training smoke test, checkpoint save/reload, and inference on a cached image. Fix failures directly. Write `vred/README.md` with setup, three stage commands, evaluation, and troubleshooting.

Use the same data exposure for comparisons and keep test benchmarks out of tuning. Report unavailable GPU, cache, checkpoint, or external scoring prerequisites plainly. Do not claim research improvement from smoke tests.

Acceptance: a new user can run the documented DPT → SFT → DFT commands after setting paths, the smoke pipeline completes end to end, and `vred/IMPLEMENTATION_STATUS.md` contains final verification results and any remaining external prerequisite.
