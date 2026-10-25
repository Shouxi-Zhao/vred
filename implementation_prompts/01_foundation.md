# 01 Foundation: data and models

Read the background(project_background.md) and coding principles(docs/coding_principles.md). Inspect existing LLaVA-KD data/model/training code before editing.

Build the smallest usable foundation under `vred/`:

1. Add a dataset/collator that reads the two specified JSON files, resolves `vision_cache/siglip2-so400m-patch14-384/index.json`, loads `.npy` features on CPU, validates `(729, 1152)` finite arrays, expands the visual placeholder, and returns labels, causal masks, visual spans, and response positions. Preserve all 729 tokens. Handle text-only examples explicitly; fail clearly on missing or multiple-image records.
2. Add a vision-free student wrapper and a frozen teacher loader using the supplied checkpoints. Reuse the existing connector implementation or match `Linear(vision, hidden) -> GELU -> Linear(hidden, hidden)`. Resolve the teacher LoRA/weight format exactly once and record the decision. Validate tokenizer/vocabulary compatibility.
3. Add a tiny loader/forward check: one text example, one cached-image example, shifted labels, connector dimensions, save/reload agreement, and proof that no vision encoder is instantiated for training.

Keep APIs simple: one batch object, one model entry point, and configuration values in `vred/configs/`. Do not implement visual-effect extraction yet. Do not silently skip bad cache entries or fall back to online vision encoding.

Acceptance: the smoke check passes on available hardware, errors identify the bad record/path, and `vred/IMPLEMENTATION_STATUS.md` records verified facts and the command for prompt 02.
