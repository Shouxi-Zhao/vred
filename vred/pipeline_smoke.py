"""Bounded train -> resume -> SFT -> DFT -> reload -> cached-image generation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoTokenizer

from .configs import load_config
from .data import CachedDataset, Collator
from .infer import generate_cached
from .model import load_model
from .smoke import assert_no_vision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic", action="store_true", help="Generate labeled fixtures when real assets are unavailable")
    parser.add_argument("--foundation-config")
    parser.add_argument("--config")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        from .tests.fixtures import make_assets
        args.foundation_config, args.config = make_assets(output / "synthetic_assets")
    common = [sys.executable, "-m", "vred.train", "--backend", "torch", "--accumulation", "1"]
    if args.foundation_config:
        common += ["--foundation-config", args.foundation_config]
    if args.config:
        common += ["--config", args.config]
    logs = []
    for stage, predecessor in (("dpt", None), ("dpt_resume", None), ("sft", "dpt"), ("dft", "sft")):
        target = "dpt" if stage == "dpt_resume" else stage
        command = common + ["--stage", target, "--output", str(output / target),
                            "--max-steps", "2" if stage == "dpt_resume" else "1"]
        if stage == "dpt_resume":
            command += ["--resume", str(output / "dpt")]
        elif predecessor:
            command += ["--input-model", str(output / predecessor)]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / f"{stage}.log").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"Smoke stage {stage} failed; see {output / (stage + '.log')}")
        logs.append(dict(stage=stage, command=command,
                         summary=json.loads((output / target / "summary.json").read_text())))
    assets = load_config(args.foundation_config)
    torch.set_num_threads(assets["cpu_threads"])
    tokenizer = AutoTokenizer.from_pretrained(output / "dft", local_files_only=True)
    dataset = CachedDataset(assets["sft_json"], assets["cache_index"], tokenizer)
    index = next(i for i, row in enumerate(dataset.records) if "image" in row)
    item = dataset[index]
    batch = Collator(tokenizer)([item])
    del dataset
    model = load_model(assets, saved=output / "dft", dtype=torch.bfloat16).eval()
    assert_no_vision(model)
    with torch.no_grad():
        reference = model(batch).logits
    exported = output / "reload"
    model.save_pretrained(exported, tokenizer)
    reloaded = load_model(assets, saved=exported, dtype=torch.bfloat16).eval()
    assert_no_vision(reloaded)
    with torch.no_grad():
        actual = reloaded(batch).logits
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    inference = generate_cached(reloaded, tokenizer, "What is shown?", item["feature"], max_new_tokens=4)
    training_device = json.loads((output / "dft/run_config.json").read_text())["device"]
    report = dict(synthetic=args.synthetic, device=training_device, inference_device="cpu", stages=logs, save_reload_max_error=0,
                  vision_encoder_instantiated=False, feature_shape=list(item["feature"].shape), inference=inference,
                  warning="Synthetic integration fixture; no research or benchmark claim" if args.synthetic else "Bounded implementation check")
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
