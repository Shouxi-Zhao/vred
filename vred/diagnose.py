"""Bounded ablation diagnostic on identical training examples and initial weights."""

import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import resource
import time

import torch
from transformers import AutoTokenizer

from .configs import load_config, load_training_config
from .data import CachedDataset, Collator
from .model import load_model
from .trainer import VREDTrainer


def variants():
    return [("ce_only", 0, 0, 3, "smoothl1"), ("final_kl", 0, 1, 3, "smoothl1")] + [
        (f"write_{kind}_k{k}", 1, 0, k, kind) for k in (3, 6)
        for kind in ("smoothl1", "effect_kl", "effect_js")] + [("final_kl_write", 1, 1, 6, "smoothl1")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-config")
    parser.add_argument("--config")
    parser.add_argument("--input-model", help="Use the same saved SFT model for every comparison")
    parser.add_argument("--examples", type=int, default=2, help="First N training examples, at most 8")
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic", action="store_true", help="Use explicit generated test fixtures instead of project assets")
    args = parser.parse_args()
    if not 1 <= args.examples <= 8:
        parser.error("--examples must be between 1 and 8")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        parser.error("Choose a fresh output directory")
    if args.synthetic:
        from .tests.fixtures import make_assets
        args.foundation_config, args.config = make_assets(output / "synthetic_assets")
    assets, config = load_config(args.foundation_config), load_training_config(args.config)
    torch.set_num_threads(assets["cpu_threads"])
    torch.manual_seed(assets["seed"])
    tokenizer = AutoTokenizer.from_pretrained(assets["student"], local_files_only=True)
    dataset = CachedDataset(assets["sft_json"], assets["cache_index"], tokenizer,
                            max_length=assets["max_length"])
    if args.examples > len(dataset):
        parser.error("--examples exceeds dataset length")
    batches = [Collator(tokenizer)([dataset[i]]) for i in range(args.examples)]
    del dataset
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    student = load_model(assets, saved=args.input_model, dtype=torch.bfloat16, device=device)
    teacher = load_model(assets, role="teacher", dtype=torch.bfloat16, device=device)
    initial = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    records = []
    for name, write, final, k, kind in variants():
        student.load_state_dict(initial)
        torch.manual_seed(assets["seed"])
        settings = copy.deepcopy(config)
        settings["write_loss_type"] = kind
        settings["stages"]["dft"].update(lambda_write=write, lambda_logits=final, pairs=k)
        trainer = VREDTrainer(student, teacher, "dft", settings, assets["seed"])
        optimizer = torch.optim.AdamW(student.parameters(), lr=settings["stages"]["dft"]["lr"], weight_decay=settings["weight_decay"])
        for step, cpu_batch in enumerate(batches):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = trainer.loss(cpu_batch.to(device))
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
                raise FloatingPointError(f"{name}, example {step}: nonfinite loss/gradient")
            gradient = float(student.connector[0].weight.grad.float().norm()) if student.connector[0].weight.grad is not None else 0.0
            torch.nn.utils.clip_grad_norm_(student.parameters(), settings["max_grad_norm"])
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record = {key: float(value) if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
            record.update(variant=name, step=step, record_ids=list(cpu_batch.record_ids), seed=assets["seed"],
                          k=k, write_loss_type=kind, lambda_write=write, lambda_logits=final,
                          layer_pairs=trainer.pairs,
                          write_vocab_reduction=settings["write_vocab_reduction"], connector_grad_norm=gradient,
                          seconds=time.monotonic() - started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None)
            records.append(record)
            del loss, metrics
        del optimizer
        gc.collect()
    report = dict(synthetic=args.synthetic, diagnostic_only=True, source=assets["sft_json"],
                  input_model=args.input_model, examples=args.examples, steps_per_variant=args.examples,
                  precision="bfloat16; FP32 probes/losses", optimizer="CPU AdamW BF16 parameters" if device == "cpu" else "AdamW",
                  training=config, records=records)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(dict(variants=len(variants()), examples=args.examples, records=len(records), output=str(output))))


if __name__ == "__main__":
    main()
