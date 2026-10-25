"""Bounded DPT gradient-norm probe for choosing ``lambda_write``.

The background document forbids picking a coefficient by equalizing scalar loss
values.  This probe measures, on the same forward, the connector gradient norms
of CE and of the configured write loss at lambda=1.  The write gradient scales
linearly with lambda in this stage, so the printed lambda values show what would
give a target gradient fraction (1%, 10%, 50%, 100% of CE).

Run on the GPU node (CPU is supported but slow):

    python -m vred.gradient_probe --examples 4 --input-model checkpoints/vred/dpt \
        --device cuda --output /tmp/vred-gradient-probe.json
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .configs import load_config, load_training_config
from .data import CachedDataset, Collator
from .effects import EffectCapture, write_positions
from .losses import write_loss
from .model import load_model
from .trainer import VREDTrainer


def _norm(gradients):
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            total += float(gradient.float().square().sum())
    return total ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=4, help="First N DPT captions (default 4)")
    parser.add_argument("--input-model", help="Evaluate the current connector; default is a fresh connector")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--foundation-config")
    parser.add_argument("--config", help="Training JSON; defaults to vred/configs/training.json")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")
    assets = load_config(args.foundation_config)
    config = load_training_config(args.config)
    if not config.get("sparse_logits", True):
        raise ValueError("Set sparse_logits=true for this probe; dense CE is not needed")
    device = torch.device(args.device)
    torch.set_num_threads(assets["cpu_threads"])
    torch.manual_seed(assets["seed"])

    tokenizer = AutoTokenizer.from_pretrained(assets["student"], local_files_only=True)
    dataset = CachedDataset(assets["dpt_json"], assets["cache_index"], tokenizer, template="pretrain",
                            max_length=assets["max_length"], vision_tokens=assets["vision_tokens"],
                            vision_width=assets["vision_width"])
    collate = Collator(tokenizer)
    student = load_model(assets, saved=args.input_model, dtype=torch.bfloat16, device=device)
    teacher = load_model(assets, role="teacher", dtype=torch.bfloat16, device=device)
    trainer = VREDTrainer(student, teacher, "dpt", config, assets["seed"])
    parameters = [parameter for parameter in student.connector.parameters() if parameter.requires_grad]

    records = []
    for index in range(min(args.examples, len(dataset))):
        batch = collate([dataset[index]]).to(device)
        positions = write_positions(batch, config["write_position_limit"], trainer.generator,
                                    scope=config.get("write_position_scope", "response"))
        if not len(positions):
            continue
        with torch.no_grad(), EffectCapture(teacher, [t for _, t in trainer.pairs], batch, positions) as targets:
            teacher.forward_backbone(batch)
        with EffectCapture(student, [s for s, _ in trainer.pairs], batch, positions) as sources:
            output = student.forward_sparse(batch)
        write, raw_write = write_loss(
            student, teacher, sources.effects, targets.effects, trainer.pairs,
            chunk_size=config["query_chunk_size"], temperature=config["tau_write"],
            kind=config["write_loss_type"], vocab_reduction=config["write_vocab_reduction"],
            beta=config["smoothl1_beta"], eta=config["effect_kl_eta"],
            checkpoint_probes=config["checkpoint_probes"],
            vocab_chunk_size=config["probe_vocab_chunk_size"])
        ce = output.loss
        ce_grad = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
        write_grad = torch.autograd.grad(write, parameters, retain_graph=True, allow_unused=True)
        ce_norm, write_norm = _norm(ce_grad), _norm(write_grad)
        ratio = write_norm / ce_norm if ce_norm else float("inf")
        record = dict(record_id=batch.record_ids[0], ce=float(ce.detach()), write=float(write.detach()),
                      write_raw=float(raw_write), write_positions=len(positions),
                      ce_grad_norm=ce_norm, write_grad_norm=write_norm,
                      write_to_ce_grad_ratio=ratio,
                      lambda_equal_gradients=1.0 / ratio if ratio else None,
                      lambda_10pct=0.1 / ratio if ratio else None,
                      lambda_1pct=0.01 / ratio if ratio else None)
        records.append(record)
        print(json.dumps(record), flush=True)
        del output, ce, write, raw_write, ce_grad, write_grad, sources, targets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not records:
        raise ValueError("No DPT example with valid write positions")
    summary = {}
    for key in ("ce", "write", "write_raw", "ce_grad_norm", "write_grad_norm",
                "write_to_ce_grad_ratio", "lambda_equal_gradients", "lambda_10pct", "lambda_1pct"):
        values = [record[key] for record in records if record[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    summary.update(stage="dpt", examples=len(records), write_loss_type=config["write_loss_type"],
                   write_vocab_reduction=config["write_vocab_reduction"],
                   note="lambda values are extrapolations at the measured connector state; validate on held-out data")
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(dict(summary=summary, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
