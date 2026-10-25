"""Isolate CPU probe memory from annotation parsing; synthetic data, full vocabulary."""

import argparse
import json
import resource

import torch

from vred.configs import load_training_config
from vred.data import Collator
from vred.trainer import VREDTrainer
from .test_numerics import tiny_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    class Tokenizer:
        pad_token_id = 0
    ids = torch.ones(859, dtype=torch.long)
    ids[1:730] = 0
    labels = torch.full_like(ids, -100)
    labels[-128:] = 1
    batch = Collator(Tokenizer())([dict(input_ids=ids, labels=labels,
        feature=torch.randn(729, 1152, dtype=torch.float16), visual_span=(1, 730), record_id="memory-probe")])
    student = tiny_model(depth=6, vision_tokens=729, vision_width=1152, vocab=151936).to(torch.bfloat16)
    teacher = tiny_model(hidden=32, heads=4, kv=2, depth=6, vision_tokens=729, vision_width=1152,
                         vocab=151936).to(torch.bfloat16)
    config = load_training_config()
    config["checkpoint_probes"] = args.checkpoint
    trainer = VREDTrainer(student, teacher, "dft", config)
    loss, metrics = trainer.loss(batch)
    loss.backward()
    print(json.dumps(dict(checkpoint_probes=args.checkpoint,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        loss=float(loss.detach()), write=float(metrics["write"]),
        connector_grad_norm=float(student.connector[0].weight.grad.float().norm()),
        positions=metrics["write_positions"])), flush=True)


if __name__ == "__main__":
    main()
