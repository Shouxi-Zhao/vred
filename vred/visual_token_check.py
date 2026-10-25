"""Compare the frozen teacher's CE with 729 cached visual tokens vs legacy-style 728.

Legacy ``VisionTower.forward`` applied ``[:, 1:]`` under
``vision_feature_select_strategy="patch"``, so the teacher was trained on 728
visual tokens while the VRED cache keeps all 729.  This bounded diagnostic
measures whether that one-token difference changes teacher caption CE, and
whether dropping the first or the last token matters.

Run on the GPU node (CPU is supported but slow):

    python -m vred.visual_token_check --examples 8 --device cuda --output /tmp/vred-token-check.json
"""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from .configs import load_config
from .data import CachedDataset, Collator
from .model import load_model


def build_examples(assets, count, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(assets["student"], local_files_only=True)
    full = CachedDataset(assets["dpt_json"], assets["cache_index"], tokenizer, template="pretrain",
                         max_length=max_length, vision_tokens=assets["vision_tokens"],
                         vision_width=assets["vision_width"])
    collate = Collator(tokenizer)
    examples = []
    for index in range(min(count, len(full))):
        item = full[index]
        if item["feature"] is None:
            continue
        start, end = item["visual_span"]
        if end - start != assets["vision_tokens"]:
            raise ValueError(f"{item['record_id']}: expected {assets['vision_tokens']} visual tokens")
        batch729 = collate([item]).to(device)
        # Legacy [:, 1:] drops the first visual token; dropping the last is a control.
        for label, position, feature in (("drop_first", start, item["feature"][1:]),
                                         ("drop_last", end - 1, item["feature"][:-1])):
            shorter = dict(
                input_ids=torch.cat((item["input_ids"][:position], item["input_ids"][position + 1:])),
                labels=torch.cat((item["labels"][:position], item["labels"][position + 1:])),
                feature=feature, visual_span=(start, end - 1), record_id=item["record_id"])
            examples.append(dict(record_id=item["record_id"], label=label,
                                 batch729=batch729, batch728=collate([shorter]).to(device)))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=8, help="First N DPT captions (default 8)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--foundation-config")
    parser.add_argument("--max-length", type=int, help="Defaults to foundation.json max_length")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")
    assets = load_config(args.foundation_config)
    max_length = args.max_length or assets["max_length"]
    device = torch.device(args.device)
    torch.set_num_threads(assets["cpu_threads"])
    started = time.monotonic()
    examples = build_examples(assets, args.examples, max_length, device)
    if not examples:
        raise ValueError("No image records found in the DPT annotation file")
    teacher = load_model(assets, role="teacher", dtype=torch.bfloat16, device=device, attn="eager")
    teacher.eval()

    records = []
    with torch.no_grad():
        for example in examples:
            teacher.vision_tokens = assets["vision_tokens"]
            loss729 = teacher(example["batch729"]).loss
            teacher.vision_tokens = assets["vision_tokens"] - 1
            loss728 = teacher(example["batch728"]).loss
            records.append(dict(record_id=example["record_id"], variant=example["label"],
                                ce_729=float(loss729), ce_728=float(loss728),
                                delta_728_minus_729=float(loss728 - loss729),
                                relative_delta=float((loss728 - loss729) / loss729.clamp_min(1e-8))))
            print(json.dumps(records[-1]), flush=True)
    by_variant = {}
    for row in records:
        by_variant.setdefault(row["variant"], []).append(row)
    summary = {}
    for variant, rows in by_variant.items():
        deltas = [row["delta_728_minus_729"] for row in rows]
        summary[variant] = dict(examples=len(rows), mean_ce_729=sum(r["ce_729"] for r in rows) / len(rows),
                                mean_ce_728=sum(r["ce_728"] for r in rows) / len(rows),
                                mean_delta=sum(deltas) / len(deltas),
                                mean_abs_delta=sum(abs(x) for x in deltas) / len(deltas),
                                max_abs_delta=max(abs(x) for x in deltas))
    report = dict(examples=args.examples, device=str(device), dtype="bfloat16",
                  visual_tokens=assets["vision_tokens"], summary=summary, records=records,
                  elapsed_seconds=round(time.monotonic() - started, 2),
                  note="Teacher CE only; this does not by itself decide which token count to use.")
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
