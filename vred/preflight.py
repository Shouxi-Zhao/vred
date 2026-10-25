"""Record coverage before training; never substitute or silently discard records."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .configs import load_config
from .data import CachedDataset, cache_key


def audit_dataset(dataset, validate_arrays=False):
    seen, missing_keys, missing_files, malformed = set(), [], [], []
    images, duplicates = 0, 0
    online = bool(getattr(dataset, "online", False))
    for i, row in enumerate(dataset.records):
        messages = row.get("conversations", [])
        if (not isinstance(messages, list) or not messages or len(messages) % 2
                or any(not isinstance(m, dict) or not isinstance(m.get("value"), str)
                       or m.get("from") != ("human" if j % 2 == 0 else "gpt")
                       for j, m in enumerate(messages))):
            malformed.append(f"{dataset.sources[i]}: malformed alternating human/gpt conversations")
            continue
        placeholders = sum(m["value"].count("<image>") for m in messages)
        if placeholders != int("image" in row) or any("<image>" in m["value"] for m in messages[1::2]):
            malformed.append(f"{dataset.sources[i]}: image field/placeholder mismatch ({placeholders} placeholders)")
            continue
        image = row.get("image")
        if "image" not in row:
            continue
        source = dataset.sources[i]
        if not isinstance(image, str) or not image:
            malformed.append(f"{source}: expected one nonempty image path, got {image!r}")
            continue
        images += 1
        key = cache_key(image)
        duplicates += int(key in seen)
        seen.add(key)
        if online:
            from .vision import resolve_image_path
            try:
                resolve_image_path(dataset.image_root, image)
            except (FileNotFoundError, OSError) as error:
                missing_files.append(f"{source}: image={image!r}, root={dataset.image_root}: {error}")
        else:
            if key not in dataset.index:
                missing_keys.append(f"{source}: image={image!r}, index={dataset.cache_index}")
            elif not Path(dataset.index[key]).is_file():
                missing_files.append(f"{source}: cache={dataset.index[key]}")
        if validate_arrays:
            if online:
                try:
                    dataset[i]
                except ValueError as error:
                    malformed.append(str(error))
            elif key in dataset.index and Path(dataset.index[key]).is_file():
                try:
                    dataset[i]
                except ValueError as error:
                    malformed.append(str(error))
    return dict(records=len(dataset), image_records=images, text_records=sum("image" not in r for r in dataset.records),
                unique_images=len(seen), repeated_image_records=duplicates,
                index_entries=0 if online else len(dataset.index), missing_keys=len(missing_keys),
                missing_files=len(missing_files),
                malformed=len(malformed), arrays_validated=validate_arrays,
                failures=(missing_keys + missing_files + malformed)[:20])


def require_coverage(report):
    if report["missing_keys"] or report["missing_files"] or report["malformed"]:
        raise ValueError("Cache preflight failed: " + json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-config")
    parser.add_argument("--validate-arrays", action="store_true", help="Read and encode every record; expensive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assets = load_config(args.foundation_config)
    tokenizer = AutoTokenizer.from_pretrained(assets["student"], local_files_only=True)
    reports = {}
    for stage, template in (("dpt", "pretrain"), ("sft", "qwen2_base")):
        dataset = CachedDataset(assets[f"{stage}_json"], assets["cache_index"], tokenizer, template=template,
                                max_length=assets["max_length"])
        reports[stage] = audit_dataset(dataset, args.validate_arrays)
    Path(args.output).write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports.values():
        require_coverage(report)


if __name__ == "__main__":
    main()
