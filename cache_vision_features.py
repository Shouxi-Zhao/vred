#!/usr/bin/env python
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

from vred.vision import infer_expected_tokens, select_patch_features

try:
    from safetensors.torch import save_file as safe_save_file
except ImportError:
    safe_save_file = None


ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize_cache_key(image_path: str) -> str:
    return image_path.replace("\\", "/").lstrip("./")


def cache_stem(image_path: str) -> str:
    return hashlib.sha1(normalize_cache_key(image_path).encode("utf-8")).hexdigest()


def read_image_paths(data_paths: Iterable[str]) -> List[str]:
    image_paths = []
    seen = set()
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            image_path = entry.get("image")
            if not image_path:
                continue
            key = normalize_cache_key(image_path)
            if key not in seen:
                seen.add(key)
                image_paths.append(image_path)
    return image_paths


def load_existing_index(index_path: str) -> dict:
    if not os.path.exists(index_path):
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_cache_index(existing_index: dict, requested_index: dict, overwrite: bool) -> dict:
    """Append requested cache entries without dropping entries from previous runs."""
    merged = dict(existing_index)
    for key, value in requested_index.items():
        if overwrite or key not in merged:
            merged[key] = value
    return merged


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str], image_folder: str):
        self.image_paths = image_paths
        self.image_folder = image_folder

    def __len__(self):
        return len(self.image_paths)

    def resolve_image_path(self, image_path: str) -> str:
        if os.path.isabs(image_path):
            return image_path

        candidates = [
            os.path.join(self.image_folder, image_path),
            os.path.join(self.image_folder, "llava", "images", image_path),
            os.path.join(os.path.dirname(self.image_folder), "llava", "images", image_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not find image '{image}' under any candidate path: {candidates}".format(
                image=image_path,
                candidates=", ".join(candidates),
            )
        )

    def __getitem__(self, index: int) -> Tuple[str, Image.Image]:
        image_path = self.image_paths[index]
        full_path = self.resolve_image_path(image_path)
        image = Image.open(full_path).convert("RGB")
        return image_path, image


def collate_images(batch):
    paths, images = zip(*batch)
    return list(paths), list(images)


def save_feature(feature: torch.Tensor, output_path: str, save_format: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    feature = feature.detach().cpu().contiguous()
    if save_format == "npy":
        if feature.dtype == torch.bfloat16:
            feature = feature.to(torch.float16)
        with open(tmp_path, "wb") as f:
            np.save(f, feature.numpy())
    elif save_format == "safetensors":
        if safe_save_file is None:
            raise ImportError("safetensors is required when --save-format safetensors is used")
        safe_save_file({"features": feature}, tmp_path)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    os.replace(tmp_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Cache SigLIP patch-token features for LLaVA-KD datasets.")
    parser.add_argument("--data-path", action="append", required=True, help="Annotation JSON. Repeat for more splits.")
    parser.add_argument("--image-folder", required=True, help="Dataset image root for relative image paths.")
    parser.add_argument("--vision-tower", required=True, help="Local SigLIP vision model path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write feature files and index.json.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--vision-feature-layer", type=int, default=-1)
    parser.add_argument("--save-format", choices=("npy", "safetensors"), default="npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    image_paths = read_image_paths(args.data_path)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    extension = "npy" if args.save_format == "npy" else "safetensors"
    requested_index = {
        normalize_cache_key(path): f"{cache_stem(path)}.{extension}"
        for path in image_paths
    }
    index_path = os.path.join(args.output_dir, "index.json")
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    existing_index = load_existing_index(index_path)
    index = merge_cache_index(existing_index, requested_index, args.overwrite)

    model_config = AutoConfig.from_pretrained(args.vision_tower)
    image_processor = AutoImageProcessor.from_pretrained(args.vision_tower)
    expected_tokens = infer_expected_tokens(model_config, image_processor)
    model = SiglipVisionModel.from_pretrained(args.vision_tower)
    model.eval().to(args.device, dtype=dtype)

    dataset = ImagePathDataset(image_paths, args.image_folder)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_images,
        pin_memory=True,
    )

    written = 0
    skipped = 0
    with torch.no_grad():
        for batch_paths, images in tqdm(loader, desc="Caching vision features"):
            uncached_paths = []
            uncached_images = []
            for image_path, image in zip(batch_paths, images):
                key = normalize_cache_key(image_path)
                feature_path = os.path.join(args.output_dir, index[key])
                if os.path.exists(feature_path) and not args.overwrite:
                    skipped += 1
                    continue
                uncached_paths.append(image_path)
                uncached_images.append(image)
            if not uncached_paths:
                continue

            inputs = image_processor(images=uncached_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(args.device, dtype=dtype, non_blocking=True)
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            hidden_state = outputs.hidden_states[args.vision_feature_layer]
            features = select_patch_features(hidden_state, expected_tokens)
            for image_path, feature in zip(uncached_paths, features):
                key = normalize_cache_key(image_path)
                feature_path = os.path.join(args.output_dir, index[key])
                save_feature(feature, feature_path, args.save_format)
                written += 1

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    manifest = {
        "data_paths": [str(Path(path)) for path in args.data_path],
        "image_folder": str(Path(args.image_folder)),
        "vision_tower": str(Path(args.vision_tower)),
        "vision_feature_layer": args.vision_feature_layer,
        "save_format": args.save_format,
        "dtype": args.dtype,
        "num_images": len(image_paths),
        "existing_index_entries": len(existing_index),
        "requested_index_entries": len(requested_index),
        "final_index_entries": len(index),
        "expected_tokens": expected_tokens,
        "written": written,
        "skipped": skipped,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Cache complete: wrote {written}, skipped {skipped}, total indexed {len(image_paths)}")


if __name__ == "__main__":
    main()
