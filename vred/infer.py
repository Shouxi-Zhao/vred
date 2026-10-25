"""Deterministic generation from a saved VRED checkpoint and optional cached image."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .configs import load_config
from .data import SYSTEM, cache_key, read_cache_index
from .model import load_model


def load_feature(index, key, shape=(729, 1152)):
    normalized = cache_key(key)
    if normalized not in index:
        raise ValueError(f"Image {key!r} missing from evaluation cache index")
    path = Path(index[normalized])
    array = np.load(path, allow_pickle=False)
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"Image {key!r}, cache={path}: expected finite floating {shape}, got {array.shape}/{array.dtype}")
    return torch.from_numpy(array)


@torch.inference_mode()
def generate_cached_batch(model, tokenizer, questions, features, max_new_tokens=128, template="qwen2_base"):
    """Deterministic batched generation with left padding and per-sample image features."""
    if len(questions) != len(features):
        raise ValueError("questions and features must have the same length")
    if not questions:
        return []
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("tokenizer requires pad_token_id for batched generation")
    device = next(model.parameters()).device
    maximum = model.language_model.config.max_position_embeddings
    sequences, image_spans, lengths = [], [], []
    for question, feature in zip(questions, features):
        if "<image>" in question:
            raise ValueError("Pass image features separately; question must not contain <image>")
        if feature is not None:
            if tuple(feature.shape) != (model.vision_tokens, model.connector[0].in_features) or not torch.isfinite(feature).all():
                raise ValueError("Invalid cached feature dimensions/values for this model")
        if template == "pretrain":
            before, after = ("", "") if feature is not None else (question + "\n", "")
        else:
            before, after = SYSTEM + "USER: ", "\n" + question + " ASSISTANT:"
            if feature is None:
                before, after = SYSTEM + "USER: " + question + " ASSISTANT:", ""
        ids = tokenizer.encode(before, add_special_tokens=False)
        span = len(ids)
        if feature is not None:
            ids += [pad] * model.vision_tokens
        ids += tokenizer.encode(after, add_special_tokens=False)
        if len(ids) + max_new_tokens > maximum:
            raise ValueError(f"Prompt plus generation exceeds context limit {maximum}")
        sequences.append(ids)
        image_spans.append((span, span + model.vision_tokens) if feature is not None else None)
        lengths.append(len(ids))
    max_len = max(lengths)
    input_ids = torch.full((len(sequences), max_len), pad, device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), device=device, dtype=torch.long)
    offsets = []
    for row, sequence in enumerate(sequences):
        offset = max_len - len(sequence)
        input_ids[row, offset:] = torch.tensor(sequence, device=device, dtype=torch.long)
        attention_mask[row, offset:] = 1
        offsets.append(offset)
    model.eval()
    embeddings = model.language_model.get_input_embeddings()(input_ids).clone()
    weight = model.connector[0].weight
    for row, (feature, span) in enumerate(zip(features, image_spans)):
        if feature is None:
            continue
        start = offsets[row] + span[0]
        stop = start + model.vision_tokens
        embeddings[row, start:stop] = model.connector(feature.to(device=weight.device, dtype=weight.dtype))
    output = model.language_model.generate(input_ids=input_ids, inputs_embeds=embeddings,
        attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False,
        use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    results = []
    for row, length in enumerate(lengths):
        generated = output[row, max_len:].tolist()
        results.append(dict(text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
                            token_ids=generated, prompt_tokens=length,
                            cached_image=features[row] is not None))
    return results


@torch.inference_mode()
def generate_cached(model, tokenizer, question, feature=None, max_new_tokens=128, template="qwen2_base"):
    return generate_cached_batch(model, tokenizer, [question], [feature], max_new_tokens, template)[0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--foundation-config")
    parser.add_argument("--cache-index")
    parser.add_argument("--image-key")
    parser.add_argument("--image", help="Encode this image online with the vision tower instead of using a cache key")
    parser.add_argument("--vision-tower", help="SigLIP2 vision tower for --image (default: assets.vision_tower)")
    parser.add_argument("--vision-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output")
    args = parser.parse_args()
    assets = load_config(args.foundation_config)
    torch.set_num_threads(assets["cpu_threads"])
    model = load_model(assets, saved=args.model, dtype=torch.bfloat16,
                       device="cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.image_key:
        feature = load_feature(read_cache_index(args.cache_index or assets["cache_index"]), args.image_key)
    elif args.image:
        from .vision import OnlineVisionEncoder, open_image
        vision_tower = args.vision_tower or assets.get("vision_tower")
        if not vision_tower:
            raise ValueError("--image requires --vision-tower or assets.vision_tower")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if args.vision_dtype == "fp16" else torch.bfloat16
        feature = OnlineVisionEncoder(vision_tower, device=device, dtype=dtype).encode(open_image(None, args.image))
    else:
        feature = None
    result = generate_cached(model, tokenizer, args.question, feature, args.max_new_tokens)
    print(json.dumps(result))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
