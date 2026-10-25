"""Strict cache loading plus opt-in online image loading; spans are expanded-sequence coordinates."""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE = -100
SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions. ")


def cache_key(value):
    return value.replace("\\", "/").lstrip("./")


def read_cache_index(path):
    path = Path(path)
    root = path.parent.resolve()
    result = {}
    for key, value in json.loads(path.read_text()).items():
        normalized = cache_key(key)
        if normalized in result:
            raise ValueError(f"{path}: normalized cache key collision: {key!r}")
        resolved = (root / value).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"{path}: cache mapping escapes root: {key!r} -> {value!r}")
        result[normalized] = str(resolved)
    return result


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    features: tuple
    visual_spans: torch.Tensor
    response_mask: torch.Tensor
    write_mask: torch.Tensor
    record_ids: tuple
    images: tuple = ()

    @property
    def causal_mask(self):
        """Boolean allowed-attention mask [B, 1, query, key], including padding."""
        length = self.input_ids.shape[1]
        lower = torch.ones(length, length, dtype=torch.bool, device=self.input_ids.device).tril()
        return (lower[None, None] & self.attention_mask[:, None, None, :]
                & self.attention_mask[:, None, :, None])

    @property
    def response_positions(self):
        """[batch, prediction position]; targets are labels[batch, position + 1]."""
        return self.response_mask.nonzero()

    def to(self, device):
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            # Cached arrays and PIL images stay on CPU; only the connector's input is transferred.
            values[field.name] = value
        return Batch(**values)


class CachedDataset(Dataset):
    def __init__(self, json_paths, cache_index=None, tokenizer=None, template="qwen2_base",
                 max_length=2048, vision_tokens=729, vision_width=1152,
                 online=False, image_root=None):
        if template not in ("qwen2_base", "pretrain"):
            raise ValueError(f"Unknown template: {template}")
        if tokenizer is None:
            raise ValueError("A tokenizer is required")
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for exact response offsets")
        self.tokenizer, self.template = tokenizer, template
        self.max_length = max_length
        self.shape = (vision_tokens, vision_width)
        self.online = bool(online)
        self.image_root = None if image_root is None else str(image_root)
        if self.online:
            if self.image_root is None:
                raise ValueError("online vision loading requires image_root")
            self.cache_index = None
            self.index = {}
        else:
            if cache_index is None:
                raise ValueError("cache_index is required unless online=True")
            self.cache_index = Path(cache_index)
            self.index = read_cache_index(self.cache_index)
        if isinstance(json_paths, (str, Path)):
            json_paths = [json_paths]
        self.records, self.sources = [], []
        for path in json_paths:
            rows = json.loads(Path(path).read_text())
            if not isinstance(rows, list):
                raise ValueError(f"{path}: expected a JSON list")
            self.records.extend(rows)
            self.sources.extend(f"{path}[{i}] id={r.get('id', '<absent>')}" for i, r in enumerate(rows))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        source = self.sources[index]
        try:
            return self._encode(self.records[index], source)
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError(f"{source}: {error}") from error

    def _encode(self, record, source):
        image = record.get("image")
        if "image" in record and (not isinstance(image, str) or not image):
            raise ValueError(f"expected one nonempty image path, got {image!r}")
        messages = record["conversations"]
        if not messages or len(messages) % 2:
            raise ValueError("expected complete alternating human/gpt turns")
        count = sum(m["value"].count("<image>") for m in messages)
        if count != int(image is not None):
            raise ValueError(f"image={image!r}: expected {int(image is not None)} visual placeholder, got {count}")
        feature, image_value = None, None
        if image is not None:
            if self.online:
                from .vision import open_image
                try:
                    image_value = open_image(self.image_root, image)
                except (OSError, ValueError) as error:
                    raise ValueError(f"image={image!r}, root={self.image_root}: {error}") from error
            else:
                key = cache_key(image)
                if key not in self.index:
                    raise ValueError(f"image={image!r} missing from cache index {self.cache_index}")
                path = Path(self.index[key])
                try:
                    array = np.load(path, allow_pickle=False)
                    if array.shape != self.shape or array.dtype.kind != "f" or not np.isfinite(array).all():
                        raise ValueError(f"expected finite floating array {self.shape}, got {array.shape}/{array.dtype}")
                    feature = torch.from_numpy(array)
                except (OSError, ValueError) as error:
                    raise ValueError(f"image={image!r}, cache={path}: {error}") from error

        prompt = SYSTEM if self.template == "qwen2_base" else ""
        responses = []
        for turn in range(0, len(messages), 2):
            user, assistant = messages[turn:turn + 2]
            if user["from"] != "human" or assistant["from"] != "gpt":
                raise ValueError(f"turn {turn}: expected human/gpt roles")
            if "<image>" in assistant["value"]:
                raise ValueError(f"turn {turn}: image placeholder in assistant response")
            question = user["value"]
            if "<image>" in question:
                question = "<image>\n" + question.replace("<image>", "").strip()
            if self.template == "pretrain":
                if len(messages) != 2:
                    raise ValueError("pretrain requires exactly one human/gpt pair")
                prompt += "<image>" if image else question + "\n"
            else:
                prompt += "USER: " + question + " ASSISTANT:"
            start = len(prompt)
            if self.template == "qwen2_base":
                # Include the leading space in the response: Qwen BPE may join it
                # to the first answer token, which must remain supervised.
                prompt += " "
            prompt += assistant["value"] + ("\n" if self.template == "pretrain" else "<|endoftext|>")
            responses.append((start, len(prompt)))

        ids, labels, span = [], [], (-1, -1)
        offset = 0
        for part_index, part in enumerate(prompt.split("<image>")):
            if part_index:
                start = len(ids)
                ids.extend([self.tokenizer.pad_token_id] * self.shape[0])
                labels.extend([IGNORE] * self.shape[0])
                span = (start, len(ids))
                offset += len("<image>")
            encoded = self.tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
            ids.extend(encoded.input_ids)
            for token, (start, end) in zip(encoded.input_ids, encoded.offset_mapping):
                supervised = any(offset + start >= a and offset + end <= b and end > start
                                 for a, b in responses)
                labels.append(token if supervised else IGNORE)
            offset += len(part)
        if span[1] > self.max_length:
            raise ValueError(f"max_length={self.max_length} would truncate visual span {span}")
        ids, labels = ids[:self.max_length], labels[:self.max_length]
        if not any(label != IGNORE for label in labels[1:]):
            raise ValueError(f"no response targets remain at max_length={self.max_length}")
        return dict(input_ids=torch.tensor(ids), labels=torch.tensor(labels), feature=feature,
                    image=image_value, visual_span=span, record_id=source)


class Collator:
    def __init__(self, tokenizer):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token")
        self.pad = tokenizer.pad_token_id

    def __call__(self, rows):
        ids = torch.nn.utils.rnn.pad_sequence([r["input_ids"] for r in rows], True, self.pad)
        labels = torch.nn.utils.rnn.pad_sequence([r["labels"] for r in rows], True, IGNORE)
        positions = torch.arange(ids.shape[1])[None, :]
        attention = positions < torch.tensor([len(r["input_ids"]) for r in rows])[:, None]
        spans = torch.tensor([r["visual_span"] for r in rows])
        response = torch.zeros_like(attention)
        response[:, :-1] = labels[:, 1:] != IGNORE
        response &= attention
        write = response & (spans[:, 0:1] >= 0) & (positions >= spans[:, 1:2])
        return Batch(ids, labels, attention, tuple(r["feature"] for r in rows), spans,
                     response, write, tuple(r["record_id"] for r in rows),
                     images=tuple(r.get("image") for r in rows))


def attach_online_features(batch, encoder):
    """Encode the batch's PIL images with a frozen tower and return tensors only."""
    if not batch.images:
        return batch
    if len(batch.images) != len(batch.record_ids):
        raise ValueError("online image count does not match batch size")
    images = [image for image in batch.images if image is not None]
    encoded = encoder.encode_batch_train(images) if images else None
    if images and len(encoded) != len(images):
        raise ValueError(f"online encoder returned {len(encoded)} features for {len(images)} images")
    if images:
        features, cursor = [], 0
        for image in batch.images:
            if image is None:
                features.append(None)
            else:
                features.append(encoded[cursor])
                cursor += 1
        batch.features = tuple(features)
    batch.images = ()
    return batch
