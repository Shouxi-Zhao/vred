"""Shared SigLIP2 encoder for evaluation and opt-in online training.

Cached training remains the default: ``vred.train`` only imports this module
when ``--online-vision`` is supplied, and neither ``vred.model`` nor
``load_model`` loads a vision tower.  Evaluation can read precomputed features
(``--cache-index``) or encode images at generation time.  Online training uses
the same frozen tower and preprocessing, but features must be produced under
``torch.no_grad`` (not ``inference_mode``) so the connector can backpropagate
through them.  The preprocessing and patch-token selection are shared with
``cache_vision_features.py`` so online and cached features are directly
comparable.
"""

import base64
import io
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageFile
from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

ImageFile.LOAD_TRUNCATED_IMAGES = True


def infer_expected_tokens(model_config, image_processor) -> Optional[int]:
    vision_config = getattr(model_config, "vision_config", model_config)
    patch_size = getattr(vision_config, "patch_size", None)
    image_size = getattr(vision_config, "image_size", None)
    if image_size is None:
        size = getattr(image_processor, "crop_size", None) or getattr(image_processor, "size", None)
        if isinstance(size, dict):
            image_size = size.get("height") or size.get("shortest_edge")
        elif isinstance(size, int):
            image_size = size
    if patch_size is None or image_size is None:
        return None
    grid = int(round(image_size / patch_size))
    return grid * grid


def select_patch_features(hidden_state: torch.Tensor, expected_tokens: Optional[int]) -> torch.Tensor:
    if expected_tokens is not None:
        if hidden_state.shape[1] == expected_tokens:
            return hidden_state
        if hidden_state.shape[1] == expected_tokens + 1:
            return hidden_state[:, 1:]
    return hidden_state[:, 1:] if hidden_state.shape[1] > 1 else hidden_state


def resolve_image_path(image_root, image) -> str:
    """Resolve a benchmark-relative image exactly like the training cache builder."""
    if isinstance(image, Path):
        image = str(image)
    if not isinstance(image, str):
        raise TypeError(f"Image must be a path string, got {type(image).__name__}")
    if os.path.isabs(image):
        if os.path.exists(image):
            return image
        raise FileNotFoundError(f"Image not found: {image}")
    if image_root is None:
        candidates = [image]
    else:
        root = Path(image_root)
        candidates = [str(root / image), str(root / "llava" / "images" / image),
                      str(root.parent / "llava" / "images" / image)]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find image '{image}' under any candidate path: {candidates}".format(
            image=image, candidates=", ".join(candidates)))


def open_image(image_root, image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(resolve_image_path(image_root, image)) as handle:
        return handle.convert("RGB")


def decode_base64_image(value) -> Image.Image:
    """Decode an embedded MMBench TSV image, accepting both raw and data-URI base64."""
    if isinstance(value, bytes):
        payload = value
    else:
        text = str(value).strip()
        if text.startswith("data:") and "," in text:
            text = text.split(",", 1)[1]
        payload = base64.b64decode(text)
    with Image.open(io.BytesIO(payload)) as handle:
        return handle.convert("RGB")


class OnlineVisionEncoder:
    """Load SigLIP2 once, freeze it, and encode images for evaluation or online training."""

    def __init__(self, vision_tower, device="cuda:0", dtype=torch.float16, feature_layer=-1):
        self.device = torch.device(device)
        self.dtype = dtype
        self.feature_layer = feature_layer
        config = AutoConfig.from_pretrained(vision_tower, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_tower, local_files_only=True)
        self.expected_tokens = infer_expected_tokens(config, self.image_processor)
        self.model = SiglipVisionModel.from_pretrained(
            vision_tower, local_files_only=True, torch_dtype=dtype)
        self.model = self.model.eval().to(device=self.device, dtype=dtype)
        self.model.requires_grad_(False)
        self.vision_width = getattr(getattr(config, "vision_config", config), "hidden_size", None)

    def _encode_batch(self, images) -> torch.Tensor:
        if not images:
            tokens = self.expected_tokens or 0
            width = self.vision_width or 0
            return torch.empty((0, tokens, width), device=self.device, dtype=self.dtype)
        prepared = [open_image(None, image) if not isinstance(image, Image.Image) else image.convert("RGB")
                    for image in images]
        inputs = self.image_processor(images=prepared, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_state = outputs.hidden_states[self.feature_layer]
        return select_patch_features(hidden_state, self.expected_tokens)

    @torch.inference_mode()
    def encode_batch(self, images) -> torch.Tensor:
        return self._encode_batch(images)

    @torch.no_grad()
    def encode_batch_train(self, images) -> torch.Tensor:
        """Return detached ordinary tensors that the connector can consume in autograd."""
        return self._encode_batch(images).detach()

    @torch.inference_mode()
    def encode(self, image) -> torch.Tensor:
        return self.encode_batch([image])[0]


__all__ = ["OnlineVisionEncoder", "decode_base64_image", "infer_expected_tokens",
           "open_image", "resolve_image_path", "select_patch_features"]
