"""Run with python -m vred.smoke; uses local assets only, including actual checkpoints."""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
from safetensors import safe_open
import torch
from torch.nn import functional as F
from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM

from .configs import load_config
from .data import CachedDataset, Collator
from .model import VREDModel, load_model, validate_tokenizers


def check_base_weights(config):
    """Establish that every adapter-targeted saved base weight equals the local base."""
    root = Path(config["teacher"])
    state = torch.load(root / "language_model/pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    base = Path(config["teacher_base"])
    mapping = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    count = 0
    for shard in set(mapping.values()):
        with safe_open(str(base / shard), framework="pt", device="cpu") as tensors:
            for key, value in state.items():
                if not key.endswith(".base_layer.weight"):
                    continue
                name = key.replace(".base_layer", "")
                if mapping[name] == shard:
                    assert torch.equal(value, tensors.get_tensor(name)), f"Teacher base differs: {root}/{key}"
                    count += 1
    assert count == 252, count
    return count


def check_errors(dataset, tokenizer):
    """Use temporary fixture files so diagnostics exercise real __getitem__ handling."""
    with tempfile.TemporaryDirectory(prefix="vred-errors-") as directory:
        root = Path(directory)
        record = copy.deepcopy(dataset.records[0])
        record["id"] = "bad-cache-smoke"
        path, index = root / "records.json", root / "index.json"

        def fails(row, cache, fragment, max_length=2048):
            path.write_text(json.dumps([row]))
            index.write_text(json.dumps(cache))
            test = CachedDataset(path, index, tokenizer, max_length=max_length)
            try:
                test[0]
            except ValueError as error:
                message = str(error)
                assert str(path) in message and "bad-cache-smoke" in message and fragment in message, message
            else:
                raise AssertionError(f"Expected failure: {fragment}")

        image = record["image"]
        fails(record, {}, str(index))
        fails(record, {image: "missing.npy"}, "missing.npy")
        np.save(root / "bad.npy", np.zeros((728, 1152), dtype=np.float16))
        fails(record, {image: "bad.npy"}, "bad.npy")
        array = np.zeros((729, 1152), dtype=np.float16)
        array[0, 0] = np.nan
        np.save(root / "bad.npy", array)
        fails(record, {image: "bad.npy"}, "finite")
        multiple = dict(record, image=[image, image])
        fails(multiple, {}, "one nonempty image path")
        multiple = copy.deepcopy(record)
        multiple["conversations"][0]["value"] += "<image>"
        fails(multiple, {}, "placeholder")
        missing = dict(record)
        del missing["image"]
        fails(missing, {}, "placeholder")
        np.save(root / "bad.npy", np.zeros((729, 1152), dtype=np.float16))
        fails(record, {image: "bad.npy"}, "truncate visual span", max_length=128)


def assert_no_vision(model):
    assert not hasattr(model, "vision_tower")
    assert all("vision" not in type(module).__name__.lower() for module in model.modules())
    assert not any(name.startswith("llavakd.model.vision_tower") for name in sys.modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--tiny-only", action="store_true", help="Skip supplied model weights, retaining real tokenizer/data/cache")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    start = time.monotonic()
    tokenizer = validate_tokenizers(config["student"], config["teacher"],
                                    AutoConfig.from_pretrained(config["student"]),
                                    AutoConfig.from_pretrained(Path(config["teacher"]) / "language_model"))
    collate = Collator(tokenizer)
    dpt = CachedDataset(config["dpt_json"], config["cache_index"], tokenizer, template="pretrain",
                        max_length=config["max_length"])
    dpt_count = len(dpt)
    caption = dpt[0]
    assert caption["feature"].device.type == "cpu" and caption["feature"].shape == (729, 1152)
    check_errors(dpt, tokenizer)
    del dpt
    gc.collect()
    sft = CachedDataset(config["sft_json"], config["cache_index"], tokenizer,
                        max_length=config["max_length"])
    text_index = next(i for i, row in enumerate(sft.records) if "image" not in row)
    image_index = next(i for i, row in enumerate(sft.records) if "image" in row)
    batch = collate([sft[text_index], sft[image_index]])
    boundary = sft._encode({"conversations": [
        {"from": "human", "value": "What color?"}, {"from": "gpt", "value": "blue"},
        {"from": "human", "value": "Again?"}, {"from": "gpt", "value": "red"}]}, "boundary-probe")
    boundary_batch = collate([boundary])
    targets = boundary_batch.labels[boundary_batch.labels != -100].tolist()
    assert targets == tokenizer.encode(" blue<|endoftext|> red<|endoftext|>", add_special_tokens=False)
    assert boundary_batch.response_positions[0, 1] + 1 == (boundary["labels"] != -100).nonzero()[0, 0]
    sft_count, cache_count = len(sft), len(sft.index)
    assert batch.features[0] is None and tuple(batch.visual_spans[0].tolist()) == (-1, -1)
    assert batch.visual_spans[1, 1] - batch.visual_spans[1, 0] == 729
    assert batch.features[1].device.type == "cpu"
    assert not batch.write_mask[0].any() and batch.write_mask[1].any()
    assert torch.equal(batch.response_mask[:, :-1], batch.labels[:, 1:] != -100)
    assert not batch.response_mask[:, -1].any()
    for row, prediction in batch.response_positions.tolist():
        assert batch.labels[row, prediction + 1] == batch.input_ids[row, prediction + 1]
    causal = batch.causal_mask
    assert not causal[0, 0].triu(1).any()
    assert not causal[0, 0, :, ~batch.attention_mask[0]].any()
    # Real EOS == PAD: valid EOS must still be attended and supervised.
    assert (batch.attention_mask & (batch.labels == tokenizer.eos_token_id)).any()
    del causal, sft
    gc.collect()
    tiny_config = Qwen2Config(**config["tiny_model"])
    tiny_config._attn_implementation = "eager"
    model = VREDModel(Qwen2ForCausalLM(tiny_config), config["vision_width"], config["vision_tokens"])
    model.language_model.requires_grad_(False)
    model.train()
    assert_no_vision(model)
    output = model(batch)
    loss = F.cross_entropy(output.logits[:, :-1].float().reshape(-1, tiny_config.vocab_size),
                           batch.labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(output.loss, loss)
    output.loss.backward()
    assert torch.isfinite(model.connector[0].weight.grad).all() and model.connector[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.language_model.parameters())
    del output, model, loss
    gc.collect()
    print("PASS real datasets/cache, strict errors, masks, tiny forward/backward, no vision", flush=True)
    results = dict(dpt_records=dpt_count, sft_records=sft_count, cache_entries=cache_count,
                   text_record=batch.record_ids[0], image_record=batch.record_ids[1],
                   batch_shape=list(batch.input_ids.shape), prediction_positions=int(batch.response_mask.sum()),
                   write_positions=int(batch.write_mask.sum()), tiny_only=args.tiny_only)
    if not args.tiny_only:
        dtype = getattr(torch, config["smoke_dtype"])
        student = load_model(config, dtype=dtype).eval()
        assert student.connector[0].weight.shape == (896, 1152)
        assert student.connector[2].weight.shape == (896, 896)
        assert_no_vision(student.train())
        student.eval()
        with torch.no_grad():
            output = student(batch)
            reference = output.logits[batch.response_mask].float().clone()
            results["student_loss"] = float(output.loss)
            del output
        assert torch.isfinite(reference).all()
        with tempfile.TemporaryDirectory(prefix="vred-reload-") as directory:
            student.save_pretrained(directory, tokenizer)
            del student
            gc.collect()
            reloaded = load_model(config, dtype=dtype, saved=directory).eval()
            with torch.no_grad():
                output = reloaded(batch)
                actual = output.logits[batch.response_mask].float()
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                results["student_reload_max_error"] = float((actual - reference).abs().max())
            del output, actual, reloaded, reference
        gc.collect()
        print("PASS supplied 0.5B student mixed forward and exact save/reload", flush=True)
        results["teacher_base_tensors_equal"] = check_base_weights(config)
        # FP32 gives a tight reference for explicit adapter vs CPU-merged weights.
        teacher = load_model(config, role="teacher", dtype=torch.float32, merge_teacher=False)
        assert teacher.connector[0].weight.shape == (2048, 1152)
        assert teacher.connector[2].weight.shape == (2048, 2048)
        assert_no_vision(teacher.train())
        assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
        # A short text probe compares full-vocabulary predictions, before image BF16 forward.
        probe = collate([dict(input_ids=torch.tensor(tokenizer.encode("USER: Hello ASSISTANT: Hi<|endoftext|>", add_special_tokens=False)),
                              labels=torch.tensor(tokenizer.encode("USER: Hello ASSISTANT: Hi<|endoftext|>", add_special_tokens=False)),
                              feature=None, visual_span=(-1, -1), record_id="teacher-merge-probe")])
        with torch.no_grad():
            reference = teacher(probe).logits.clone()
            teacher.merge_teacher_once()
            actual = teacher(probe).logits
        torch.testing.assert_close(actual, reference, atol=config["merge_atol"], rtol=config["merge_rtol"])
        results["teacher_merge_fp32_max_error"] = float((actual - reference).abs().max())
        try:
            teacher.merge_teacher_once()
        except ValueError as error:
            assert "twice" in str(error)
        else:
            raise AssertionError("Double merge was accepted")
        del actual, reference
        teacher.to(dtype=dtype)
        with torch.no_grad():
            output = teacher(batch)
            assert torch.isfinite(output.loss) and torch.isfinite(output.logits).all()
            results["teacher_loss"] = float(output.loss)
        assert all(p.grad is None for p in teacher.parameters())
        results["teacher_decision"] = teacher.load_decision
        del output, teacher
        print("PASS supplied frozen 3B teacher, base identity, FP32 LoRA merge, mixed forward", flush=True)
    results["elapsed_seconds"] = round(time.monotonic() - start, 2)
    results["device"] = "cpu"
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    # A full multimodal wrapper would use these factories for a vision tower.
    # Fail immediately if this foundation accidentally calls one of them.
    with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("Vision factory called")):
        main()
