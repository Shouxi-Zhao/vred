"""Explicit synthetic, local-only assets for pipeline integration checks."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from vred.configs import load_config, load_training_config
from vred.model import VREDModel


def make_assets(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    words = ["<|endoftext|>", "[UNK]", "USER", "ASSISTANT", ":", ".", "What", "is", "shown", "?", "A", "red", "square", "Describe", "the", "image", "Hello", "world"]
    words += [f"unused{i}" for i in range(128 - len(words))]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token=words[0], pad_token=words[0], unk_token="[UNK]")
    student_path, teacher_path = directory / "student", directory / "teacher"
    config = Qwen2Config(vocab_size=128, hidden_size=24, intermediate_size=48,
        num_hidden_layers=6, num_attention_heads=3, num_key_value_heads=1,
        max_position_embeddings=2048, tie_word_embeddings=True, use_sliding_window=False,
        bos_token_id=0, eos_token_id=0, pad_token_id=0)
    lm = Qwen2ForCausalLM(config)
    lm.save_pretrained(student_path)
    tokenizer.save_pretrained(student_path)
    teacher = VREDModel(Qwen2ForCausalLM(config))
    language = teacher_path / "language_model"
    language.mkdir(parents=True)
    teacher.language_model.config.save_pretrained(language)
    target = "model.layers.0.self_attn.q_proj"
    state = {key.replace(target + ".", target + ".base_layer."): value
             for key, value in teacher.language_model.state_dict().items()}
    torch.save(state, language / "pytorch_model.bin")
    (teacher_path / "connector").mkdir()
    torch.save({"_connector." + k: v for k, v in teacher.connector.state_dict().items()}, teacher_path / "connector/pytorch_model.bin")
    tokenizer.save_pretrained(teacher_path)
    (teacher_path / "adapter_config.json").write_text(json.dumps(dict(r=1, lora_alpha=1, bias="none", peft_type="LORA",
        target_modules=["language_model." + target])) + "\n")
    prefix = "base_model.model.language_model." + target
    save_file({prefix + ".lora_A.weight": torch.randn(1, 24) * 0.01,
               prefix + ".lora_B.weight": torch.randn(24, 1) * 0.01}, teacher_path / "adapter_model.safetensors")
    (directory / "synthetic").mkdir(exist_ok=True)
    Image.new("RGB", (8, 8), (128, 64, 32)).save(directory / "synthetic" / "square.jpg")
    cache = directory / "cache"
    cache.mkdir()
    np.save(cache / "square.npy", np.random.default_rng(1234).normal(size=(729, 1152)).astype(np.float16))
    (cache / "index.json").write_text(json.dumps({"synthetic/square.jpg": "square.npy"}) + "\n")
    rows = [dict(id="synthetic-image", image="synthetic/square.jpg", conversations=[
        dict(**{"from": "human", "value": "<image>\nWhat is shown?"}),
        dict(**{"from": "gpt", "value": "A red square."})]),
        dict(id="synthetic-text", conversations=[dict(**{"from": "human", "value": "Hello"}),
                                                dict(**{"from": "gpt", "value": "Hello world."})])]
    text_files = directory / "text_files"
    text_files.mkdir()
    for name in ("dpt", "sft"):
        (text_files / f"{name}.json").write_text(json.dumps(rows) + "\n")
    assets = load_config()
    assets.update(student=str(student_path), teacher=str(teacher_path), teacher_base=str(student_path),
                  dpt_json=str(text_files / "dpt.json"), sft_json=str(text_files / "sft.json"),
                  cache_index=str(cache / "index.json"), cpu_threads=2)
    asset_file = directory / "foundation.json"
    asset_file.write_text(json.dumps(assets, indent=2) + "\n")
    training = load_training_config()
    training.update(epochs=2, query_chunk_size=2, micro_batch_size=2, warmup_ratio=0)
    for settings in training["stages"].values():
        settings["accumulation"] = 1
    training_file = directory / "training.json"
    training_file.write_text(json.dumps(training, indent=2) + "\n")
    return str(asset_file), str(training_file)
