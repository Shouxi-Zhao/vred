"""Bounded BF16 DPT/SFT/DFT backward with real annotations and all 729 cache tokens."""

import argparse
import gc
import json
from pathlib import Path
import resource
import time

import torch
from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM

from .configs import load_config, load_training_config
from .data import CachedDataset, Collator
from .effects import EffectCapture, write_positions
from .model import VREDModel, load_model, validate_tokenizers
from .smoke import assert_no_vision
from .trainer import VREDTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-models", action="store_true", help="Use supplied 0.5B/3B weights instead of tiny widths")
    parser.add_argument("--no-probe-checkpoint", action="store_true", help="Memory comparison against the eager reference")
    parser.add_argument("--report", help="Write measurements as JSON")
    parser.add_argument("--diagnostics-only", action="store_true", help="Skip backward checks; compare BF16/FP32 reconstructions")
    args = parser.parse_args()
    assets, config = load_config(), load_training_config()
    shapes = json.loads((Path(__file__).with_name("configs") / "stage_smoke.json").read_text())
    config["checkpoint_probes"] = not args.no_probe_checkpoint
    torch.set_num_threads(assets["cpu_threads"])
    torch.manual_seed(assets["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = validate_tokenizers(assets["student"], assets["teacher"],
                                    AutoConfig.from_pretrained(assets["student"], local_files_only=True),
                                    AutoConfig.from_pretrained(Path(assets["teacher"]) / "language_model", local_files_only=True))
    collate = Collator(tokenizer)
    dataset = CachedDataset(assets["dpt_json"], assets["cache_index"], tokenizer, template="pretrain")
    dpt_batch = collate([dataset[0]]).to(device)
    del dataset
    gc.collect()
    dataset = CachedDataset(assets["sft_json"], assets["cache_index"], tokenizer)
    text_index = next(i for i, row in enumerate(dataset.records) if "image" not in row)
    mixed_batch = collate([dataset[text_index], dataset[0]]).to(device)
    del dataset
    gc.collect()
    if args.real_models:
        student = load_model(assets, dtype=torch.bfloat16, device=device)
        teacher = None if args.diagnostics_only else load_model(assets, role="teacher", dtype=torch.bfloat16, device=device)
    else:
        models = []
        for role in ("student", "teacher"):
            shape = Qwen2Config(vocab_size=151936, **shapes[role], use_sliding_window=False)
            shape._attn_implementation = "eager"
            models.append(VREDModel(Qwen2ForCausalLM(shape), 1152, 729).to(device=device, dtype=torch.bfloat16))
        student, teacher = models
    assert_no_vision(student)
    if teacher is not None:
        assert_no_vision(teacher)
    reports = []
    for stage in (() if args.diagnostics_only else ("dpt", "sft", "dft")):
        student.zero_grad(set_to_none=True)
        teacher.zero_grad(set_to_none=True)
        batch = dpt_batch if stage == "dpt" else mixed_batch
        trainer = VREDTrainer(student, None if stage == "sft" else teacher, stage, config)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.monotonic()
        loss, metrics = trainer.loss(batch)
        loss.backward()
        assert torch.isfinite(loss)
        connector_grad = float(student.connector[0].weight.grad.float().norm())
        assert connector_grad > 0
        assert all(p.grad is None for p in teacher.parameters())
        decoder = student.language_model.model.layers[0].mlp.up_proj.weight
        decoder_grad = float(decoder.grad.float().norm()) if decoder.grad is not None else None
        if stage == "dpt":
            assert all(p.grad is None for p in student.language_model.parameters())
        else:
            assert decoder_grad > 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        report = {key: float(value) if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
        report.update(stage=stage, device=str(device), dtype="bfloat16", real_models=args.real_models,
                      checkpoint_probes=config["checkpoint_probes"], pairs=trainer.pairs,
                      batch_shape=list(batch.input_ids.shape), connector_grad_norm=connector_grad,
                      decoder_grad_norm=decoder_grad, seconds=elapsed,
                      valid_tokens_per_second=metrics["response_tokens"] / elapsed,
                      process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None)
        reports.append(report)
        print(json.dumps(report), flush=True)
        if args.report:
            Path(args.report).write_text(json.dumps(dict(stages=reports), indent=2) + "\n")
        del loss, metrics
        gc.collect()
    student.zero_grad(set_to_none=True)
    # Use a normwise criterion for reassociation: near-zero coordinates can have
    # large relative cancellation error even when the whole state agrees closely.
    # Ordinary forward equality remains bit-exact, independent of this diagnostic.
    diagnostic = {}
    for name in ("bfloat16", "float32"):
        student.to(dtype=getattr(torch, name)).eval()
        with torch.no_grad():
            reference = student(mixed_batch).logits
            with EffectCapture(student, [23], mixed_batch, write_positions(mixed_batch)) as capture:
                actual = student(mixed_batch).logits
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            effect = capture.effects[23]
            layer = student.language_model.model.layers[23]
            baseline = effect.incoming + effect.o_text
            plus = baseline + effect.o_vision
            reconstructed = plus + layer.mlp(layer.post_attention_layernorm(plus))
            difference = reconstructed.float() - effect.u_plus.float()
            relative = difference.norm(dim=-1) / effect.u_plus.float().norm(dim=-1).clamp_min(1e-12)
            maximum = float(relative.max())
            diagnostic[name] = dict(block_reconstruction_max_absolute_error=float(difference.abs().max()),
                                    block_reconstruction_max_relative_l2=maximum,
                                    relative_l2_tolerance=shapes["reconstruction_relative_l2"][name],
                                    ordinary_forward_max_error=0.0)
            assert maximum <= shapes["reconstruction_relative_l2"][name], diagnostic[name]
        del reference, actual, capture, effect, reconstructed
        gc.collect()
    print(json.dumps(diagnostic), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(dict(stages=reports, diagnostics=diagnostic), indent=2) + "\n")


if __name__ == "__main__":
    main()
