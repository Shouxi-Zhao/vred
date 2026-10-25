"""BF16 DPT/SFT/DFT entry point with local AdamW or DeepSpeed ZeRO-2."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from .configs import load_config, load_training_config
from .data import CachedDataset, Collator, attach_online_features
from .model import load_model, validate_tokenizers
from .trainer import VREDTrainer, accumulation_groups
from .preflight import audit_dataset, require_coverage
from .options import add_training_overrides, apply_training_overrides


def zero2_config(config, stage, world_size):
    path = Path(__file__).with_name("configs") / "zero2.json"
    resolved = json.loads(path.read_text())
    micro = config["micro_batch_size"]
    accumulation = config["stages"][stage]["accumulation"]
    resolved.update(train_micro_batch_size_per_gpu=micro, gradient_accumulation_steps=accumulation,
                    train_batch_size=world_size * micro * accumulation, gradient_clipping=config["max_grad_norm"])
    return resolved


def make_online_encoder(assets, device):
    """Load the frozen SigLIP2 tower used only by the opt-in online training path."""
    from .vision import OnlineVisionEncoder
    vision_tower = str(assets["vision_tower"])
    if not Path(vision_tower).is_absolute():
        vision_tower = str(Path(__file__).resolve().parents[1] / vision_tower)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return OnlineVisionEncoder(vision_tower, device=device, dtype=dtype)


def online_image_root(data_path, assets):
    root = assets.get("online_vision_root")
    if root is None:
        root = Path(data_path).resolve().parent.parent
    else:
        root = Path(root)
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
    return str(root)


def save_training_checkpoint(student, tokenizer, optimizer, scheduler, engine, output, rank, state):
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "checkpoint_complete.json"
    if rank == 0:
        marker.unlink(missing_ok=True)
    if engine:
        torch.distributed.barrier()
    if not engine:
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    torch.save(state, output / f"trainer_state_rank{rank}.pt")
    if engine:
        engine.save_checkpoint(str(output / "deepspeed"), tag=f"step-{state['steps']}",
                               client_state={"steps": state["steps"]})
        torch.distributed.barrier()
    if rank == 0:
        student.save_pretrained(output, tokenizer)
        marker.write_text(json.dumps(dict(steps=state["steps"])) + "\n")
    if engine:
        torch.distributed.barrier()


def _main(resources):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("dpt", "sft", "dft"))
    parser.add_argument("--config", help="Training JSON; defaults to vred/configs/training.json")
    parser.add_argument("--foundation-config")
    parser.add_argument("--input-model", help="Saved VRED checkpoint from the previous stage")
    parser.add_argument("--resume", help="Resume this stage from a VRED output directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("torch", "deepspeed"), default="torch")
    add_training_overrides(parser)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()
    writer_type = None
    if args.tensorboard is not False:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            parser.error("TensorBoard requires tensorboard; run 'python -m pip install tensorboard' or use --no-tensorboard")
        writer_type = SummaryWriter
    if args.resume and not (Path(args.resume) / "checkpoint_complete.json").is_file():
        parser.error("Resume checkpoint is missing its completion marker; export may be incomplete")
    if args.stage != "dpt" and args.input_model is None and args.resume is None:
        parser.error("SFT/DFT require --input-model from the previous stage")
    config, assets = load_training_config(args.config), load_config(args.foundation_config)
    try:
        apply_training_overrides(args, config, assets, args.stage)
    except ValueError as error:
        parser.error(str(error))
    if config.get("precision", "bfloat16") != "bfloat16":
        parser.error("VRED training requires bfloat16 model computation")
    torch.manual_seed(assets["seed"])
    torch.set_num_threads(assets["cpu_threads"])
    rank, world = 0, 1
    if args.backend == "deepspeed":
        if not torch.cuda.is_available():
            raise RuntimeError("DeepSpeed ZeRO-2 requires CUDA; use --backend torch for CPU checks")
        import deepspeed
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed(dist_backend="nccl")
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    elif int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Use --backend deepspeed for distributed training")
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    tokenizer = validate_tokenizers(assets["student"], assets["teacher"],
                                    AutoConfig.from_pretrained(assets["student"], local_files_only=True),
                                    AutoConfig.from_pretrained(Path(assets["teacher"]) / "language_model", local_files_only=True))
    data_path = assets["dpt_json"] if args.stage == "dpt" else assets["sft_json"]
    online = bool(config.get("online_vision", False))
    image_root = online_image_root(data_path, assets) if online else None
    dataset = CachedDataset(data_path, assets["cache_index"], tokenizer,
                            template="pretrain" if args.stage == "dpt" else "qwen2_base",
                            max_length=assets["max_length"], vision_tokens=assets["vision_tokens"],
                            vision_width=assets["vision_width"], online=online, image_root=image_root)
    coverage = audit_dataset(dataset)
    require_coverage(coverage)
    student = load_model(assets, saved=args.resume or args.input_model, device=device, dtype=torch.bfloat16)
    if config.get("gradient_checkpointing", False):
        if args.stage != "sft":
            parser.error("gradient_checkpointing is supported for SFT only; DPT/DFT need the ordinary backbone forward for effect capture")
        student.language_model.gradient_checkpointing_enable()
        student.language_model.config.use_cache = False
    if config.get("selective_checkpointing", False) and args.stage == "sft":
        parser.error("selective_checkpointing is for DPT/DFT; use gradient_checkpointing for SFT")
    teacher = None if args.stage == "sft" else load_model(assets, role="teacher", device=device, dtype=torch.bfloat16)
    vision_encoder = make_online_encoder(assets, device) if online else None
    trainer = VREDTrainer(student, teacher, args.stage, config, assets["seed"] + rank)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=assets["seed"], shuffle=True)
    loader = DataLoader(dataset, batch_size=config["micro_batch_size"], sampler=sampler,
                        collate_fn=Collator(tokenizer), num_workers=config["num_workers"])
    settings = config["stages"][args.stage]
    accumulation = settings["accumulation"]
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = min(config["epochs"] * steps_per_epoch, args.max_steps or math.inf)
    trainer.set_write_schedule(total_steps)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                  lr=settings["lr"], weight_decay=config["weight_decay"])
    planned_steps = config["epochs"] * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(planned_steps * config["warmup_ratio"]), planned_steps)
    engine = None
    resolved = zero2_config(config, args.stage, world)
    if args.backend == "deepspeed":
        engine, optimizer, _, scheduler = deepspeed.initialize(model=student, optimizer=optimizer,
                                                               lr_scheduler=scheduler, config=resolved)
    resume_state = None
    if args.resume:
        previous = json.loads((Path(args.resume) / "run_config.json").read_text())
        for key, value in (("stage", args.stage), ("training", config), ("assets", assets),
                           ("backend", args.backend), ("world_size", world)):
            if previous[key] != value:
                raise ValueError(f"Resume {key} differs from saved run")
        resume_state = torch.load(Path(args.resume) / f"trainer_state_rank{rank}.pt", map_location="cpu", weights_only=False)
        completed = json.loads((Path(args.resume) / "checkpoint_complete.json").read_text())
        if completed["steps"] != resume_state["steps"]:
            raise ValueError("Resume trainer state and completed export have different step counts")
        if engine:
            loaded, _ = engine.load_checkpoint(str(Path(args.resume) / "deepspeed"))
            if loaded is None:
                raise ValueError(f"Cannot load DeepSpeed checkpoint: {args.resume}")
        else:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
        torch.set_rng_state(resume_state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume_state["cuda_rng"], device)
        trainer.generator.set_state(resume_state["position_rng"])
    output = Path(args.output)
    tensorboard_dir = Path(args.tensorboard_log_dir) if args.tensorboard_log_dir else output / "tensorboard"
    if engine:
        torch.distributed.barrier()
    if rank == 0:
        resuming_in_place = args.resume and output.resolve() == Path(args.resume).resolve()
        if (output / "run_config.json").exists() and not resuming_in_place:
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(json.dumps(dict(
            stage=args.stage, training=config, assets=assets, input_model=args.input_model,
            backend=args.backend, zero2=resolved if engine else None, layer_pairs=trainer.pairs,
            dtype="bfloat16", device=str(device), world_size=world, max_steps=args.max_steps,
            resume=args.resume, seed=assets["seed"], coverage=coverage,
            tensorboard=dict(enabled=writer_type is not None, log_dir=str(tensorboard_dir.resolve())),
            tokenizer_ids={name: getattr(tokenizer, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")},
            tokenizer_size=len(tokenizer), model_vocab_size=student.language_model.config.vocab_size,
            loss_reduction="valid prediction tokens across ranks/accumulation; write mean over layer pairs",
            write_vocab_reduction=config["write_vocab_reduction"],
            **({"online_vision": True, "online_image_root": image_root} if online else {}),
            trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in student.parameters()),
            trainable_modules=[name for name, module in student.named_modules()
                               if any(p.requires_grad for p in module.parameters(recurse=False))]), indent=2) + "\n")
        print(json.dumps(dict(stage=args.stage, coverage=coverage, layer_pairs=trainer.pairs,
                              seed=assets["seed"], global_batch=world * config["micro_batch_size"] * accumulation,
                              online_vision=online,
                              tensorboard_log_dir=str(tensorboard_dir.resolve()) if writer_type else None)), flush=True)
    if engine:
        torch.distributed.barrier()
    optimizer.zero_grad()
    steps, examples, tokens, microsteps = 0, 0, 0, 0
    start_epoch, cursor = 0, 0
    if resume_state:
        steps, examples, tokens, microsteps = (resume_state[k] for k in ("steps", "examples", "tokens", "microsteps"))
        start_epoch, cursor = resume_state["epoch"], resume_state["cursor"]
    if steps >= total_steps:
        raise ValueError("Resume has already reached --max-steps/epoch budget; increase --max-steps")
    writer = None
    if rank == 0 and writer_type is not None:
        writer = resources.enter_context(writer_type(log_dir=str(tensorboard_dir),
                                                     purge_step=steps + 1 if args.resume else 0, flush_secs=30))
        writer.add_text("run/config", (output / "run_config.json").read_text(), steps)

    def save(epoch, cursor):
        state = dict(steps=steps, examples=examples, tokens=tokens, microsteps=microsteps,
                     epoch=epoch, cursor=cursor, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                     position_rng=trainer.generator.get_state())
        save_training_checkpoint(student, tokenizer, optimizer, scheduler, engine, output, rank, state)
        if writer is not None:
            writer.flush()

    progress = tqdm(total=total_steps, initial=steps, desc=f"{args.stage.upper()} training",
                    unit="step", dynamic_ncols=True, disable=rank != 0)
    for epoch in range(start_epoch, config["epochs"]):
        sampler.set_epoch(epoch)
        from itertools import islice
        # A dedicated loader generator avoids consuming the restored model RNG.
        loader.generator = torch.Generator().manual_seed(assets["seed"] + epoch)
        iterator = islice(loader, cursor, None)
        groups = accumulation_groups(iterator, accumulation, config["write_position_limit"], device,
                                     config.get("write_position_scope", "response"))
        for group, normalization in groups:
            group_started = time.monotonic()
            group_metrics = {}
            for index, batch in enumerate(group):
                batch = batch.to(device)
                if vision_encoder is not None:
                    batch = attach_online_features(batch, vision_encoder)
                step_start = time.monotonic()
                trainer.set_write_step(steps)
                loss, metrics = trainer.loss(batch, forward_model=engine, normalization_counts=normalization)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {args.stage} loss for {batch.record_ids}")
                group_size = len(group)
                boundary = index + 1 == group_size
                if engine:
                    engine.set_gradient_accumulation_boundary(boundary)
                    engine.backward(loss * accumulation / group_size)
                    engine.step()
                else:
                    (loss / group_size).backward()
                    if boundary:
                        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], config["max_grad_norm"])
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                examples += len(batch.record_ids)
                tokens += metrics["response_tokens"]
                microsteps += 1
                cursor += 1
                if boundary:
                    steps += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.monotonic() - step_start
                log = {key: float(value) if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
                global_counts = torch.tensor([examples, tokens], device=device, dtype=torch.long)
                if engine:
                    torch.distributed.all_reduce(global_counts)
                    names = ("loss", "ce", "write_raw", "write", "write_weighted", "logits_kd", "logits_kd_weighted")
                    means = torch.tensor([log[name] for name in names], device=device)
                    torch.distributed.all_reduce(means)
                    for name, value in zip(names, means / world):
                        log[name] = float(value)
                log.update(epoch=epoch, optimizer_steps=steps, microsteps=microsteps, local_examples=examples,
                           local_response_tokens=tokens, microstep_seconds=seconds,
                           global_examples=int(global_counts[0]), global_response_tokens=int(global_counts[1]),
                           valid_tokens_per_second=metrics["response_tokens"] / seconds,
                           peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                           peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                           peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None)
                if rank == 0:
                    progress.set_postfix(epoch=f"{epoch + 1}/{config['epochs']}", loss=f"{log['loss']:.4f}",
                                         lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                         tokens_per_s=f"{log['valid_tokens_per_second']:.1f}", refresh=False)
                    progress.update(int(boundary))
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(log) + "\n")
                    if writer is not None:
                        for name in ("loss", "ce", "write_raw", "write", "write_weighted",
                                     "logits_kd", "logits_kd_weighted"):
                            group_metrics[name] = group_metrics.get(name, 0.0) + log[name] / group_size
                        if boundary:
                            for name, value in group_metrics.items():
                                writer.add_scalar(f"train/{name}", value, steps)
                            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], steps)
                            writer.add_scalar("train/epoch", epoch + cursor / len(loader), steps)
                            group_seconds = time.monotonic() - group_started
                            writer.add_scalar("speed/optimizer_step_seconds", group_seconds, steps)
                            writer.add_scalar("speed/global_tokens_per_second", normalization[0] / group_seconds, steps)
                            for name in ("global_examples", "global_response_tokens", "microsteps"):
                                writer.add_scalar(f"progress/{name}", log[name], steps)
                            for name in ("peak_rss_mib", "peak_allocated_mib", "peak_reserved_mib"):
                                if log[name] is not None:
                                    writer.add_scalar(f"memory/{name}", log[name], steps)
                if boundary and steps % config.get("save_every_steps", 1000) == 0 and steps < total_steps:
                    save(epoch + 1 if cursor == len(loader) else epoch, 0 if cursor == len(loader) else cursor)
                del loss, metrics
                if steps >= total_steps:
                    break
            if steps >= total_steps:
                break
        if steps >= total_steps:
            break
        cursor = 0
    progress.close()
    next_epoch = config["epochs"] if cursor == 0 else (epoch + 1 if cursor == len(loader) else epoch)
    next_cursor = 0 if cursor == len(loader) else cursor
    save(next_epoch, next_cursor)
    if rank == 0:
        elapsed = time.monotonic() - started
        global_counts = torch.tensor([examples, tokens], device=device, dtype=torch.long) if not engine else global_counts
        (output / "summary.json").write_text(json.dumps(dict(
            stage=args.stage, optimizer_steps=steps, elapsed_seconds=elapsed,
            local_examples=examples, local_response_tokens=tokens,
            global_examples=int(global_counts[0]), global_response_tokens=int(global_counts[1]),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            gpu_hours=elapsed * world / 3600 if device.type == "cuda" else 0,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None), indent=2) + "\n")
    if engine:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main():
    with ExitStack() as resources:
        _main(resources)


if __name__ == "__main__":
    main()
