"""Shared, explicit training overrides for the runner and six-GPU launcher."""

import argparse
import math
from pathlib import Path


COMMON = {
    "epochs": int, "micro_batch_size": int, "num_workers": int,
    "weight_decay": float, "max_grad_norm": float, "warmup_ratio": float,
    "save_every_steps": int, "query_chunk_size": int, "ce_chunk_size": int, "probe_vocab_chunk_size": int,
    "tau_write": float, "tau_logits": float, "smoothl1_beta": float,
    "effect_kl_eta": float,
}
STAGE = {"lr": float, "accumulation": int, "pairs": int,
         "lambda_write": float, "lambda_logits": float}
ASSET = {"seed": int, "max_length": int, "cpu_threads": int,
         "student": str, "teacher": str, "teacher_base": str,
         "dpt_json": str, "sft_json": str, "cache_index": str,
         "online_vision_root": str}


def add_training_overrides(parser):
    for name, kind in {**COMMON, **STAGE, **ASSET}.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=None,
                            help=f"Override {name} from JSON")
    parser.add_argument("--write-loss-type", choices=("smoothl1", "effect_kl", "effect_js", "reverse_kl"))
    parser.add_argument("--write-vocab-reduction", choices=("sum", "mean"))
    parser.add_argument("--write-position-scope", choices=("response", "prompt", "prompt_response"),
                        help="Where to apply the DFT write loss (default: response)")
    parser.add_argument("--write-effect-sampling", choices=("uniform", "stratified"),
                        help="Position sampling inside the scope (default: uniform)")
    parser.add_argument("--write-lambda-schedule", choices=("constant", "warmup_cosine"),
                        help="Scale lambda_write over optimizer steps (default: constant)")
    parser.add_argument("--reverse-kl", action=argparse.BooleanOptionalAction, default=None,
                        help="Use reverse KL for the write effect loss (alias for --write-loss-type reverse_kl)")
    parser.add_argument("--online-vision", action=argparse.BooleanOptionalAction, default=None,
                        help="Load and freeze the SigLIP2 vision tower during training instead of cached features")
    parser.add_argument("--write-position-limit", type=int, help="Write positions per example; 0 uses all")
    parser.add_argument("--effect-weighted-kd", action=argparse.BooleanOptionalAction, default=None,
                        help="Add effect-weighted final-logit KD on response positions (DFT only)")
    parser.add_argument("--lambda-effect-kd", type=float,
                        help="Coefficient for the effect-weighted KD term (default: 1.0)")
    parser.add_argument("--effect-kd-weight-source", choices=("logits", "hidden"),
                        help="Effect-KD weight source: logits difference (default) or teacher hidden effect")
    parser.add_argument("--effect-kd-floor", type=float,
                        help="Floor for hidden-effect KD weights in [0,1); default 0.1")
    parser.add_argument("--checkpoint-probes", action=argparse.BooleanOptionalAction, default=None,
                        help="Recompute probe losses in backward to reduce memory")
    parser.add_argument("--sparse-logits", action=argparse.BooleanOptionalAction, default=None,
                        help="CE/final KD only at shifted response positions (default: enabled)")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable student backbone checkpointing (SFT only; default: disabled)")
    parser.add_argument("--selective-checkpointing", action=argparse.BooleanOptionalAction, default=None,
                        help="Checkpoint non-selected student blocks in DPT/DFT; selected effect blocks stay ordinary")
    parser.add_argument("--max-steps", type=int, help="Absolute optimizer-step cap for a bounded run")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable TensorBoard monitoring (default: enabled)")
    parser.add_argument("--tensorboard-log-dir", help="Event directory; defaults to OUTPUT/tensorboard")


def apply_training_overrides(args, config, assets, stage):
    for name in COMMON:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    for name in STAGE:
        if getattr(args, name) is not None:
            config["stages"][stage][name] = getattr(args, name)
    for name, kind in ASSET.items():
        value = getattr(args, name)
        if value is not None:
            if kind is str:
                value = str((Path(__file__).resolve().parents[1] / value).resolve())
            assets[name] = value
    for name in ("write_loss_type", "write_vocab_reduction", "checkpoint_probes",
                 "sparse_logits", "gradient_checkpointing", "selective_checkpointing",
                 "write_position_scope", "write_effect_sampling", "write_lambda_schedule"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.online_vision is not None:
        config["online_vision"] = bool(args.online_vision)
    if args.reverse_kl is not None:
        if args.reverse_kl:
            if args.write_loss_type not in (None, "reverse_kl"):
                raise ValueError("--reverse-kl conflicts with --write-loss-type "
                                 f"{args.write_loss_type}; choose one reverse-KL spelling")
            config["write_loss_type"] = "reverse_kl"
        elif args.write_loss_type == "reverse_kl":
            raise ValueError("--no-reverse-kl conflicts with --write-loss-type reverse_kl")
    if args.write_position_limit is not None:
        if args.write_position_limit < 0:
            raise ValueError("--write-position-limit must be nonnegative (0 uses all)")
        config["write_position_limit"] = args.write_position_limit or None
    if args.effect_weighted_kd is not None:
        config["effect_weighted_kd"] = bool(args.effect_weighted_kd)
    if args.lambda_effect_kd is not None:
        config["lambda_effect_kd"] = args.lambda_effect_kd
    if config.get("effect_weighted_kd"):
        config.setdefault("lambda_effect_kd", 1.0)
        if stage != "dft":
            raise ValueError("--effect-weighted-kd is only supported for DFT")
        if not config.get("sparse_logits", True):
            raise ValueError("--effect-weighted-kd requires sparse logits")
    elif args.lambda_effect_kd is not None:
        raise ValueError("--lambda-effect-kd requires --effect-weighted-kd")
    if args.effect_kd_weight_source is not None:
        if not config.get("effect_weighted_kd"):
            raise ValueError("--effect-kd-weight-source requires --effect-weighted-kd")
        config["effect_kd_weight_source"] = args.effect_kd_weight_source
    source = config.get("effect_kd_weight_source", "logits")
    if source not in ("logits", "hidden"):
        raise ValueError("effect_kd_weight_source must be 'logits' or 'hidden'")
    if source == "hidden":
        if not config.get("effect_weighted_kd"):
            raise ValueError("hidden effect KD requires --effect-weighted-kd")
        if stage != "dft":
            raise ValueError("hidden effect KD is only supported for DFT")
        if not config.get("sparse_logits", True):
            raise ValueError("hidden effect KD requires sparse logits")
        if config.get("write_position_scope", "response") != "response":
            raise ValueError("hidden effect KD requires write_position_scope=response")
        if config.get("write_position_limit") is not None:
            raise ValueError("hidden effect KD requires all response positions; omit --write-position-limit")
        if config.get("write_effect_sampling", "uniform") != "uniform":
            raise ValueError("hidden effect KD requires write_effect_sampling=uniform")
        if args.effect_kd_floor is not None:
            config["effect_kd_floor"] = args.effect_kd_floor
        config.setdefault("effect_kd_floor", 0.1)
        floor = config["effect_kd_floor"]
        if not math.isfinite(floor) or not 0 <= floor < 1:
            raise ValueError("--effect-kd-floor must be finite and in [0,1)")
    elif args.effect_kd_floor is not None:
        raise ValueError("--effect-kd-floor requires --effect-kd-weight-source hidden")
    if (config.get("write_effect_sampling") == "stratified"
            and config.get("write_position_limit") is None):
        raise ValueError("write_effect_sampling=stratified requires write_position_limit")
    settings = config["stages"][stage]
    positive = {name: config[name] for name in ("epochs", "micro_batch_size", "save_every_steps",
        "query_chunk_size", "probe_vocab_chunk_size", "max_grad_norm", "tau_write", "tau_logits", "smoothl1_beta")}
    positive.update(lr=settings["lr"], accumulation=settings["accumulation"],
                    max_length=assets["max_length"], cpu_threads=assets["cpu_threads"])
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config["ce_chunk_size"] < 0:
        raise ValueError("ce_chunk_size must be nonnegative (0 uses all response positions)")
    for name, value in dict(weight_decay=config["weight_decay"], num_workers=config["num_workers"],
                            lambda_write=settings["lambda_write"], lambda_logits=settings["lambda_logits"]).items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= config["warmup_ratio"] <= 1 or not 0 < config["effect_kl_eta"] < 1:
        raise ValueError("warmup_ratio must be in [0,1]; effect_kl_eta must be in (0,1)")
    if not math.isfinite(config.get("lambda_effect_kd", 0.0)) or config.get("lambda_effect_kd", 0.0) < 0:
        raise ValueError("lambda_effect_kd must be finite and nonnegative")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if stage == "sft":
        if settings["lambda_write"] or settings["lambda_logits"] or settings["pairs"]:
            raise ValueError("SFT requires zero write/final-KL coefficients and zero pairs")
    elif settings["pairs"] < 1:
        raise ValueError("DPT/DFT require a positive layer-pair count")
    if stage == "dpt" and settings["lambda_logits"]:
        raise ValueError("DPT must not include final-logit KD")


def forwarded_overrides(args):
    result = []
    for name in (*COMMON, *STAGE, *ASSET, "write_loss_type", "write_vocab_reduction",
                 "write_position_scope", "write_effect_sampling", "write_lambda_schedule",
                 "write_position_limit", "lambda_effect_kd", "effect_kd_weight_source",
                 "effect_kd_floor", "max_steps", "tensorboard_log_dir"):
        value = getattr(args, name)
        if value is not None:
            result.extend(("--" + name.replace("_", "-"), str(value)))
    if args.effect_weighted_kd is not None:
        result.append("--effect-weighted-kd" if args.effect_weighted_kd else "--no-effect-weighted-kd")
    if args.checkpoint_probes is not None:
        result.append("--checkpoint-probes" if args.checkpoint_probes else "--no-checkpoint-probes")
    if args.sparse_logits is not None:
        result.append("--sparse-logits" if args.sparse_logits else "--no-sparse-logits")
    if args.gradient_checkpointing is not None:
        result.append("--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing")
    if args.selective_checkpointing is not None:
        result.append("--selective-checkpointing" if args.selective_checkpointing else "--no-selective-checkpointing")
    if args.tensorboard is not None:
        result.append("--tensorboard" if args.tensorboard else "--no-tensorboard")
    if args.reverse_kl is not None:
        result.append("--reverse-kl" if args.reverse_kl else "--no-reverse-kl")
    if args.online_vision is not None:
        result.append("--online-vision" if args.online_vision else "--no-online-vision")
    return result
