"""Launch one explicit stage profile from a clean repository-root shell."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .configs import load_config, load_training_config
from .options import add_training_overrides, apply_training_overrides, forwarded_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("dpt", "sft", "dft"))
    parser.add_argument("--stage", dest="stage_option", choices=("dpt", "sft", "dft"),
                        help="Stage name; equivalent to the positional stage argument")
    parser.add_argument("--profile")
    parser.add_argument("--resume", help="Override the profile's same-stage resume path")
    parser.add_argument("--gpus", type=int, help="Number of local GPU processes")
    parser.add_argument("--config", help="Override training JSON path")
    parser.add_argument("--foundation-config", help="Override asset JSON path")
    parser.add_argument("--input-model", help="Override predecessor checkpoint")
    parser.add_argument("--output", help="Override stage output directory")
    add_training_overrides(parser)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage and args.stage_option and args.stage != args.stage_option:
        parser.error("Positional stage and --stage disagree")
    args.stage = args.stage_option or args.stage
    if args.stage is None:
        parser.error("Specify a stage: dpt/sft/dft, or --stage dpt/sft/dft")
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((Path(args.profile) if args.profile else root / "vred/configs/launch" / f"{args.stage}.json").read_text())
    for name, field in (("gpus", "gpus"), ("config", "training_config"), ("foundation_config", "foundation_config"),
                        ("input_model", "input_model"), ("output", "output"), ("seed", "seed")):
        if getattr(args, name) is not None:
            profile[field] = getattr(args, name)
    if profile["gpus"] < 1:
        parser.error("--gpus must be positive")
    training = load_training_config(root / profile["training_config"])
    assets = load_config(root / profile["foundation_config"])
    args.seed = profile["seed"]
    try:
        apply_training_overrides(args, training, assets, args.stage)
    except ValueError as error:
        parser.error(str(error))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={profile['gpus']}", "-m", "vred.train", "--stage", args.stage,
               "--backend", "deepspeed", "--config", profile["training_config"],
               "--foundation-config", profile["foundation_config"], "--output", profile["output"]]
    command += forwarded_overrides(args)
    resume = args.resume or profile["resume"]
    if resume:
        command += ["--resume", resume]
    elif profile["input_model"]:
        command += ["--input-model", profile["input_model"]]
    print(json.dumps(dict(profile=profile, command=command, precision=training["precision"],
                          micro_batch_size=training["micro_batch_size"], settings=training["stages"][args.stage],
                          global_batch=profile["gpus"] * training["micro_batch_size"] * training["stages"][args.stage]["accumulation"],
                          training=training, assets=assets)), flush=True)
    if not args.dry_run:
        environment = dict(os.environ, PYTHONHASHSEED=str(profile["seed"]))
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
