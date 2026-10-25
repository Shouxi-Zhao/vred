"""Inference adapter for LLaVA-KD's ten benchmark/scorer layouts.

Images can be served from a precomputed evaluation index (``--cache-index``) or
encoded online with the base SigLIP2 vision tower (``--vision-tower``); training
remains cache-only and never loads a vision encoder.
"""

import argparse
import csv
from itertools import islice
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .configs import load_config
from .data import read_cache_index
from .infer import generate_cached_batch, load_feature
from .model import load_model
from .summary import write_metrics
from .vision import OnlineVisionEncoder, decode_base64_image, open_image, resolve_image_path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def read_questions(path, kind):
    if kind == "tsv":
        # Embedded base64 images need an explicit question-ID -> cache-key map.
        import pandas as pd
        rows = pd.read_csv(path, sep="\t").fillna("").to_dict("records")
        for row in rows:
            options = []
            for letter in "ABCD":
                if letter not in row or not str(row[letter]):
                    break
                options.append(str(row[letter]))
            text = (str(row.get("hint", "")) + "\n" if row.get("hint") else "") + str(row["question"])
            text += "".join(f"\n{letter}. {value}" for letter, value in zip("ABCD", options))
            yield dict(question_id=row["index"], text=text, image_map_required=True,
                       image_base64=str(row.get("image", "")),
                       options=options, option_char=list("ABCD"[:len(options)]))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if kind == "jsonl" else json.loads(path.read_text())
        for row in rows:
            item = dict(row)
            if kind == "scienceqa":
                item.update(question_id=row["id"], text=row["conversations"][0]["value"])
            elif kind == "mmmu":
                item.update(question_id=row["id"], text=row["prompt"])
            yield item


def run_command(command, log, cwd=ROOT):
    result = subprocess.run([str(x) for x in command], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with log.open("a") as stream:
        stream.write(json.dumps([str(x) for x in command]) + "\n" + result.stdout + "\n")
    if result.returncode:
        raise RuntimeError(f"Legacy scorer/converter failed ({result.returncode}); see {log}")
    return result.stdout


def _write_scores(output, payload):
    (output / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def score_or_export(name, spec, directory, questions, answers, output):
    """Invoke local or legacy scoring/conversion code; retain its original output."""
    from .local_scorers import (parse_mme_log, parse_mmmu_stdout, score_gqa, score_mmbench,
                                score_vizwiz, score_vqav2)

    py, log = sys.executable, output / "scorer.log"
    log.write_text("")  # existing scorer output is overwritten, not appended
    def run(*command, cwd=ROOT):
        return run_command(command, log, cwd)
    if name == "scienceqa":
        run(py, ROOT / "llavakd/eval/eval_science_qa.py", "--base-dir", directory,
            "--result-file", answers, "--output-file", output / "analysis.json",
            "--output-result", output / "scores.json")
    elif name == "textvqa":
        run(py, "-m", "llavakd.eval.eval_textvqa", "--annotation-file", directory / "TextVQA_0.5.1_val.json", "--result-file", answers)
    elif name == "pope":
        run(py, ROOT / "llavakd/eval/eval_pope.py", "--annotation-dir", directory / "coco",
            "--question-file", questions, "--result-file", answers)
    elif name == "gqa":
        if not spec.get("annotations"):
            raise ValueError("GQA local scoring requires spec['annotations'] (official questions file)")
        score_gqa(answers, directory / spec["annotations"], output)
    elif name == "vqav2":
        if not spec.get("annotations"):
            raise ValueError("VQAv2 local scoring requires spec['annotations'] (val2014 annotations)")
        score_vqav2(answers, directory / spec["annotations"], output)
    elif name == "mme":
        # The supplied converter's interface fixes answers/<experiment>.jsonl.
        experiment = "vred_" + output.parent.name
        destination = directory / "answers" / f"{experiment}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(answers, destination)
        run(py, directory / "convert_answer_to_mme.py", "--experiment", experiment, cwd=directory)
        text = run(py, directory / "eval_tool/calculation.py", "--results_dir", f"answers/{experiment}",
                   cwd=directory / "eval_tool")
        _write_scores(output, parse_mme_log(text))
    elif name == "mmmu":
        prediction = output / "predictions.json"
        run(py, ROOT / "scripts/convert_answer_to_mmmu.py", "--answers-file", answers, "--answers-output", prediction)
        text = run(py, directory / "eval/main_eval_only.py", "--output_path", prediction, cwd=directory / "eval")
        _write_scores(output, parse_mmmu_stdout(text))
    elif name in ("mmbench", "mmbench_cn"):
        if spec["scoring"] == "local":
            score_mmbench(answers, questions, output, dataset=name)
        else:
            try:
                import openpyxl  # noqa: F401
            except ImportError as error:
                raise RuntimeError("MMBench conversion requires openpyxl in the active Python environment") from error
            run(py, ROOT / "scripts/convert_mmbench_for_submission.py", "--annotation-file", questions,
                "--result-dir", output, "--upload-dir", output, "--experiment", "answers")
    elif name == "vizwiz":
        if spec["scoring"] == "local":
            if not spec.get("annotations"):
                raise ValueError("VizWiz local scoring requires spec['annotations'] (Annotations/val.json)")
            score_vizwiz(answers, directory / spec["annotations"], output)
        else:
            # The test split has no public answers; retain the documented
            # submission export for a spec that explicitly asks for it.
            from llavakd.eval.m4c_evaluator import EvalAIAnswerProcessor
            processor = EvalAIAnswerProcessor()
            predictions = [json.loads(line) for line in answers.read_text().splitlines()]
            source = {str(q["question_id"]): q for q in read_questions(questions, "jsonl")}
            exported = [dict(image=source[str(prediction["question_id"])]["image"],
                             answer=processor(prediction["text"])) for prediction in predictions]
            (output / "submission.json").write_text(json.dumps(exported) + "\n")
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return "submission_required" if spec["scoring"] == "submission" else "locally_scored"


def benchmark_record(name, spec, directory):
    report = directory / "result.json"
    record = (json.loads(report.read_text()) if report.exists()
              else dict(benchmark=name, status="not_run", scoring=spec["scoring"]))
    if (directory / "scores.json").exists() and record["status"] == "locally_scored":
        record["metrics"] = json.loads((directory / "scores.json").read_text())
    if (directory / "scorer.log").exists():
        record["scorer_log"] = str(directory / "scorer.log")
        if record["status"] == "locally_scored" and "metrics" not in record:
            # Preserve repeated labels (e.g. POPE categories); never invent
            # an aggregate from a scorer's printed per-category values.
            matches = re.findall(r"^([A-Za-z][^:\n]{0,80}):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(%?)\s*$",
                                 (directory / "scorer.log").read_text(), re.MULTILINE)
            if matches:
                record["metrics"] = [dict(name=label.strip(), value=float(value), unit=unit)
                                     for label, value, unit in matches]
    return record


def _compact_metrics(metrics):
    """Reduce a scores.json/scorer-log record to one-line primary metrics."""
    if isinstance(metrics, list):
        ignored = {"samples", "sample", "num", "count", "n"}
        return " ".join(f"{item.get('name')}={item.get('value')}{item.get('unit', '')}"
                        for item in metrics
                        if item.get("name") and item["name"].strip().lower() not in ignored)
    if isinstance(metrics, dict):
        parts = []
        for key in ("accuracy", "acc", "score", "total", "f1", "precision", "recall"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value}")
        if not parts:
            overall = metrics.get("overall")
            if isinstance(overall, dict) and isinstance(overall.get("acc"), (int, float)):
                parts.append(f"overall_acc={overall['acc']}")
            elif isinstance(overall, (int, float)):
                parts.append(f"overall={overall}")
        if not parts:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    parts.append(f"{key}={value}")
                if len(parts) >= 3:
                    break
        return " ".join(parts) if parts else "scored"
    return "scored"


def print_benchmark_score(name, record):
    """Print exactly one terminal line with this benchmark's score/status."""
    status = record.get("status")
    if status == "locally_scored":
        print(f"{name}: {_compact_metrics(record.get('metrics'))}", flush=True)
    elif status == "submission_required":
        print(f"{name}: submission_required", flush=True)
    elif status == "subset_only":
        print(f"{name}: subset_only (no official score)", flush=True)
    elif status == "scoring_failed":
        error = str(record.get("error", "")).replace("\n", " ")[:180]
        print(f"{name}: scoring_failed {error}", flush=True)
    else:
        print(f"{name}: {status}", flush=True)


def collect(config, output):
    records = [benchmark_record(name, spec, output / name)
               for name, spec in config["benchmarks"].items()]
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["benchmark", "status", "scoring", "metrics", "scorer_log"])
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(record[k]) if k == "metrics" and k in record else record.get(k, "") for k in writer.fieldnames})
    return records


def prerequisites(name, spec, directory):
    """Return missing local files/tools needed to score this benchmark."""
    missing = []
    if not (directory / spec["questions"]).is_file():
        missing.append(spec["questions"])
    if spec.get("annotations") and not (directory / spec["annotations"]).is_file():
        missing.append(spec["annotations"])
    if name == "scienceqa":
        for item in ("pid_splits.json", "problems.json"):
            if not (directory / item).is_file():
                missing.append(item)
    elif name == "textvqa":
        if not (directory / "TextVQA_0.5.1_val.json").is_file():
            missing.append("TextVQA_0.5.1_val.json")
    elif name == "pope":
        if not (directory / "coco").is_dir():
            missing.append("coco/")
    elif name == "mme":
        for item in ("MME_Benchmark_release_version", "convert_answer_to_mme.py", "eval_tool/calculation.py"):
            if not (directory / item).exists():
                missing.append(item)
    elif name == "mmmu":
        for item in ("eval/main_eval_only.py", "eval/answer_dict_val.json"):
            if not (directory / item).is_file():
                missing.append(item)
    elif name in ("mmbench", "mmbench_cn") and spec["scoring"] == "submission":
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            missing.append("openpyxl")
    if spec.get("images") and name not in ("mmbench", "mmbench_cn"):
        if not (directory / spec["images"]).is_dir():
            missing.append(spec["images"] + "/")
    return missing


def merge_shards(name, spec, directory, questions, destination, shard_count, limit=None):
    """Merge shard answer files in original row order, then score once.

    TextVQA reuses one image question_id for multiple questions, so the merge
    key is ``benchmark_index`` (the row position in the benchmark), never the
    question id. Duplicate question ids are therefore valid.
    """
    expected = list(islice(read_questions(questions, spec["format"]), limit))
    for position, item in enumerate(expected):
        item["benchmark_index"] = position
    merged = {}
    first_report = None
    for index in range(shard_count):
        answer_path = destination / f"answers.shard-{index}-of-{shard_count}.jsonl"
        report_path = destination / f"result.shard-{index}-of-{shard_count}.json"
        if not answer_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"Missing shard output: {answer_path} / {report_path}")
        if first_report is None:
            first_report = json.loads(report_path.read_text())
        for line in answer_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "benchmark_index" not in record:
                raise ValueError(f"{name}: shard answer lacks benchmark_index; rerun the shards")
            position = int(record["benchmark_index"])
            if position in merged:
                raise ValueError(f"{name}: duplicate benchmark_index {position} across shards")
            merged[position] = record
    expected_positions = list(range(len(expected)))
    expected_set = set(expected_positions)
    missing = [position for position in expected_positions if position not in merged]
    extra = [position for position in merged if position not in expected_set]
    if missing or extra:
        raise ValueError(f"{name}: shard coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    answers = destination / "answers.jsonl"
    with answers.open("w") as stream:
        for position in expected_positions:
            stream.write(json.dumps(merged[position], ensure_ascii=False) + "\n")
    report = dict(first_report or {})
    report.update(benchmark=name, status="subset_only" if limit is not None else "predictions_ready",
                  count=len(expected), shards=shard_count, shard_index=None, shard_count=shard_count)
    report_path = destination / "result.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if limit is not None:
        return report
    missing_prereq = prerequisites(name, spec, directory)
    if missing_prereq:
        raise FileNotFoundError(f"{name}: missing evaluation prerequisites: {missing_prereq}")
    expected_ids = [str(item["question_id"]) for item in expected]
    actual = [str(json.loads(line)["question_id"]) for line in answers.read_text().splitlines()]
    if actual != expected_ids:
        raise ValueError("Answer coverage/order differs from complete benchmark questions")
    try:
        report["status"] = score_or_export(name, spec, directory, questions, answers, destination)
    except Exception as error:
        report.update(status="scoring_failed", error=str(error))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def write_run_all_summary(args, records, output):
    """Write metrics.json automatically after a complete ``--benchmarks all`` run."""
    if args.benchmarks != "all" or args.limit is not None:
        return None
    metrics = write_metrics(records, output)
    print(f"[run-all] metrics written: {output / 'metrics.json'}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def run_all(args, config, root, output):
    """Run each benchmark on all GPU workers, then start the next benchmark immediately.

    The six shard workers of benchmark N are joined; their answer merge/scoring
    runs on CPU in a background thread while the same GPU workers immediately
    start benchmark N+1. A final collect writes the combined results.
    """
    if not args.model:
        raise SystemExit("--action run-all requires --model")
    if args.gpus < 1:
        raise SystemExit("--gpus must be positive")
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.gpus))
    if len(gpu_ids) != args.gpus:
        raise SystemExit("--gpu-ids count must match --gpus")
    if args.benchmarks == "all":
        benchmarks = list(config["benchmarks"])
    else:
        benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    unknown = [x for x in benchmarks if x not in config["benchmarks"]]
    if unknown:
        raise SystemExit("unknown benchmarks: " + ", ".join(unknown))
    count = len(gpu_ids)
    output.mkdir(parents=True, exist_ok=True)

    def shard_command(name, index):
        command = [sys.executable, "-m", "vred.evaluate", "--action", "run",
                   "--benchmark", name, "--model", args.model,
                   "--output", str(output), "--config", args.config,
                   "--shard-index", str(index), "--shard-count", str(count),
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--eval-batch-size", str(args.eval_batch_size),
                   "--vision-dtype", args.vision_dtype]
        if args.foundation_config:
            command += ["--foundation-config", args.foundation_config]
        if args.root:
            command += ["--root", args.root]
        if args.cache_index:
            command += ["--cache-index", args.cache_index]
        if args.vision_tower:
            command += ["--vision-tower", args.vision_tower]
        if args.image_root:
            command += ["--image-root", args.image_root]
        if args.image_map:
            command += ["--image-map", args.image_map]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        return command

    def run_shards(name, spec, questions):
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        expected_total = sum(1 for _ in islice(read_questions(questions, spec["format"]), args.limit))
        answer_paths = ([destination / "answers.jsonl"] if count == 1 else
                        [destination / f"answers.shard-{index}-of-{count}.jsonl" for index in range(count)])
        processes = []
        for index, gpu in enumerate(gpu_ids):
            log_path = destination / f"shard-{gpu}.log"
            stream = log_path.open("w")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                shard_command(name, index), cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
            processes.append((gpu, process, stream, log_path))

        def completed():
            done = 0
            for path in answer_paths:
                if path.is_file():
                    with path.open() as handle:
                        done += sum(1 for line in handle if line.strip())
            return done

        progress = tqdm(total=expected_total, desc=f"{name} [{count} GPUs]", unit="q",
                        dynamic_ncols=True, bar_format=PROGRESS_FORMAT)
        failures = []
        try:
            while any(process.poll() is None for _, process, _, _ in processes):
                done = completed()
                if done > progress.n:
                    progress.update(done - progress.n)
                time.sleep(1.0)
        finally:
            for gpu, process, stream, log_path in processes:
                code = process.wait()
                stream.close()
                if code:
                    failures.append((gpu, code, log_path))
            done = completed()
            if done > progress.n:
                progress.update(done - progress.n)
            progress.close()
        if failures:
            details = ", ".join(f"gpu={gpu} code={code} log={log}" for gpu, code, log in failures)
            raise RuntimeError(f"{name}: shard generation failed: {details}")
        print(f"[run-all] {name}: {count} shards complete", flush=True)

    errors = []
    pending = []
    def finalize(name):
        spec = config["benchmarks"][name]
        directory = root / spec["directory"]
        questions = directory / spec["questions"]
        try:
            if count == 1:
                print(f"[run-all] {name}: merged/scored by single worker", flush=True)
                record = benchmark_record(name, spec, output / name)
                print_benchmark_score(name, record)
                return
            merge_shards(name, spec, directory, questions, output / name, count, args.limit)
            record = benchmark_record(name, spec, output / name)
            print_benchmark_score(name, record)
            print(f"[run-all] {name}: merged/scored", flush=True)
        except Exception as error:  # noqa: BLE001 - surfaced after all GPU work
            errors.append((name, error))
            print(f"[run-all] {name}: FAILED: {error}", flush=True)

    for name in benchmarks:
        print(f"[run-all] {name}: starting {count} shards", flush=True)
        spec = config["benchmarks"][name]
        directory = root / spec["directory"]
        questions = directory / spec["questions"]
        run_shards(name, spec, questions)
        thread = threading.Thread(target=finalize, args=(name,), name=f"finalize-{name}")
        thread.start()
        pending.append(thread)
        print(f"[run-all] {name}: next benchmark starts while merge/scoring continues", flush=True)
    for thread in pending:
        thread.join()
    if errors:
        details = "; ".join(f"{name}: {error}" for name, error in errors)
        raise RuntimeError("run-all merge/scoring failures: " + details)
    records = collect(config, output)
    write_run_all_summary(args, records, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("list", "run", "run-all", "score", "collect", "summary"), default="run")
    parser.add_argument("--config", default=str(ROOT / "vred/configs/evaluation.json"))
    parser.add_argument("--benchmark")
    parser.add_argument("--benchmarks", default="all",
                        help="run-all: comma-separated benchmark names, or all")
    parser.add_argument("--gpus", type=int, default=6, help="run-all: shard count / GPU workers")
    parser.add_argument("--gpu-ids", help="run-all: comma-separated physical GPU ids (default 0..gpus-1)")
    parser.add_argument("--shard-index", type=int, default=0, help="run: 0-based shard index")
    parser.add_argument("--shard-count", type=int, default=1, help="run: number of shard workers")
    parser.add_argument("--root", help="Override benchmark root")
    parser.add_argument("--model")
    parser.add_argument("--foundation-config")
    parser.add_argument("--cache-index", help="Optional precomputed evaluation feature index; omit to encode images online")
    parser.add_argument("--vision-tower", help="SigLIP2 vision tower for online evaluation (default: assets.vision_tower)")
    parser.add_argument("--vision-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--image-root", help="Override the benchmark image root for online evaluation")
    parser.add_argument("--image-map", help="JSON question ID -> image/cache key, required for cached embedded TSV images")
    parser.add_argument("--output", default="results/vred")
    parser.add_argument("--limit", type=int, help="Generate a diagnostic subset without official scoring")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="Questions/images per generation batch (default 1; memory/throughput knob)")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root, output = Path(args.root or config["root"]).resolve(), Path(args.output).resolve()
    if args.action == "collect":
        print(json.dumps(collect(config, output), indent=2))
        return
    if args.action == "summary":
        metrics = write_metrics(collect(config, output), output)
        print(json.dumps(metrics, indent=2))
        return
    if args.action == "run-all":
        run_all(args, config, root, output)
        return
    if args.action == "list":
        records = {}
        for name, spec in config["benchmarks"].items():
            directory = root / spec["directory"]
            missing = prerequisites(name, spec, directory)
            records[name] = dict(**spec, available=(directory / spec["questions"]).is_file(),
                                 scorer_ready=not missing, missing=missing)
        print(json.dumps(records, indent=2))
        return
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-count must be positive and --shard-index must be in [0, shard-count)")
    if args.benchmark not in config["benchmarks"]:
        parser.error("--benchmark must name one of: " + ", ".join(config["benchmarks"]))
    name, spec = args.benchmark, config["benchmarks"][args.benchmark]
    directory = root / spec["directory"]
    questions = directory / spec["questions"]
    missing = prerequisites(name, spec, directory)
    if not questions.is_file():
        raise FileNotFoundError(f"Benchmark questions unavailable: {questions}")
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    sharded = args.shard_count > 1
    suffix = f".shard-{args.shard_index}-of-{args.shard_count}" if sharded else ""
    answers = destination / f"answers{suffix}.jsonl"
    report_path = destination / f"result{suffix}.json"
    if args.action == "run":
        if missing:
            raise FileNotFoundError(f"{name}: missing evaluation prerequisites: {missing}")
        if not args.model:
            parser.error("run requires --model")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.eval_batch_size < 1:
            parser.error("--eval-batch-size must be positive")
        assets = load_config(args.foundation_config)
        vision_tower = args.vision_tower or assets.get("vision_tower")
        if not args.cache_index and not vision_tower:
            parser.error("run requires --cache-index or --vision-tower (or assets.vision_tower)")
        torch.set_num_threads(assets["cpu_threads"])
        torch.manual_seed(assets["seed"])
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = load_model(assets, saved=args.model, dtype=torch.bfloat16, device=device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        index = read_cache_index(args.cache_index) if args.cache_index else None
        encoder = None
        if index is None:
            vision_dtype = torch.float16 if args.vision_dtype == "fp16" else torch.bfloat16
            encoder = OnlineVisionEncoder(vision_tower, device=device, dtype=vision_dtype)
        image_map = json.loads(Path(args.image_map).read_text()) if args.image_map else {}
        if args.image_root:
            image_root = Path(args.image_root).resolve()
        elif spec.get("images"):
            image_root = (directory / spec["images"]).resolve()
        else:
            image_root = directory.resolve()
        from itertools import islice
        items = list(islice(read_questions(questions, spec["format"]), args.limit))
        for position, item in enumerate(items):
            item["benchmark_index"] = position
        if sharded:
            items = items[args.shard_index::args.shard_count]
        # Resolve all requested images before opening the answer output.
        for item in items:
            qid = str(item["question_id"])
            if "images" in item or item["text"].count("<image>") > 1:
                raise ValueError(f"{questions}, question={qid}: multiple-image inputs are unsupported")
            embedded = item.get("image_base64") if index is None else None
            if item.get("image_map_required") and qid not in image_map and not embedded:
                raise ValueError(f"{questions}, question={qid}: embedded image requires --image-map")
            image = image_map.get(qid, item.get("image"))
            if image is None and embedded:
                image = decode_base64_image(embedded)
            if image is not None and not isinstance(image, (str, Path, Image.Image)):
                raise ValueError(f"{questions}, question={qid}: multiple/invalid images: {image!r}")
            item["cache_key"] = (str(image) if isinstance(image, (str, Path))
                                 else (f"embedded:{qid}" if image is not None else None))
            if image is not None:
                if index is not None:
                    load_feature(index, image)
                elif isinstance(image, (str, Path)):
                    resolve_image_path(image_root, image)
        report = dict(benchmark=name, scoring=spec["scoring"], status="generating", model=args.model,
                      cache_index=str(args.cache_index) if args.cache_index else None,
                      vision_tower=vision_tower if index is None else None,
                      online_vision=index is None, image_root=str(image_root),
                      subset=args.limit, count=len(items), seed=assets["seed"],
                      max_new_tokens=args.max_new_tokens, eval_batch_size=args.eval_batch_size,
                      precision="bfloat16", questions=str(questions), image_map=args.image_map,
                      shard_index=args.shard_index, shard_count=args.shard_count)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        progress_desc = name + (f" shard {args.shard_index + 1}/{args.shard_count}" if sharded else "")
        with answers.open("w") as stream, tqdm(total=len(items), desc=progress_desc, unit="q",
                                              dynamic_ncols=True,
                                              bar_format=PROGRESS_FORMAT) as progress:
            for batch_start in range(0, len(items), args.eval_batch_size):
                chunk = items[batch_start:batch_start + args.eval_batch_size]
                prompts, features = [], []
                image_positions, images = [], []
                for item in chunk:
                    question = item["text"].replace("<image>", "").strip()
                    if name in ("scienceqa", "mmbench", "mmbench_cn"):
                        question += "\n" + ("请直接回答选项字母。" if name == "mmbench_cn" else "Answer with the option's letter from the given choices directly.")
                    prompts.append(question)
                    if index is not None:
                        features.append(load_feature(index, item["cache_key"]) if item["cache_key"] is not None else None)
                    else:
                        qid = str(item["question_id"])
                        image = image_map.get(qid, item.get("image"))
                        if image is None and item.get("image_base64"):
                            image = decode_base64_image(item["image_base64"])
                        if isinstance(image, (str, Path)):
                            image = open_image(image_root, image)
                        if image is not None:
                            image_positions.append(len(features))
                            images.append(image)
                        features.append(None)
                if images:
                    encoded = encoder.encode_batch(images)
                    for position, feature in zip(image_positions, encoded):
                        features[position] = feature
                predictions = generate_cached_batch(model, tokenizer, prompts, features, args.max_new_tokens)
                for item, question, feature, prediction in zip(chunk, prompts, features, predictions):
                    text = prediction["text"]
                    if name == "mmmu" and item.get("question_type") == "multiple-choice":
                        choices = item["all_choices"]
                        found = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text)
                        text = next((x for x in reversed(found) if x in choices), text)
                    if name == "textvqa":
                        scoring_prompt = item["text"]
                    elif name == "mme":
                        # MME's converter GT key is the raw question text; an
                        # <image> prefix breaks GT[category, file, prompt].
                        scoring_prompt = question
                    else:
                        scoring_prompt = ("<image>\n" if feature is not None else "") + question
                    answer = dict(question_id=item["question_id"], benchmark_index=item["benchmark_index"],
                                  prompt=scoring_prompt, text=text, answer_id=str(item["question_id"]),
                                  model_id=args.model,
                                  metadata=dict(cache_key=item["cache_key"], prompt_tokens=prediction["prompt_tokens"]))
                    if "options" in item:
                        answer.update(options=item["options"], option_char=item["option_char"], round_id=0)
                    stream.write(json.dumps(answer, ensure_ascii=False) + "\n")
                stream.flush()
                progress.update(len(chunk))
        report["status"] = "shard_ready" if sharded else ("subset_only" if args.limit else "predictions_ready")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if report_path.name.startswith("result.shard-"):
        return
    report = json.loads(report_path.read_text())
    if report["status"] in ("subset_only", "generating"):
        if args.action == "score":
            raise ValueError("Incomplete/subset predictions cannot be reported as official benchmark scores")
    else:
        if missing:
            raise FileNotFoundError(f"{name}: missing evaluation prerequisites: {missing}")
        expected = [str(x["question_id"]) for x in read_questions(questions, spec["format"])]
        actual = [str(json.loads(line)["question_id"]) for line in answers.read_text().splitlines()]
        if actual != expected:
            raise ValueError("Answer coverage/order differs from complete benchmark questions")
        try:
            report["status"] = score_or_export(name, spec, directory, questions, answers, destination)
        except Exception as error:
            report.update(status="scoring_failed", error=str(error))
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            raise
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    records = collect(config, output)
    record = next((item for item in records if item["benchmark"] == name), None)
    if record is not None:
        print_benchmark_score(name, record)


if __name__ == "__main__":
    main()
