import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from vred.configs import load_config
from vred.data import CachedDataset, Collator, read_cache_index
from vred.evaluate import (collect, main as evaluation_main, merge_shards, read_questions,
                           score_or_export, write_run_all_summary)
from vred.infer import generate_cached, generate_cached_batch, load_feature
from vred.model import load_model
from vred.vision import decode_base64_image
from vred.local_scorers import (parse_mme_log, parse_mmmu_stdout, score_gqa, score_mmbench,
                                score_vizwiz, score_vqav2)
from vred.preflight import audit_dataset, require_coverage
from vred.summary import AVG7, BENCHMARKS, summarize_metrics, write_metrics
from vred.tests.fixtures import make_assets
from vred.train import main as train_main


class Delivery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="vred-delivery-")
        cls.root = Path(cls.temporary.name)
        cls.foundation, cls.training = make_assets(cls.root / "assets")
        cls.assets = load_config(cls.foundation)
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def train(self, destination, steps, resume=None, config=None, stage="dpt", input_model=None, extra_flags=()):
        command = ["vred.train", "--stage", stage, "--foundation-config", self.foundation,
                   "--config", config or self.training, "--output", str(destination), "--max-steps", str(steps)]
        if resume:
            command += ["--resume", str(resume)]
        if input_model:
            command += ["--input-model", str(input_model)]
        command += list(extra_flags)
        with patch("sys.argv", command), contextlib.redirect_stdout(io.StringIO()):
            train_main()

    def test_tensorboard_all_stages_accumulation_and_resume(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from torch.utils.tensorboard import SummaryWriter

        config = json.loads(Path(self.training).read_text())
        config["micro_batch_size"] = 1
        for settings in config["stages"].values():
            settings["accumulation"] = 2
        config_path = self.root / "tensorboard-training.json"
        config_path.write_text(json.dumps(config))
        predecessor = None
        for stage in ("dpt", "sft", "dft"):
            output = self.root / f"tensorboard-{stage}"
            self.train(output, 1, config=str(config_path), stage=stage, input_model=predecessor)
            events = EventAccumulator(str(output / "tensorboard")).Reload()
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            for name in ("loss", "ce", "write_raw", "write", "write_weighted", "logits_kd", "logits_kd_weighted"):
                scalars = events.Scalars(f"train/{name}")
                self.assertEqual([scalar.step for scalar in scalars], [1])
                self.assertAlmostEqual(scalars[0].value, sum(row[name] for row in rows) / len(rows), places=5)
            for tag in ("train/learning_rate", "train/epoch", "speed/optimizer_step_seconds",
                        "speed/global_tokens_per_second", "progress/global_examples", "memory/peak_rss_mib"):
                self.assertEqual(len(events.Scalars(tag)), 1)
            self.assertIn("run/config/text_summary", events.Tags()["tensors"])
            predecessor = output
        output = self.root / "tensorboard-dpt"
        with SummaryWriter(str(output / "tensorboard")) as writer:
            writer.add_scalar("train/loss", 999, 2)
            writer.add_scalar("train/loss", 999, 99)
        self.train(output, 2, resume=output, config=str(config_path))
        events = EventAccumulator(str(output / "tensorboard")).Reload()
        self.assertEqual([scalar.step for scalar in events.Scalars("train/loss")], [1, 2])
        self.assertLess(events.Scalars("train/loss")[-1].value, 999)
        self.train(output, 1, config=str(config_path))
        events = EventAccumulator(str(output / "tensorboard")).Reload()
        self.assertEqual([scalar.step for scalar in events.Scalars("train/loss")], [1])

    def test_tensorboard_directory_disable_and_exception_cleanup(self):
        from torch.utils.tensorboard import SummaryWriter

        output, logs = self.root / "tensorboard-custom", self.root / "custom-events"
        self.train(output, 1, extra_flags=("--tensorboard-log-dir", str(logs)))
        self.assertTrue(list(logs.glob("events.out.tfevents.*")))
        run = json.loads((output / "run_config.json").read_text())
        self.assertEqual(run["tensorboard"], dict(enabled=True, log_dir=str(logs.resolve())))
        output = self.root / "tensorboard-disabled"
        with patch("torch.utils.tensorboard.SummaryWriter", side_effect=AssertionError("disabled")):
            self.train(output, 1, extra_flags=("--no-tensorboard",))
        self.assertFalse((output / "tensorboard").exists())
        writers = []

        def create_writer(*args, **kwargs):
            writer = SummaryWriter(*args, **kwargs)
            writers.append(writer)
            return writer

        with patch("torch.utils.tensorboard.SummaryWriter", side_effect=create_writer), \
                patch("vred.trainer.VREDTrainer.loss", side_effect=RuntimeError("intentional failure")), \
                self.assertRaisesRegex(RuntimeError, "intentional failure"):
            self.train(self.root / "tensorboard-failed", 1)
        self.assertEqual(len(writers), 1)
        self.assertIsNone(writers[0].all_writers)

    def test_fresh_run_overwrites_previous_checkpoints_and_metrics(self):
        output = self.root / "overwrite"
        self.train(output, 2)
        (output / "deepspeed" / "step-old").mkdir(parents=True)
        (output / "trainer_state_rank5.pt").write_text("stale rank")
        self.train(output, 1)
        state = torch.load(output / "trainer_state_rank0.pt", weights_only=False)
        self.assertEqual(state["steps"], 1)
        self.assertEqual(state["examples"], 2)
        self.assertFalse((output / "deepspeed").exists())
        self.assertFalse((output / "trainer_state_rank5.pt").exists())
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["optimizer_steps"], 1)
        self.assertEqual(json.loads((output / "checkpoint_complete.json").read_text())["steps"], 1)

    def test_all_stages_show_optimizer_progress_and_resume_position(self):
        config = json.loads(Path(self.training).read_text())
        config["micro_batch_size"] = 1
        for settings in config["stages"].values():
            settings["accumulation"] = 2
        config_path = self.root / "progress-training.json"
        config_path.write_text(json.dumps(config))
        predecessor = None
        for stage in ("dpt", "sft", "dft"):
            output = self.root / f"progress-{stage}"
            display = io.StringIO()
            with contextlib.redirect_stderr(display), patch("vred.train.tqdm", wraps=tqdm) as progress:
                self.train(output, 1, stage=stage, input_model=predecessor, config=str(config_path))
            self.assertEqual(progress.call_args.kwargs["total"], 1)
            self.assertEqual(progress.call_args.kwargs["initial"], 0)
            self.assertFalse(progress.call_args.kwargs["disable"])
            rendered = display.getvalue()
            self.assertIn(f"{stage.upper()} training", rendered)
            self.assertIn("100%", rendered)
            self.assertIn("tokens_per_s", rendered)
            self.assertIn("loss=", rendered)
            state = torch.load(output / "trainer_state_rank0.pt", weights_only=False)
            self.assertEqual((state["steps"], state["microsteps"]), (1, 2))
            predecessor = output
        display = io.StringIO()
        output = self.root / "progress-dpt"
        with contextlib.redirect_stderr(display), patch("vred.train.tqdm", wraps=tqdm) as progress:
            self.train(output, 2, resume=output, config=str(config_path))
        self.assertEqual(progress.call_args.kwargs["initial"], 1)
        self.assertEqual(progress.call_args.kwargs["total"], 2)
        self.assertIn("100%", display.getvalue())

    def test_resume_matches_uninterrupted_weights_optimizer_and_exposure(self):
        resumed, whole = self.root / "resumed", self.root / "whole"
        self.train(resumed, 1)
        first = torch.load(resumed / "trainer_state_rank0.pt", weights_only=False)
        self.assertEqual((first["epoch"], first["cursor"]), (1, 0))
        self.train(resumed, 2, resume=resumed)
        self.train(whole, 2)
        a = load_model(self.assets, saved=resumed, dtype=torch.bfloat16)
        b = load_model(self.assets, saved=whole, dtype=torch.bfloat16)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        states = [torch.load(p / "trainer_state_rank0.pt", weights_only=False) for p in (resumed, whole)]
        for key in ("steps", "examples", "tokens", "microsteps", "epoch", "cursor", "scheduler"):
            self.assertEqual(states[0][key], states[1][key])
        for key in ("torch_rng", "position_rng"):
            torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
        for key, state in states[0]["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, states[1]["optimizer"]["state"][key][name], atol=0, rtol=0)
        summary = json.loads((resumed / "summary.json").read_text())
        self.assertEqual(summary["global_examples"], 4)
        run = json.loads((resumed / "run_config.json").read_text())
        self.assertEqual(run["coverage"]["missing_files"], 0)
        self.assertGreater(run["trainable_parameters"], 0)

    def test_mid_epoch_resume_and_configuration_mismatch(self):
        config = json.loads(Path(self.training).read_text())
        config["micro_batch_size"] = 1
        config["save_every_steps"] = 1
        path = self.root / "micro1.json"
        path.write_text(json.dumps(config))
        resumed, whole = self.root / "mid-resumed", self.root / "mid-whole"
        self.train(resumed, 1, config=str(path))
        state = torch.load(resumed / "trainer_state_rank0.pt", weights_only=False)
        self.assertEqual((state["epoch"], state["cursor"]), (0, 1))
        self.train(resumed, 2, resume=resumed, config=str(path))
        self.train(whole, 2, config=str(path))
        for filename in ("connector.pt",):
            left, right = [torch.load(p / filename, weights_only=True) for p in (resumed, whole)]
            for key in left:
                torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "training differs"):
            self.train(resumed, 3, resume=resumed)

    def test_online_vision_dataset_collator_and_preflight(self):
        tokenizer = AutoTokenizer.from_pretrained(self.assets["student"], local_files_only=True)
        image_root = Path(self.assets["sft_json"]).parent.parent
        dataset = CachedDataset(self.assets["sft_json"], None, tokenizer,
                                online=True, image_root=image_root)
        item = dataset[0]
        self.assertIsInstance(item["image"], Image.Image)
        self.assertIsNone(item["feature"])
        batch = Collator(tokenizer)([dataset[0], dataset[1]])
        self.assertIsInstance(batch.images[0], Image.Image)
        self.assertIsNone(batch.images[1])
        report = audit_dataset(dataset, validate_arrays=True)
        self.assertEqual((report["records"], report["image_records"], report["text_records"]), (2, 1, 1))
        self.assertEqual((report["index_entries"], report["missing_keys"], report["missing_files"]), (0, 0, 0))
        require_coverage(report)
        broken = copy.copy(dataset)
        broken.records = [dict(dataset.records[0], image="synthetic/missing.jpg"), dataset.records[1]]
        failure = audit_dataset(broken)
        self.assertEqual(failure["missing_files"], 1)
        with self.assertRaisesRegex(ValueError, "synthetic/missing"):
            require_coverage(failure)

    def test_online_vision_training_path(self):
        class FakeEncoder:
            def __init__(self):
                self.images_seen = 0

            def encode_batch_train(self, images):
                self.images_seen += len(images)
                return torch.ones(len(images), 729, 1152, dtype=torch.float32)

        encoder = FakeEncoder()
        output = self.root / "online-vision-dpt"
        with patch("vred.train.make_online_encoder", return_value=encoder):
            self.train(output, 1, stage="dpt", extra_flags=("--online-vision",))
        run = json.loads((output / "run_config.json").read_text())
        self.assertTrue(run["online_vision"])
        self.assertTrue(run["online_image_root"].endswith("assets"))
        self.assertGreater(encoder.images_seen, 0)
        self.assertEqual(json.loads((output / "summary.json").read_text())["optimizer_steps"], 1)

    def test_cache_coverage_paths_missing_records_and_inference(self):
        tokenizer = AutoTokenizer.from_pretrained(self.assets["student"], local_files_only=True)
        dataset = CachedDataset(self.assets["sft_json"], self.assets["cache_index"], tokenizer)
        report = audit_dataset(dataset, validate_arrays=True)
        require_coverage(report)
        self.assertEqual((report["records"], report["image_records"], report["text_records"]), (2, 1, 1))
        broken = copy.copy(dataset)
        broken.index = {}
        failure = audit_dataset(broken)
        self.assertEqual(failure["missing_keys"], 1)
        with self.assertRaisesRegex(ValueError, "synthetic-image"):
            require_coverage(failure)
        index = self.root / "bad-index.json"
        index.write_text(json.dumps({"x": "../escape.npy"}))
        with self.assertRaisesRegex(ValueError, "escapes"):
            read_cache_index(index)
        index.write_text(json.dumps({"./a": "x.npy", "a": "y.npy"}))
        with self.assertRaisesRegex(ValueError, "collision"):
            read_cache_index(index)
        model = load_model(self.assets, dtype=torch.bfloat16)
        feature = load_feature(dataset.index, "./synthetic\\square.jpg")
        prediction = generate_cached(model, tokenizer, "What is shown?", feature, 2)
        self.assertEqual(len(prediction["token_ids"]), 2)
        self.assertTrue(prediction["cached_image"])
        self.assertGreaterEqual(prediction["prompt_tokens"], 729)
        self.assertFalse(hasattr(model, "vision_tower"))

    def test_evaluation_reuses_local_scorer_and_distinguishes_submission(self):
        directory = self.root / "textvqa"
        directory.mkdir()
        output = self.root / "evaluation/textvqa"
        output.mkdir(parents=True)
        question = dict(question_id=1, image="synthetic/square.jpg", text="What is shown?\nShort answer:")
        questions = directory / "questions.jsonl"
        questions.write_text(json.dumps(question) + "\n")
        (directory / "TextVQA_0.5.1_val.json").write_text(json.dumps(dict(data=[dict(image_id=1,
            question="What is shown?", answers=["square"] * 10)])))
        answers = output / "answers.jsonl"
        answers.write_text(json.dumps(dict(question_id=1, prompt=question["text"], text="square")) + "\n")
        spec = dict(scoring="local", format="jsonl", directory="textvqa", questions="questions.jsonl")
        status = score_or_export("textvqa", spec, directory, questions, answers, output)
        self.assertEqual(status, "locally_scored")
        self.assertIn("Accuracy: 100.00%", (output / "scorer.log").read_text())
        (output / "result.json").write_text(json.dumps(dict(benchmark="textvqa", scoring="local", status=status)))
        export = self.root / "evaluation/vizwiz"
        export.mkdir()
        self.assertEqual(score_or_export("vizwiz", dict(scoring="submission"), directory, questions, answers, export), "submission_required")
        (export / "result.json").write_text(json.dumps(dict(benchmark="vizwiz", scoring="submission", status="submission_required")))
        records = collect(dict(benchmarks={"textvqa": spec, "vizwiz": dict(scoring="submission")}), output.parent)
        self.assertEqual([r["status"] for r in records], ["locally_scored", "submission_required"])
        self.assertNotIn("metrics", records[1])

    def test_local_vqav2_gqa_and_legacy_metric_parsers(self):
        root = self.root / "local-scorers"
        root.mkdir()
        annotations = dict(annotations=[
            dict(question_id=1, answers=[dict(answer="square")] * 3 + [dict(answer="cube")] * 7),
            dict(question_id=2, answers=[dict(answer="square")] + [dict(answer="cube")] * 9)])
        (root / "vqa-annotations.json").write_text(json.dumps(annotations))
        (root / "vqa-answers.jsonl").write_text("".join(
            json.dumps(dict(question_id=q, text="a square.")) + "\n" for q in (1, 2)))
        vqa = score_vqav2(root / "vqa-answers.jsonl", root / "vqa-annotations.json", root / "vqa-out")
        self.assertAlmostEqual(vqa["accuracy"], 100 * (1.0 + 1 / 3) / 2)
        self.assertEqual(vqa["count"], 2)
        self.assertTrue((root / "vqa-out/scores.json").is_file())

        questions = {
            "1": dict(answer="yes", isBalanced=True, types=dict(structural="verify", semantic="attr")),
            "2": dict(answer="no", isBalanced=True, types=dict(structural="verify", semantic="attr")),
            "3": dict(answer="yes", isBalanced=False, types=dict(structural="query", semantic="obj"))}
        (root / "gqa-questions.json").write_text(json.dumps(questions))
        (root / "gqa-answers.jsonl").write_text("".join(
            json.dumps(dict(question_id=q, text=text)) + "\n" for q, text in (("1", "Yes."), ("2", "yes"), ("3", "yes"))))
        gqa = score_gqa(root / "gqa-answers.jsonl", root / "gqa-questions.json", root / "gqa-out")
        self.assertAlmostEqual(gqa["accuracy"], 50.0)
        self.assertEqual(gqa["accuracy_per_structural_type"]["verify"]["num"], 2)
        self.assertEqual(gqa["accuracy_per_structural_type"]["verify"]["accuracy"], 50.0)

        mme_text = """=========== Perception ===========
total score: 150.0

\t existence  score: 100.0
\t count  score: 50.0

=========== Cognition ===========
total score: 40.0

\t commonsense_reasoning  score: 40.0
"""
        mme = parse_mme_log(mme_text)
        self.assertEqual(mme["categories"]["Perception"]["tasks"]["existence"], 100.0)
        self.assertEqual(mme["categories"]["Cognition"]["total"], 40.0)
        self.assertEqual(mme["total"], 190.0)
        mmmu = parse_mmmu_stdout("{'Accounting': {'num': 2, 'acc': 0.5}, 'Overall': {'num': 2, 'acc': 0.5}}\n")
        self.assertEqual(mmmu["overall"], dict(num=2, acc=0.5))

    def test_local_mmbench_accuracy(self):
        root = self.root / "mmbench-local"
        root.mkdir()
        annotations = root / "annotations.tsv"
        annotations.write_text(
            "index\tquestion\thint\tA\tB\tC\tD\tanswer\n"
            "1\tWhat?\t\tcat\tdog\t\t\tB\n"
            "2\tWhat?\t\tred\tblue\tgreen\t\tA\n"
            "3\tWhat?\t\tx\ty\t\t\tA\n")
        (root / "answers.jsonl").write_text("".join(
            json.dumps(dict(question_id=qid, text=text)) + "\n"
            for qid, text in ((1, "B. dog. A cat."), (2, "A"), (3, "no idea"))))
        result = score_mmbench(root / "answers.jsonl", annotations, root / "out", dataset="mmbench")
        self.assertAlmostEqual(result["accuracy"], 200 / 3)
        self.assertEqual((result["correct"], result["count"], result["missing"], result["unparsed"]),
                         (2, 3, 0, 1))
        self.assertTrue((root / "out/scores.json").is_file())

        (root / "wired").mkdir()
        spec = dict(scoring="local", questions="annotations.tsv")
        status = score_or_export("mmbench", spec, root, annotations, root / "answers.jsonl", root / "wired")
        self.assertEqual(status, "locally_scored")
        wired = json.loads((root / "wired/scores.json").read_text())
        self.assertAlmostEqual(wired["accuracy"], 200 / 3)
        self.assertEqual(wired["unparsed"], 1)

    def test_local_vizwiz_soft_accuracy(self):
        root = self.root / "vizwiz-local"
        annotations = [
            dict(image="a.jpg", answers=[dict(answer="cat")] * 2 + [dict(answer="dog")] * 2
                 + [dict(answer="bird")] * 6),
            dict(image="b.jpg", answers=[dict(answer="yes")] * 10),
        ]
        (root / "Annotations").mkdir(parents=True)
        (root / "Annotations/val.json").write_text(json.dumps(annotations))
        (root / "answers.jsonl").write_text("".join(
            json.dumps(dict(question_id=qid, text=text)) + "\n"
            for qid, text in ((0, "cat"), (1, "yes"))))
        result = score_vizwiz(root / "answers.jsonl", root / "Annotations/val.json", root / "out")
        self.assertAlmostEqual(result["accuracy"], 80.0)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["missing"], 0)
        self.assertTrue((root / "out/scores.json").is_file())

        (root / "wired").mkdir()
        status = score_or_export("vizwiz", dict(scoring="local", annotations="Annotations/val.json"),
                                 root, root / "questions.jsonl", root / "answers.jsonl", root / "wired")
        self.assertEqual(status, "locally_scored")
        wired = json.loads((root / "wired/scores.json").read_text())
        self.assertAlmostEqual(wired["accuracy"], 80.0)

    def test_summary_metrics_scaling_averages_and_missing(self):
        records = [
            dict(benchmark="vqav2", metrics={"accuracy": 50.0}),
            dict(benchmark="gqa", metrics={"accuracy": 60.0}),
            dict(benchmark="vizwiz", metrics={"accuracy": 40.0}),
            dict(benchmark="scienceqa", metrics={"acc": 58.0}),
            dict(benchmark="textvqa", metrics=[
                dict(name="Samples", value=5000.0),
                dict(name="Accuracy", value=52.5, unit="%")]),
            dict(benchmark="mme", metrics={"total": 1500.0}),
            dict(benchmark="mmbench", metrics={"accuracy": 70.0}),
            dict(benchmark="mmbench_cn", metrics={"accuracy": 65.0}),
            dict(benchmark="pope", metrics=[
                dict(name="Accuracy", value=0.80),
                dict(name="Accuracy", value=0.85),
                dict(name="Accuracy", value=0.90)]),
            dict(benchmark="mmmu", metrics={"overall": {"acc": 0.30}}),
        ]
        metrics = summarize_metrics(records)
        self.assertEqual(tuple(metrics), (*BENCHMARKS, "avg10", "avg7"))
        self.assertEqual(tuple(AVG7), ("gqa", "vizwiz", "scienceqa", "textvqa",
                                       "mme", "mmbench", "mmbench_cn"))
        self.assertAlmostEqual(metrics["mme"], 75.0)
        self.assertAlmostEqual(metrics["mmmu"], 30.0)
        self.assertAlmostEqual(metrics["pope"], 85.0)
        self.assertAlmostEqual(metrics["textvqa"], 52.5)
        self.assertAlmostEqual(metrics["avg10"], 58.55)
        self.assertAlmostEqual(metrics["avg7"], 60.0714)
        self.assertEqual(set(metrics), {*BENCHMARKS, "avg10", "avg7"})

        output = self.root / "summary-metrics"
        written = write_metrics(records, output)
        self.assertEqual(written, metrics)
        self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)

        with self.assertRaisesRegex(ValueError, "mmmu"):
            summarize_metrics([record for record in records if record["benchmark"] != "mmmu"])
        with self.assertRaisesRegex(ValueError, "pope"):
            summarize_metrics([record for record in records if record["benchmark"] != "pope"])

    def test_evaluation_summary_action_prints_only_metrics(self):
        records = [dict(benchmark=name, metrics={"accuracy": 10.0})
                   for name in BENCHMARKS if name not in ("mme", "mmmu")]
        records.extend([dict(benchmark="mme", metrics={"total": 1500.0}),
                        dict(benchmark="mmmu", metrics={"overall": {"acc": 0.5}})])
        metrics = summarize_metrics(records)
        with patch("vred.evaluate.collect", return_value=records), \
                patch("vred.evaluate.write_metrics", return_value=metrics) as write, \
                patch("sys.argv", ["vred.evaluate", "--action", "summary",
                                   "--output", str(self.root / "summary-action")]):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                evaluation_main()
        write.assert_called_once()
        self.assertEqual(json.loads(stdout.getvalue()), metrics)
        self.assertEqual(set(metrics), {*BENCHMARKS, "avg10", "avg7"})

    def test_run_all_all_benchmarks_writes_summary(self):
        records = [dict(benchmark=name, metrics={"accuracy": 10.0})
                   for name in BENCHMARKS if name not in ("mme", "mmmu", "pope", "textvqa")]
        records.extend([
            dict(benchmark="textvqa", metrics={"accuracy": 50.0}),
            dict(benchmark="mme", metrics={"total": 1500.0}),
            dict(benchmark="pope", metrics={"accuracy": 0.80}),
            dict(benchmark="mmmu", metrics={"overall": {"acc": 0.50}}),
        ])
        output = self.root / "run-all-summary"
        args = SimpleNamespace(benchmarks="all", limit=None)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            metrics = write_run_all_summary(args, records, output)
        self.assertIsNotNone(metrics)
        self.assertEqual(json.loads(stdout.getvalue().split("\n", 1)[1]), metrics)
        self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)

        subset_output = self.root / "run-all-subset"
        self.assertIsNone(write_run_all_summary(SimpleNamespace(benchmarks="gqa", limit=None),
                                                records, subset_output))
        self.assertFalse((subset_output / "metrics.json").exists())
        limited_output = self.root / "run-all-limited"
        self.assertIsNone(write_run_all_summary(SimpleNamespace(benchmarks="all", limit=8),
                                                records, limited_output))
        self.assertFalse((limited_output / "metrics.json").exists())

    def test_subset_cannot_be_scored(self):
        directory = self.root / "subset/textvqa"
        directory.mkdir(parents=True)
        (directory / "questions.jsonl").write_text(json.dumps(dict(question_id=1, text="test")) + "\n")
        config = dict(root=str(directory.parent), benchmarks={"textvqa": dict(directory="textvqa", questions="questions.jsonl", format="jsonl", scoring="local")})
        path = self.root / "eval-config.json"
        path.write_text(json.dumps(config))
        out = self.root / "subset-output/textvqa"
        out.mkdir(parents=True)
        (out / "result.json").write_text(json.dumps(dict(status="subset_only")))
        with patch("sys.argv", ["vred.evaluate", "--action", "score", "--config", str(path),
                                "--benchmark", "textvqa", "--output", str(out.parent)]):
            with self.assertRaisesRegex(ValueError, "subset"):
                evaluation_main()

    def test_benchmark_adapter_generates_scores_and_collects_synthetic_fixture(self):
        tokenizer = AutoTokenizer.from_pretrained(self.assets["student"], local_files_only=True)
        checkpoint = self.root / "eval-model"
        load_model(self.assets, dtype=torch.bfloat16).save_pretrained(checkpoint, tokenizer)
        directory = self.root / "eval-run-data/textvqa"
        directory.mkdir(parents=True)
        question = dict(question_id=101, text="What is shown?\nShort answer:", image="synthetic/square.jpg")
        (directory / "questions.jsonl").write_text(json.dumps(question) + "\n")
        (directory / "TextVQA_0.5.1_val.json").write_text(json.dumps(dict(data=[dict(image_id=101,
            question="What is shown?", answers=["square"] * 10)])))
        config = dict(root=str(directory.parent), benchmarks={"textvqa": dict(directory="textvqa", questions="questions.jsonl", format="jsonl", scoring="local")})
        path = self.root / "eval-run-config.json"
        path.write_text(json.dumps(config))
        output = self.root / "eval-run-output"
        command = ["vred.evaluate", "--config", str(path), "--benchmark", "textvqa", "--model", str(checkpoint),
                   "--foundation-config", self.foundation, "--cache-index", self.assets["cache_index"],
                   "--output", str(output), "--max-new-tokens", "2"]
        for _ in range(2):
            buffer = io.StringIO()
            with patch("sys.argv", command), contextlib.redirect_stdout(buffer):
                evaluation_main()
            self.assertIn("textvqa:", buffer.getvalue())
        records = json.loads((output / "results.json").read_text())
        self.assertEqual(records[0]["status"], "locally_scored")
        answer = json.loads((output / "textvqa/answers.jsonl").read_text())
        self.assertEqual(answer["metadata"]["cache_key"], "synthetic/square.jpg")

    def test_evaluation_online_vision_encoder_without_cache(self):
        from PIL import Image
        tokenizer = AutoTokenizer.from_pretrained(self.assets["student"], local_files_only=True)
        checkpoint = self.root / "eval-online-model"
        load_model(self.assets, dtype=torch.bfloat16).save_pretrained(checkpoint, tokenizer)
        directory = self.root / "eval-online-data/textvqa"
        image_dir = directory / "images"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32), (255, 0, 0)).save(image_dir / "square.jpg")
        question = dict(question_id=101, text="What is shown?\nShort answer:", image="square.jpg")
        (directory / "questions.jsonl").write_text(json.dumps(question) + "\n")
        (directory / "TextVQA_0.5.1_val.json").write_text(json.dumps(dict(data=[dict(image_id=101,
            question="What is shown?", answers=["square"] * 10)])))
        config = dict(root=str(directory.parent), benchmarks={"textvqa": dict(directory="textvqa",
            questions="questions.jsonl", format="jsonl", scoring="local", images="images")})
        path = self.root / "eval-online-config.json"
        path.write_text(json.dumps(config))
        output = self.root / "eval-online-output"
        images_seen = []
        class FakeEncoder:
            def __init__(self, *args, **kwargs):
                pass
            def encode_batch(self, images):
                images_seen.extend(image.size for image in images)
                return torch.zeros(len(images), 729, 1152, dtype=torch.float16)
            def encode(self, image):
                return self.encode_batch([image])[0]
        command = ["vred.evaluate", "--config", str(path), "--benchmark", "textvqa", "--model", str(checkpoint),
                   "--foundation-config", self.foundation, "--vision-tower", "fake-vision",
                   "--output", str(output), "--max-new-tokens", "2"]
        with patch("vred.evaluate.OnlineVisionEncoder", FakeEncoder), patch("sys.argv", command), \
                contextlib.redirect_stdout(io.StringIO()):
            evaluation_main()
        records = json.loads((output / "results.json").read_text())
        self.assertEqual(records[0]["status"], "locally_scored")
        answer = json.loads((output / "textvqa/answers.jsonl").read_text())
        self.assertEqual(answer["metadata"]["cache_key"], "square.jpg")
        report = json.loads((output / "textvqa/result.json").read_text())
        self.assertTrue(report["online_vision"])
        self.assertIsNone(report["cache_index"])
        self.assertEqual(images_seen, [(32, 32)])

    def test_shard_merge_allows_duplicate_question_ids(self):
        tokenizer = AutoTokenizer.from_pretrained(self.assets["student"], local_files_only=True)
        checkpoint = self.root / "duplicate-qid-model"
        load_model(self.assets, dtype=torch.bfloat16).save_pretrained(checkpoint, tokenizer)
        directory = self.root / "duplicate-qid-data/textvqa"
        directory.mkdir(parents=True)
        rows = [dict(question_id=101, text="What is shown?\nShort answer:", image="synthetic/square.jpg"),
                dict(question_id=101, text="What color is it?\nShort answer:", image="synthetic/square.jpg")]
        question_path = directory / "questions.jsonl"
        question_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "TextVQA_0.5.1_val.json").write_text(json.dumps(dict(data=[])))
        config = dict(root=str(directory.parent), benchmarks={"textvqa": dict(directory="textvqa",
            questions="questions.jsonl", format="jsonl", scoring="local")})
        path = self.root / "duplicate-qid-config.json"
        path.write_text(json.dumps(config))
        output = self.root / "duplicate-qid-output"
        for shard in (0, 1):
            command = ["vred.evaluate", "--config", str(path), "--benchmark", "textvqa",
                       "--model", str(checkpoint), "--foundation-config", self.foundation,
                       "--cache-index", self.assets["cache_index"], "--output", str(output),
                       "--max-new-tokens", "2", "--limit", "2",
                       "--shard-index", str(shard), "--shard-count", "2"]
            with patch("sys.argv", command), contextlib.redirect_stdout(io.StringIO()):
                evaluation_main()
        report = merge_shards("textvqa", config["benchmarks"]["textvqa"], directory, question_path,
                              output / "textvqa", 2, limit=2)
        self.assertEqual(report["status"], "subset_only")
        merged = [json.loads(line) for line in (output / "textvqa/answers.jsonl").read_text().splitlines()]
        self.assertEqual([item["benchmark_index"] for item in merged], [0, 1])
        self.assertEqual([str(item["question_id"]) for item in merged], ["101", "101"])

    def test_generate_cached_batch_matches_single(self):
        tokenizer = AutoTokenizer.from_pretrained(self.assets["student"], local_files_only=True)
        model = load_model(self.assets, dtype=torch.bfloat16)
        dataset = CachedDataset(self.assets["sft_json"], self.assets["cache_index"], tokenizer)
        feature = load_feature(dataset.index, "./synthetic\\square.jpg")
        prompts = ["What is shown?", "Hello world"]
        features = [feature, None]
        singles = [generate_cached(model, tokenizer, prompt, item, 2)
                   for prompt, item in zip(prompts, features)]
        batched = generate_cached_batch(model, tokenizer, prompts, features, 2)
        for single, item in zip(singles, batched):
            self.assertEqual(single["token_ids"], item["token_ids"])
            self.assertEqual(single["text"], item["text"])
            self.assertEqual(single["prompt_tokens"], item["prompt_tokens"])
            self.assertEqual(single["cached_image"], item["cached_image"])

    def test_tsv_questions_keep_embedded_image(self):
        import base64
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (0, 0, 255)).save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        path = self.root / "embedded.tsv"
        path.write_text("index\tquestion\tA\tB\tC\tD\timage\n"
                        "1\tWhat?\tone\ttwo\tthree\tfour\t" + encoded + "\n")
        item = next(read_questions(path, "tsv"))
        self.assertEqual(item["image_base64"], encoded)
        self.assertEqual(decode_base64_image(encoded).size, (8, 8))

    def test_evaluation_shards_merge_in_original_order(self):
        tokenizer = AutoTokenizer.from_pretrained(self.assets["student"], local_files_only=True)
        checkpoint = self.root / "eval-shard-model"
        load_model(self.assets, dtype=torch.bfloat16).save_pretrained(checkpoint, tokenizer)
        directory = self.root / "eval-shard-data/textvqa"
        directory.mkdir(parents=True)
        questions = [
            dict(question_id=101, text="What is shown?\nShort answer:", image="synthetic/square.jpg"),
            dict(question_id=102, text="What is shown?\nShort answer:", image="synthetic/square.jpg")]
        question_path = directory / "questions.jsonl"
        question_path.write_text("".join(json.dumps(q) + "\n" for q in questions))
        (directory / "TextVQA_0.5.1_val.json").write_text(json.dumps(dict(data=[
            dict(image_id=101, question="What is shown?", answers=["square"] * 10),
            dict(image_id=102, question="What is shown?", answers=["square"] * 10)])))
        config = dict(root=str(directory.parent), benchmarks={"textvqa": dict(directory="textvqa",
            questions="questions.jsonl", format="jsonl", scoring="local")})
        path = self.root / "eval-shard-config.json"
        path.write_text(json.dumps(config))
        output = self.root / "eval-shard-output"
        for shard in (0, 1):
            command = ["vred.evaluate", "--config", str(path), "--benchmark", "textvqa",
                       "--model", str(checkpoint), "--foundation-config", self.foundation,
                       "--cache-index", self.assets["cache_index"], "--output", str(output),
                       "--max-new-tokens", "2", "--shard-index", str(shard), "--shard-count", "2"]
            with patch("sys.argv", command), contextlib.redirect_stdout(io.StringIO()):
                evaluation_main()
        shard_answers = output / "textvqa/answers.shard-0-of-2.jsonl"
        self.assertTrue(shard_answers.is_file())
        report = merge_shards("textvqa", config["benchmarks"]["textvqa"], directory, question_path,
                              output / "textvqa", 2)
        self.assertEqual(report["status"], "locally_scored")
        merged = [json.loads(line) for line in (output / "textvqa/answers.jsonl").read_text().splitlines()]
        self.assertEqual([item["question_id"] for item in merged], [101, 102])
        self.assertTrue((output / "textvqa/scorer.log").is_file())

    def test_launch_profiles_resolve_six_gpus_and_all_benchmarks_are_declared(self):
        from vred.launch import main
        for stage in ("dpt", "sft", "dft"):
            output = io.StringIO()
            with patch("sys.argv", ["vred.launch", stage, "--dry-run"]), contextlib.redirect_stdout(output):
                main()
            resolved = json.loads(output.getvalue())
            self.assertIn("--nproc_per_node=6", resolved["command"])
            self.assertIn("deepspeed", resolved["command"])
            self.assertEqual(resolved["profile"]["seed"], 1234)
        config = json.loads((Path(__file__).parents[1] / "configs/evaluation.json").read_text())
        self.assertEqual(len(config["benchmarks"]), 10)
        self.assertEqual(sum(b["scoring"] == "submission" for b in config["benchmarks"].values()), 0)
        for name in ("gqa", "vqav2", "vizwiz"):
            self.assertEqual(config["benchmarks"][name]["scoring"], "local")
            self.assertTrue(config["benchmarks"][name]["annotations"])
        for name in ("mmbench", "mmbench_cn"):
            self.assertEqual(config["benchmarks"][name]["scoring"], "local")

    def test_online_vision_flags_resolve_and_forward(self):
        from vred.launch import main
        output = io.StringIO()
        command = ["vred.launch", "dft", "--dry-run", "--online-vision",
                   "--online-vision-root", "mllm_datasets"]
        with patch("sys.argv", command), contextlib.redirect_stdout(output):
            main()
        resolved = json.loads(output.getvalue())
        self.assertTrue(resolved["training"]["online_vision"])
        self.assertTrue(resolved["assets"]["online_vision_root"].endswith("mllm_datasets"))
        self.assertIn("--online-vision", resolved["command"])
        self.assertIn("--online-vision-root", resolved["command"])

    def test_launch_flags_resolve_and_forward_overrides(self):
        from vred.launch import main
        output = io.StringIO()
        command = ["vred.launch", "dft", "--dry-run", "--gpus", "2", "--seed", "9",
                   "--lr", "0.0001", "--epochs", "3", "--micro-batch-size", "2", "--accumulation", "4",
                   "--pairs", "3", "--lambda-write", "0.1", "--lambda-logits", "0.5",
                   "--write-loss-type", "effect_js", "--query-chunk-size", "8", "--write-position-limit", "0",
                   "--no-checkpoint-probes", "--output", "/tmp/vred-flags-example", "--input-model", "/tmp/previous",
                   "--sft-json", "/tmp/training.json", "--tensorboard-log-dir", "/tmp/vred-events", "--no-tensorboard"]
        with patch("sys.argv", command), contextlib.redirect_stdout(output):
            main()

        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["global_batch"], 16)
        self.assertEqual(resolved["settings"]["lr"], 1e-4)
        self.assertEqual(resolved["settings"]["lambda_write"], 0.1)
        self.assertEqual(resolved["training"]["write_loss_type"], "effect_js")
        self.assertIsNone(resolved["training"]["write_position_limit"])
        self.assertFalse(resolved["training"]["checkpoint_probes"])
        self.assertEqual(resolved["assets"]["seed"], 9)
        self.assertEqual(resolved["assets"]["sft_json"], "/tmp/training.json")
        self.assertIn("--no-checkpoint-probes", resolved["command"])
        self.assertIn("--no-tensorboard", resolved["command"])
        self.assertIn("--tensorboard-log-dir", resolved["command"])
        self.assertIn("/tmp/vred-events", resolved["command"])
        self.assertIn("--nproc_per_node=2", resolved["command"])
        with patch("sys.argv", ["vred.launch", "sft", "--lambda-write", "1", "--dry-run"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()

    def test_optional_write_flags_resolve_and_validate(self):
        from vred.launch import main
        command = ["vred.launch", "dft", "--dry-run", "--write-position-scope", "prompt",
                   "--write-effect-sampling", "stratified", "--write-position-limit", "2",
                   "--write-lambda-schedule", "warmup_cosine", "--reverse-kl"]
        output = io.StringIO()
        with patch("sys.argv", command), contextlib.redirect_stdout(output):
            main()
        resolved = json.loads(output.getvalue())
        training = resolved["training"]
        self.assertEqual(training["write_position_scope"], "prompt")
        self.assertEqual(training["write_effect_sampling"], "stratified")
        self.assertEqual(training["write_position_limit"], 2)
        self.assertEqual(training["write_lambda_schedule"], "warmup_cosine")
        self.assertEqual(training["write_loss_type"], "reverse_kl")
        for flag in ("--write-position-scope", "--write-effect-sampling",
                     "--write-position-limit", "--write-lambda-schedule", "--reverse-kl"):
            self.assertIn(flag, resolved["command"])
        with patch("sys.argv", ["vred.launch", "dft", "--write-effect-sampling", "stratified", "--dry-run"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        with patch("sys.argv", ["vred.launch", "dft", "--reverse-kl", "--write-loss-type", "effect_js",
                                "--dry-run"]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()

    def test_effect_weighted_kd_flags_and_defaults(self):
        from vred.launch import main
        output = io.StringIO()
        with patch("sys.argv", ["vred.launch", "dft", "--dry-run", "--effect-weighted-kd",
                                "--lambda-effect-kd", "2.0"]), contextlib.redirect_stdout(output):
            main()
        resolved = json.loads(output.getvalue())
        training = resolved["training"]
        self.assertTrue(training["effect_weighted_kd"])
        self.assertEqual(training["lambda_effect_kd"], 2.0)
        self.assertIn("--effect-weighted-kd", resolved["command"])
        self.assertIn("--lambda-effect-kd", resolved["command"])
        default = io.StringIO()
        with patch("sys.argv", ["vred.launch", "dft", "--dry-run"]), contextlib.redirect_stdout(default):
            main()
        self.assertNotIn("effect_weighted_kd", json.loads(default.getvalue())["training"])
        with patch("sys.argv", ["vred.launch", "dft", "--lambda-effect-kd", "2", "--dry-run"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        with patch("sys.argv", ["vred.launch", "sft", "--effect-weighted-kd", "--dry-run"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        with patch("sys.argv", ["vred.launch", "dft", "--effect-weighted-kd", "--no-sparse-logits",
                                "--dry-run"]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        hidden = io.StringIO()
        with patch("sys.argv", ["vred.launch", "dft", "--dry-run", "--effect-weighted-kd",
                                "--effect-kd-weight-source", "hidden", "--effect-kd-floor", "0.15",
                                "--lambda-effect-kd", "0.5"]), contextlib.redirect_stdout(hidden):
            main()
        resolved = json.loads(hidden.getvalue())
        self.assertEqual(resolved["training"]["effect_kd_weight_source"], "hidden")
        self.assertEqual(resolved["training"]["effect_kd_floor"], 0.15)
        self.assertEqual(resolved["training"]["lambda_effect_kd"], 0.5)
        self.assertIn("--effect-kd-weight-source", resolved["command"])
        self.assertIn("--effect-kd-floor", resolved["command"])
        for extra in (["--effect-kd-weight-source", "hidden"],
                      ["--effect-weighted-kd", "--effect-kd-floor", "0.15"],
                      ["--effect-weighted-kd", "--effect-kd-weight-source", "hidden",
                       "--write-position-limit", "64"],
                      ["--effect-weighted-kd", "--effect-kd-weight-source", "hidden",
                       "--write-position-scope", "prompt"]):
            with patch("sys.argv", ["vred.launch", "dft", *extra, "--dry-run"]),                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

    def test_launch_accepts_stage_option_and_rejects_conflicts(self):
        from vred.launch import main
        outputs = []
        for selector in (["dpt"], ["--stage", "dpt"]):
            stream = io.StringIO()
            with patch("sys.argv", ["vred.launch", *selector, "--gpus", "6", "--micro-batch-size", "32",
                                    "--accumulation", "1", "--dry-run"]), contextlib.redirect_stdout(stream):
                main()
            outputs.append(json.loads(stream.getvalue()))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["global_batch"], 192)
        for selector in ([], ["dpt", "--stage", "dft"]):
            with patch("sys.argv", ["vred.launch", *selector, "--dry-run"]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

    def test_training_flags_affect_resolved_run_and_position_budget(self):
        destination = self.root / "flag-run"
        command = ["vred.train", "--stage", "dpt", "--foundation-config", self.foundation,
                   "--config", self.training, "--output", str(destination), "--max-steps", "1",
                   "--seed", "42", "--epochs", "3", "--lr", "0.005", "--micro-batch-size", "1",
                   "--accumulation", "2", "--lambda-write", "0.2", "--write-loss-type", "effect_js",
                   "--query-chunk-size", "1", "--write-position-limit", "2", "--no-checkpoint-probes",
                   "--no-sparse-logits", "--selective-checkpointing"]
        with patch("sys.argv", command), contextlib.redirect_stdout(io.StringIO()):
            train_main()
        run = json.loads((destination / "run_config.json").read_text())
        self.assertEqual(run["training"]["stages"]["dpt"]["lr"], 0.005)
        self.assertEqual(run["training"]["stages"]["dpt"]["lambda_write"], 0.2)
        self.assertEqual(run["training"]["epochs"], 3)
        self.assertEqual(run["seed"], 42)
        self.assertFalse(run["training"]["checkpoint_probes"])
        self.assertFalse(run["training"]["sparse_logits"])
        self.assertTrue(run["training"]["selective_checkpointing"])
        logs = [json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["write_positions"] for row in logs), 2)
        self.assertEqual(logs[-1]["optimizer_steps"], 1)
        self.assertEqual(logs[-1]["global_examples"], 2)

    def test_gradient_checkpointing_is_sft_only(self):
        with self.assertRaises(SystemExit):
            self.train(self.root / "gc-dpt", 1, extra_flags=("--gradient-checkpointing",))
        predecessor = self.root / "gc-predecessor"
        self.train(predecessor, 1)
        output = self.root / "gc-sft"
        self.train(output, 1, stage="sft", input_model=predecessor,
                   extra_flags=("--gradient-checkpointing", "--no-tensorboard"))
        run = json.loads((output / "run_config.json").read_text())
        self.assertTrue(run["training"]["gradient_checkpointing"])
        self.assertTrue((output / "checkpoint_complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
