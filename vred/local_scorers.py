"""Local (no submission) VQAv2/GQA/VizWiz accuracy and legacy-metric parsing.

All functions write ``scores.json`` into the benchmark output directory.  They
never invent a value: VQAv2/GQA/VizWiz read local annotations, MME/MMMU parse
the stdout of the supplied legacy tools.
"""

import ast
import json
import re
from pathlib import Path


def _read_predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if any("question_id" not in row or "text" not in row for row in rows):
        raise ValueError(f"{path}: predictions require question_id and text")
    identifiers = [str(row["question_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: duplicate question_id values")
    return rows


def _write(output_dir, payload):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def score_vqav2(answers_path, annotations_path, output_dir):
    """Official VQAv2 soft accuracy on the local val2014 annotations.

    Accuracy per question is ``min(number of matching gold answers / 3, 1)``
    after the EvalAI answer normalization used by the legacy pipeline.
    """
    from llavakd.eval.m4c_evaluator import EvalAIAnswerProcessor

    annotations = json.loads(Path(annotations_path).read_text())["annotations"]
    gold = {str(item["question_id"]): [answer["answer"] for answer in item["answers"]]
            for item in annotations}
    if len(gold) != len(annotations):
        raise ValueError(f"{annotations_path}: duplicate question ids")
    predictions = _read_predictions(answers_path)
    missing = [row["question_id"] for row in predictions if str(row["question_id"]) not in gold]
    if missing:
        raise ValueError(f"{answers_path}: {len(missing)} predictions missing from annotations, e.g. {missing[:3]}")
    if len(predictions) != len(gold):
        raise ValueError(f"{answers_path}: {len(predictions)} predictions vs {len(gold)} annotations")

    processor = EvalAIAnswerProcessor()
    scores = []
    for row in predictions:
        prediction = processor(str(row["text"]))
        answers = [processor(str(answer)) for answer in gold[str(row["question_id"])]]
        scores.append(min(sum(answer == prediction for answer in answers) / 3.0, 1.0))
    accuracy = 100.0 * sum(scores) / len(scores)
    return _write(output_dir, dict(dataset="vqav2", split="val2014", metric="vqa_soft_accuracy",
                                   accuracy=accuracy, count=len(scores), missing=0,
                                   annotations=str(annotations_path)))


def score_gqa(answers_path, questions_path, output_dir):
    """GQA balanced exact-match accuracy from the local official questions file.

    The legacy pipeline lowercases predictions and strips one trailing period;
    the same normalization is applied here for comparability.
    """
    questions = json.loads(Path(questions_path).read_text())
    predictions = {str(row["question_id"]): row["text"] for row in _read_predictions(answers_path)}
    missing = sorted(set(questions) - set(predictions))
    extra = sorted(set(predictions) - set(questions))
    if missing or extra:
        raise ValueError(f"{answers_path}: missing {len(missing)} / extra {len(extra)} question ids; "
                         f"examples missing={missing[:3]} extra={extra[:3]}")

    balanced, structural, semantic = [], {}, {}
    correct_all = 0
    for question_id, question in questions.items():
        prediction = str(predictions[question_id]).strip().rstrip(".").lower()
        gold = str(question["answer"]).strip().lower()
        correct = float(prediction == gold)
        correct_all += int(correct)
        if question.get("isBalanced", False):
            balanced.append(correct)
            types = question.get("types", {})
            for name, target in (("structural", structural), ("semantic", semantic)):
                key = str(types.get(name, "unknown"))
                entry = target.setdefault(key, [0, 0])
                entry[0] += int(correct)
                entry[1] += 1
    if not balanced:
        raise ValueError(f"{questions_path}: no isBalanced questions")
    result = dict(
        dataset="gqa", split="testdev_balanced", metric="exact_match",
        accuracy=100.0 * sum(balanced) / len(balanced), count=len(balanced),
        balanced=len(balanced), all_questions=len(questions), all_correct=correct_all,
        all_accuracy=100.0 * correct_all / len(questions), missing=0,
        accuracy_per_structural_type={key: dict(accuracy=100.0 * correct / total, num=total)
                                     for key, (correct, total) in sorted(structural.items())},
        accuracy_per_semantic_type={key: dict(accuracy=100.0 * correct / total, num=total)
                                    for key, (correct, total) in sorted(semantic.items())},
        questions=str(questions_path))
    return _write(output_dir, result)


def _predicted_choice(text, choices):
    """Extract the first standalone choice letter, as the legacy MMBench scorer does."""
    matches = [match for match in re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", str(text).upper())
               if match in choices]
    return matches[0] if matches else ""


def score_mmbench(answers_path, annotations_path, output_dir, dataset="mmbench"):
    """Local MMBench/MBench-CN dev accuracy from the TSV ``answer`` column.

    MMBench dev TSVs carry the correct option, so no server submission is
    needed.  This mirrors the previous local scorer: the prediction is the first
    standalone uppercase option letter present in ``text``, and it is compared
    exactly with the TSV answer.  ``unparsed`` counts predictions with no
    extractable option; ``missing`` counts question ids absent from the TSV.
    """
    import pandas as pd

    annotations = pd.read_table(annotations_path, usecols=["index", "A", "B", "C", "D", "answer"])
    gold, choices = {}, {}
    for _, row in annotations.iterrows():
        question_id = str(row["index"])
        if pd.notna(row["answer"]):
            gold[question_id] = str(row["answer"]).upper()
        choices[question_id] = [choice for choice in "ABCD" if pd.notna(row[choice])]
    predictions = _read_predictions(answers_path)
    correct = count = missing = unparsed = 0
    for row in predictions:
        question_id = str(row["question_id"])
        if question_id not in gold:
            missing += 1
            continue
        count += 1
        predicted = _predicted_choice(row["text"], choices[question_id])
        if not predicted:
            unparsed += 1
        correct += int(predicted == gold[question_id])
    accuracy = 100.0 * correct / count if count else 0.0
    return _write(output_dir, dict(dataset=dataset, split="dev", metric="exact_match",
                                   accuracy=accuracy, correct=correct, count=count,
                                   missing=missing, unparsed=unparsed,
                                   annotations=str(annotations_path)))


def score_vizwiz(answers_path, annotations_path, output_dir):
    """VizWiz val soft accuracy on the local ``Annotations/val.json``.

    This mirrors the previous local pipeline's ``TextVQAAccuracyEvaluator``:
    answers are normalized with ``EvalAIAnswerProcessor`` and each of the ten
    annotator answers is scored against the other nine with
    ``min(matching / 3, 1)``.  The per-position scores are averaged; a
    prediction absent from the unique answer set scores zero.  Predictions
    with an unknown ``question_id`` are counted as missing, matching the
    previous scorer instead of silently inventing a score.
    """
    from llavakd.eval.m4c_evaluator import EvalAIAnswerProcessor

    annotations = json.loads(Path(annotations_path).read_text())
    gold = {str(index): [answer["answer"] for answer in item["answers"]]
            for index, item in enumerate(annotations)}
    if len(gold) != len(annotations):
        raise ValueError(f"{annotations_path}: duplicate annotation indices")
    predictions = _read_predictions(answers_path)
    processor = EvalAIAnswerProcessor()
    scores, missing = [], 0
    for row in predictions:
        question_id = str(row["question_id"])
        if question_id not in gold:
            missing += 1
            continue
        answers = [processor(str(answer)) for answer in gold[question_id]]
        if not answers:
            raise ValueError(f"{annotations_path}: question {question_id} has no gold answers")
        prediction = processor(str(row["text"]))
        per_annotator = [
            min(sum(answer == prediction for answer in answers[:index] + answers[index + 1:]) / 3.0, 1.0)
            for index in range(len(answers))
        ]
        scores.append(sum(per_annotator) / len(per_annotator))
    accuracy = 100.0 * sum(scores) / len(scores) if scores else 0.0
    return _write(output_dir, dict(dataset="vizwiz", split="val", metric="vizwiz_soft_accuracy",
                                   accuracy=accuracy, count=len(scores), missing=missing,
                                   annotations=str(annotations_path)))


def parse_mme_log(text):
    """Parse the legacy ``calculation.py`` stdout into category/task scores."""
    categories, current = {}, None
    for line in text.splitlines():
        heading = re.match(r"^=+\s+(\w+)\s+=+$", line.strip())
        if heading:
            current = categories.setdefault(heading.group(1), dict(total=None, tasks={}))
            continue
        total = re.match(r"^total score:\s*([+-]?\d+(?:\.\d*)?)", line.strip())
        if total and current is not None:
            current["total"] = float(total.group(1))
            continue
        task = re.match(r"^\s+(\w+)\s+score:\s*([+-]?\d+(?:\.\d*)?)\s*$", line)
        if task and current is not None:
            current["tasks"][task.group(1)] = float(task.group(2))
    if not categories or any(entry["total"] is None or not entry["tasks"] for entry in categories.values()):
        raise ValueError("Could not parse MME calculation output")
    totals = {name: entry["total"] for name, entry in categories.items()}
    return dict(metric="mme_score", categories=categories, categories_total=totals,
                total=sum(totals.values()))


def parse_mmmu_stdout(text):
    """Parse ``main_eval_only.py``'s printed ``{category: {num, acc}}`` dict."""
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict) or "Overall" not in parsed:
            continue
        overall = parsed["Overall"]
        return dict(metric="mmmu_accuracy", categories=parsed,
                    overall=dict(num=overall.get("num"), acc=overall.get("acc")))
    raise ValueError("Could not parse MMMU evaluation output")
