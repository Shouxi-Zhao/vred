"""Compact ten-benchmark metrics for the evaluation adapter.

The output contains only the ten benchmark metrics plus ``avg10`` and ``avg7``.
It never copies statuses, answer counts, model names, or timestamps.
"""

import json
from pathlib import Path

BENCHMARKS = (
    "vqav2",
    "gqa",
    "vizwiz",
    "scienceqa",
    "textvqa",
    "mme",
    "mmbench",
    "mmbench_cn",
    "pope",
    "mmmu",
)
# The current seven-benchmark average excludes VQAv2, POPE, and MMMU.
AVG7 = (
    "gqa",
    "vizwiz",
    "scienceqa",
    "textvqa",
    "mme",
    "mmbench",
    "mmbench_cn",
)


def _get(mapping, key):
    """Case-insensitive dict lookup, returning None when absent."""
    if not isinstance(mapping, dict):
        return None
    wanted = key.casefold()
    for name, value in mapping.items():
        if str(name).casefold() == wanted:
            return value
    return None


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _list_values(metrics, label):
    """All numeric values named ``label`` in a scorer-log metric list."""
    if not isinstance(metrics, list):
        return []
    wanted = label.casefold()
    return [float(item["value"]) for item in metrics
            if isinstance(item, dict) and str(item.get("name", "")).strip().casefold() == wanted
            and _is_number(item.get("value"))]


def _scalar(metrics, keys=("accuracy", "acc")):
    """Read one scalar metric from a scores.json dict or scorer-log list."""
    if isinstance(metrics, dict):
        for key in keys:
            value = _get(metrics, key)
            if _is_number(value):
                return float(value)
        nested = _get(metrics, "metrics")
        if isinstance(nested, dict):
            for key in keys:
                value = _get(nested, key)
                if _is_number(value):
                    return float(value)
    for key in keys:
        values = _list_values(metrics, key)
        if values:
            return values[0]
    raise ValueError(f"no {'/'.join(keys)} metric in {type(metrics).__name__}")


def _mme_metric(metrics):
    value = _get(metrics, "total") if isinstance(metrics, dict) else None
    if not _is_number(value):
        values = _list_values(metrics, "total")
        value = values[0] if values else None
    if not _is_number(value):
        raise ValueError("no total metric in MME scores")
    return float(value) / 20.0


def _mmmu_metric(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("MMMU scores are not a dict")
    overall = _get(metrics, "overall")
    value = None
    if isinstance(overall, dict):
        value = _get(overall, "acc")
    elif _is_number(overall):
        value = overall
    if not _is_number(value):
        value = _get(metrics, "overall_acc")
    if not _is_number(value):
        value = _get(metrics, "accuracy")
    if not _is_number(value):
        value = _get(metrics, "acc")
    if not _is_number(value):
        raise ValueError("no overall/acc metric in MMMU scores")
    value = float(value)
    # The scorer stores a 0-1 accuracy; the benchmark metric is a percentage.
    return value * 100.0 if value <= 1.0 else value


def _pope_metric(metrics):
    # eval_pope.py prints one accuracy per category (0-1), with no aggregate.
    values = _list_values(metrics, "accuracy")
    if isinstance(metrics, dict):
        category_values = _get(metrics, "category_accuracies")
        if isinstance(category_values, (list, tuple)):
            values.extend(float(value) for value in category_values if _is_number(value))
    if values:
        value = sum(values) / len(values)
    else:
        value = _scalar(metrics)
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def _normalize(name, metrics):
    if name == "mme":
        value = _mme_metric(metrics)
    elif name == "mmmu":
        value = _mmmu_metric(metrics)
    elif name == "pope":
        value = _pope_metric(metrics)
    else:
        value = _scalar(metrics)
    return round(float(value), 4)


def summarize_metrics(records):
    """Return exactly the ten metrics plus ``avg10`` and ``avg7``.

    Raises ``ValueError`` when a benchmark has no usable metric rather than
    quietly averaging a subset.
    """
    by_name = {}
    for record in records:
        name = record.get("benchmark")
        if name in BENCHMARKS:
            by_name[name] = record

    metrics = {}
    missing = []
    for name in BENCHMARKS:
        record = by_name.get(name)
        if not record or record.get("metrics") is None:
            missing.append(name)
            continue
        try:
            metrics[name] = _normalize(name, record["metrics"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"{name}: {error}") from error
    if missing:
        raise ValueError("missing metrics for: " + ", ".join(missing))

    metrics["avg10"] = round(sum(metrics[name] for name in BENCHMARKS) / len(BENCHMARKS), 4)
    metrics["avg7"] = round(sum(metrics[name] for name in AVG7) / len(AVG7), 4)
    return metrics


def write_metrics(records, output):
    """Write ``<output>/metrics.json`` and return the same compact mapping."""
    metrics = summarize_metrics(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
