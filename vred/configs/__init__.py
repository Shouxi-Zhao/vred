import json
from pathlib import Path


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name("foundation.json")
    config = json.loads(path.read_text())
    root = Path(__file__).resolve().parents[2]
    for key in ("dpt_json", "sft_json", "cache_index", "student", "teacher", "teacher_base"):
        config[key] = str(root / config[key])
    return config


def load_training_config(path=None):
    path = Path(path) if path else Path(__file__).with_name("training.json")
    return json.loads(path.read_text())
