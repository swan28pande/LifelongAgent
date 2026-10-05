"""Keep experiment inputs identical to the canonical generated dataset."""

import shutil

from . import config

DATA_FILES = (
    "world_state.json", "qa_pool.json", "qa_pairs.json", "probing_questions.json",
    "conversations.json", "stats.json", "fidelity.json", "simulate_report.txt",
)


def sync_outputs(user_id: str) -> None:
    source = config.OUTPUT_DIR / user_id
    target = config.EXPERIMENT_DATA_DIR / user_id
    target.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        if (source / name).exists():
            shutil.copyfile(source / name, target / name)
