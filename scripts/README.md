# scripts: dataset generation and older pipeline runs

Benchmark runners are no longer here; they moved to [benchmarks/](../benchmarks/).

## Dataset generation

| File | What it does |
|---|---|
| [generate_eval_dataset.py](generate_eval_dataset.py) | Builds the synthetic dataset in `datasets/eval/`: a preference schedule with cycles and transitions, daily conversations covering every domain, and typed QA pairs. |
| [generate_learning_conversations.py](generate_learning_conversations.py) | Generates daily conversations with structured LLM output (earlier dataset format). |
| [generate_insights.py](generate_insights.py) | Earlier insight generation over extracted memories (builds an entity normalisation map). No module docstring; check the file before relying on it. |
| `generate_activities/` | `generate_activities.py` and its output `activities_dataset_full.csv`, an activity catalogue used when generating conversations. |

## Older memory_v2 pipeline runs

Each runs memory_v2 end to end on the synthetic dataset and writes to `results/legacy/`.

| File | Model | Output |
|---|---|---|
| [run_full_gpt5_pipeline.py](run_full_gpt5_pipeline.py) | gemini-3-flash-preview (file name is historical) | `results/legacy/memory_v2_run/` |
| [run_gemini_extraction.py](run_gemini_extraction.py) | gemini-3.1-pro-preview | `results/legacy/gemini_extraction_run/` |
| [run_gemini_flash.py](run_gemini_flash.py) | gemini-3.1-flash-lite | `results/legacy/gemini_flash_run/` |
| [verify_extraction.py](verify_extraction.py) | — | Checks extracted memories against the dataset's ground-truth patterns; `results/legacy/verify_run/` |
