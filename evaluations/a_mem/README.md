# A-Mem on LoCoMo with Gemini Flash

This folder runs [A-Mem's official reproduction code](https://github.com/WujiangXu/A-mem)
on its LoCoMo dataset with `gemini-3.5-flash` and `gemini-3.8-flash` through
Google Vertex AI. `upstream/COMMIT` records the copied source revision. Its
`LICENSE`, dataset, memory layers, parser, and original evaluation script are
included under `upstream/` for provenance.

## Setup and run

From the project root:

```bash
bash evaluations/a_mem/setup.sh
gcloud auth application-default login
evaluations/a_mem/.venv/bin/python evaluations/a_mem/run.py --check-api
evaluations/a_mem/.venv/bin/python evaluations/a_mem/run.py
```

`setup.sh` creates `.venv/`, installs a CPU-only PyTorch build and pinned Python
packages, and caches `all-MiniLM-L6-v2`. All of these files stay under
`evaluations/a_mem/`. The runner uses the Google Cloud project selected by
`GOOGLE_CLOUD_PROJECT`, ADC, or `gcloud config`; set `GOOGLE_CLOUD_LOCATION` if
the default `global` location is unsuitable. Authentication is managed by
Google Cloud outside this evaluation folder.

The default command runs all 10 conversations and 1,986 questions for **each**
model. A-Mem ingests all 5,882 turns for each model and may make several API
calls per turn. It runs two independent conversations per model at a time
(four workers total). Each worker ingests its own conversation in order, then
answers that conversation's questions. For a smaller pilot, run:

```bash
evaluations/a_mem/.venv/bin/python evaluations/a_mem/run.py --run pilot --conversations 1 --questions 5
```

Use `--models gemini-3.8-flash` to select one model. `--workers-per-model N`
changes the number of concurrent conversations for each selected model;
reduce it to 1 if Vertex rate limits the run. `--retrieve-k` defaults to 10.
Repeating the command resumes each conversation from its last saved session
and answer. Worker count can change when resuming. Changes to evaluation
settings or code require a different `--run` name so cached memories cannot
be mixed across configurations.

The default run checks access to **both** models before creating results or
starting ingestion. `--check-api` performs that check by itself with one small
request per model.

Outputs are isolated at `evaluations/a_mem/.runs/<run>/<model>/`:
`config.json` records the source and data hashes, `memory/` holds session
checkpoints, and `traces/<conversation>.jsonl` holds each conversation's scored
answers and retrieved context. After every conversation for a model finishes,
the runner merges those traces in dataset order into `trace.jsonl` and writes
`summary.json` with overall and category scores. Runtime files are ignored by
Git.

## Evaluation choices

- `run.py` uses A-Mem's robust note creation, memory evolution, embedding
  retriever, query keyword generation, and category prompts. The local Gemini
  controller is the only new backend in `upstream/memory_layer_robust.py`.
- Google says Gemini 3.8 Flash rejects `temperature`, so the controller omits
  it for **both** models. Both use the provider's default thinking setting and
  a 32,000 token output limit so thinking tokens do not truncate short answers.
- The upstream category 5 prompt includes the gold answer among answer choices.
  This runner asks for abstention when evidence is absent and never includes
  the gold answer in the prompt.
- Scores use the LoCoMo paper's category-specific F1 and abstention rule, the
  same rule used by this project's LoCoMo benchmark. This differs from the
  upstream script's generic F1, BLEU, ROUGE, and BERTScore bundle.
- API failures stop scheduling more conversations for the affected model.
  Active workers finish, and session checkpoints and question traces let the
  same command resume after the service is available again.

Run local validation without API calls with:

```bash
evaluations/a_mem/.venv/bin/python evaluations/a_mem/run.py --check
evaluations/a_mem/.venv/bin/python -m unittest discover -s evaluations/a_mem -p 'test_run.py'
```
