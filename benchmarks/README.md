# Benchmarks

One folder per benchmark, one runner per system. Every runner writes to
`results/<benchmark>/<run>/`, where `<run>` defaults to the system name and can be
changed with `--run`.

| Benchmark | Folder | What it tests | Systems |
|---|---|---|---|
| LoCoMo | [locomo/](locomo/) | Long two-person conversations (months, ~19 sessions each): facts, dates, multi-hop, adversarial questions | v3 agentic, v3 one-pass, mem0 |
| Synthetic | [synthetic/](synthetic/) | One user over 60 days whose preferences change on cycles: recall, transitions, patterns, prediction | v3 |

## Conventions every runner follows

- **Run from the project root** with `venv/bin/python benchmarks/<benchmark>/<runner>.py`.
- **A trace line per answer.** Each question's record (question, gold answer,
  response, score, plus what was retrieved or which tools were called) is appended to a
  `*_trace.jsonl` file as soon as it is answered, so an interrupted run loses nothing.
- **Same model and embeddings across systems:** Gemini (`--model`, default
  `gemini-3.5-flash`) on Vertex, `nomic-embed-text-v1` embeddings, so differences come
  from the memory system and not the model.
