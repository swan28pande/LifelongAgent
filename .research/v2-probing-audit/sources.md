# Source inventory

All evidence is from the local checkout. No web sources or paid API calls were used, and no dataset files were rewritten. Read-only calls to the existing builders and simulator reproduced their outputs in memory. `manifest.json` records primary input hashes; the family/source inventories also hash personas, individual sessions, and additional source files.

- `datasets/v2/u1..u5/probing_questions.json`: canonical 1,000 scheduled question instances, references, grading, viewpoint dates, and retrospective links.
- `datasets/v2/u1..u5/qa_pool.json` and `qa_pairs.json`: source question payloads and legacy final-viewpoint selections.
- `datasets/v2/u1..u5/world_state.json`: daily synthetic truth, resolved preference regimes, effective/stated fact and event dates, sampling, and exceptions.
- `datasets/v2/u1..u5/conversations.json` and `sessions/*.json`: actual recorded text, session metadata, stored validator extraction, and validation failures.
- `datasets/v2/u1..u5/fidelity.json`: aggregate record of retained conversation-generation flags; these require independent interpretation.
- `generator_v2/DECISIONS.md`: original QA semantics, truth/knowledge distinction, cycle anchors, inference, and conversational sampling rules.
- `generator_v2/probing.py`: monthly scheduling, evidence eligibility, sampling cap, unchanged source references, and retrospective copying.
- `generator_v2/qa.py`, `rules.py`, `schema.py`, `simulator.py`, `checks.py`: original question construction and synthetic truth definitions.
- `generator_v2/conversation.py` and `validator.py`: required statements, writer constraints, validator interpretation, and stored failures.
- `experiments/data/synthetic_v2/u1..u5/`: executable experiment copies compared with the canonical data.
- `experiments/benchmarks/v2.py`, `core/runner.py`, `core/grading.py`: what the current evaluator actually loads, ingests, asks, and grades.
