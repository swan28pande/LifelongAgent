"""Small offline integration checks for the local A-Mem runner."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import run
from memory_layer_robust import RobustGeminiVertexController
from scoring import score_answer


def fake_completion(_self, prompt, temperature=0.7):
    if prompt.startswith("Analyze the following content"):
        return "KEYWORDS: art, hiking, weekend\nCONTEXT: A weekend activity.\nTAGS: activity, personal, chat"
    if "memory evolution agent" in prompt:
        return "DECISION: NO_EVOLUTION\nREASON: Separate memory"
    if "generate several keywords" in prompt:
        return "hiking, weekend"
    return "I don't know."


def fake_parallel_worker(args, model, sample, run_dir_name):
    """Picklable offline worker for the real spawn-based scheduler."""
    sample_id = sample["sample_id"]
    trace = Path(run_dir_name) / "traces" / f"{sample_id}.jsonl"
    existing = run._read_records(trace)
    for index, qa in run._questions(sample, args):
        if (sample_id, index) not in existing:
            run._append_record(trace, {"sample_id": sample_id, "qa_index": index,
                                       "category": qa["category"], "score": 1.0})
    return sample_id


def fake_failing_worker(args, model, sample, run_dir_name):
    """Leave one question checkpoint before a single synthetic worker failure."""
    sample_id = sample["sample_id"]
    trace = Path(run_dir_name) / "traces" / f"{sample_id}.jsonl"
    marker = Path(run_dir_name) / "traces" / "failed_once"
    if model == run.MODELS[0] and sample_id == "first" and not marker.exists():
        run._append_record(trace, {"sample_id": sample_id, "qa_index": 0,
                                   "category": 1, "score": 1.0})
        marker.touch()
        raise RuntimeError("synthetic failure")
    return fake_parallel_worker(args, model, sample, run_dir_name)


class RunnerTests(unittest.TestCase):
    def test_paper_scoring(self):
        self.assertEqual(score_answer("I don't know.", "secret answer", 5), 1.0)
        self.assertEqual(score_answer("secret answer", "secret answer", 5), 0.0)
        self.assertEqual(score_answer("blue, green", "blue, green", 1), 1.0)

    def test_category_five_prompt_has_no_reference(self):
        class FakeLLM:
            prompts = []

            def get_completion(self, prompt, temperature=0.7):
                self.prompts.append(prompt)
                return "hiking" if "generate several keywords" in prompt else "I don't know."

        class FakeSystem:
            llm_controller = type("Controller", (), {"llm": FakeLLM()})()

            def find_related_memories_raw(self, query, k):
                return "Memory about hiking"

        system = FakeSystem()
        answer, _, _ = run._answer(system, "What did Caroline realize?", 5, 10)
        self.assertEqual(answer, "I don't know.")
        self.assertIn("I don't know", system.llm_controller.llm.prompts[-1])
        self.assertNotIn("self-care is important", system.llm_controller.llm.prompts[-1])

    @patch.object(RobustGeminiVertexController, "get_completion", fake_completion)
    def test_memory_checkpoint_and_reload(self):
        source = json.loads(run.DATASET.read_text())[0]
        conversation = source["conversation"]
        sample = {"sample_id": "tiny", "conversation": {
            "speaker_a": conversation["speaker_a"],
            "speaker_b": conversation["speaker_b"],
            "session_1": conversation["session_1"][:1],
            "session_1_date_time": conversation["session_1_date_time"],
            "session_2": conversation["session_2"][:1],
            "session_2_date_time": conversation["session_2_date_time"],
        }}
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".cache") as directory:
            cache = Path(directory)
            built = run._load_or_build_memory("gemini-3.5-flash", sample, cache)
            self.assertEqual(len(built.memories), 2)
            reloaded = run._load_or_build_memory("gemini-3.5-flash", sample, cache)
            self.assertEqual(len(reloaded.memories), 2)
            self.assertEqual(len(reloaded.retriever.corpus), 2)
            self.assertEqual(json.loads((cache / "progress.json").read_text())["next_session"], 2)

    @patch.object(RobustGeminiVertexController, "get_completion", fake_completion)
    def test_results_resume_without_duplicate_answers(self):
        source = json.loads(run.DATASET.read_text())[0]
        conversation = source["conversation"]
        sample = {"sample_id": "tiny", "conversation": {
            "speaker_a": conversation["speaker_a"],
            "speaker_b": conversation["speaker_b"],
            "session_1": conversation["session_1"][:1],
            "session_1_date_time": conversation["session_1_date_time"],
        }, "qa": [{"question": "What was the secret answer?", "category": 5,
                    "answer": None, "adversarial_answer": "private gold text"}]}
        args = Namespace(run="tiny", retrieve_k=10, conversations=1, questions=1,
                         categories=[1, 2, 3, 4, 5], workers_per_model=2)
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".cache") as directory:
            with patch.object(run, "RUNS", Path(directory)):
                result_dir = run._prepare_run(args, "gemini-3.5-flash")
                for _ in range(2):
                    run._run_sample(args, "gemini-3.5-flash", sample, str(result_dir))
                    run._merge_model(args, "gemini-3.5-flash", [sample], result_dir)
                lines = (result_dir / "trace.jsonl").read_text().splitlines()
                self.assertEqual(len(lines), 1)
                self.assertEqual(json.loads(lines[0])["score"], 1.0)
                self.assertEqual(json.loads((result_dir / "summary.json").read_text())["summary"]["n"], 1)

    def test_parallel_models_merge_in_dataset_order_and_resume(self):
        samples = [
            {"sample_id": "later", "qa": [{"category": 1}, {"category": 2}]},
            {"sample_id": "earlier", "qa": [{"category": 5}]},
        ]
        args = Namespace(run="parallel", models=list(run.MODELS), retrieve_k=10,
                         conversations=2, questions=None, categories=[1, 2, 3, 4, 5],
                         workers_per_model=2)
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".cache") as directory:
            with patch.object(run, "RUNS", Path(directory)):
                run._run_parallel(args, samples, fake_parallel_worker)
                args.workers_per_model = 1
                run._run_parallel(args, samples, fake_parallel_worker)
                for model in run.MODELS:
                    result_dir = Path(directory) / "parallel" / model
                    merged = [json.loads(line) for line in
                              (result_dir / "trace.jsonl").read_text().splitlines()]
                    self.assertEqual([(r["sample_id"], r["qa_index"]) for r in merged],
                                     [("later", 0), ("later", 1), ("earlier", 0)])
                    self.assertEqual(len((result_dir / "traces" / "later.jsonl")
                                         .read_text().splitlines()), 2)
                    self.assertEqual(json.loads((result_dir / "summary.json").read_text())
                                     ["summary"]["n"], 3)

    def test_failed_conversation_stops_its_model_and_resumes(self):
        samples = [
            {"sample_id": "first", "qa": [{"category": 1}, {"category": 2}]},
            {"sample_id": "second", "qa": [{"category": 5}]},
        ]
        args = Namespace(run="failure", models=list(run.MODELS), retrieve_k=10,
                         conversations=2, questions=None, categories=[1, 2, 3, 4, 5],
                         workers_per_model=1)
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".cache") as directory:
            with patch.object(run, "RUNS", Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    run._run_parallel(args, samples, fake_failing_worker)
                failed_dir = Path(directory) / "failure" / run.MODELS[0]
                other_dir = Path(directory) / "failure" / run.MODELS[1]
                self.assertFalse((failed_dir / "summary.json").exists())
                self.assertFalse((failed_dir / "traces" / "second.jsonl").exists())
                self.assertEqual(len((failed_dir / "traces" / "first.jsonl")
                                     .read_text().splitlines()), 1)
                self.assertEqual(json.loads((other_dir / "summary.json").read_text())
                                 ["summary"]["n"], 3)
                run._run_parallel(args, samples, fake_failing_worker)
                self.assertEqual(len((failed_dir / "traces" / "first.jsonl")
                                     .read_text().splitlines()), 2)
                self.assertEqual(json.loads((failed_dir / "summary.json").read_text())
                                 ["summary"]["n"], 3)


if __name__ == "__main__":
    unittest.main()
