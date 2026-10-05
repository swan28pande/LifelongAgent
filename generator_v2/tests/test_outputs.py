from generator_v2 import config
from generator_v2.outputs import sync_outputs


def test_dataset_copies_are_updated_without_copying_credentials_or_api_logs(tmp_path, monkeypatch):
    source, target = tmp_path / "canonical", tmp_path / "experiment"
    monkeypatch.setattr(config, "OUTPUT_DIR", source)
    monkeypatch.setattr(config, "EXPERIMENT_DATA_DIR", target)
    (source / "u1").mkdir(parents=True)
    (target / "u1").mkdir(parents=True)
    (source / "u1/qa_pool.json").write_text('[{"id": "new"}]')
    (target / "u1/qa_pool.json").write_text('[{"id": "old"}]')
    (source / "u1/llm_log.jsonl").write_text("source API log")
    (target / "u1/llm_log.jsonl").write_text("existing API log")
    sync_outputs("u1")
    assert (target / "u1/qa_pool.json").read_bytes() == (source / "u1/qa_pool.json").read_bytes()
    assert (target / "u1/llm_log.jsonl").read_text() == "existing API log"
