import json
from pathlib import Path

import pytest

from bugforge.generator import REPOSITORY_DIFFICULTIES
from bugforge.provenance import config, fingerprint
from bugforge.study import expected_keys, main, read_plan, report


@pytest.fixture
def plan():
    return read_plan(Path(__file__).resolve().parents[1] / "study_plan.json")


def make_record(plan, model, key, solved, task_hash=None):
    family, seed = key
    manifest = {"provider": model["provider"], "model": model["model"],
                    "model_revision_claimed": model["revision"], "agent": {"max_steps": 30},
                    "runner": {"mode": "docker"}, "runtime": {}, "source_sha256": "source",
                    "tasks_sha256": "tasks", "generation_fuzz_seed": "task_seed",
                    "inference": {"temperature": 0}}
    return {"provider": model["provider"], "model": model["model"], "manifest": manifest,
                "experiment_id": fingerprint(manifest), "difficulty": family, "seed": seed,
                "config": config(REPOSITORY_DIFFICULTIES[family]), "sandbox": "docker",
                "task_sha256": task_hash or fingerprint(key), "solved": solved, "reward": int(solved),
                "steps": 3, "input_tokens": 4, "output_tokens": 2, "seconds": 1.0}


def write(directory, model, rows):
    (directory / (model["id"] + ".jsonl")).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_plan_and_empty_report(plan, tmp_path):
    assert len(expected_keys(plan)) == 60
    assert len(plan["models"]) == 3
    assert "0/60 | 0 | —" in report(plan, tmp_path)


def test_partial_pairing(plan, tmp_path):
    a, b, _ = plan["models"]
    keys = sorted(expected_keys(plan))[:3]
    write(tmp_path, a, [make_record(plan, a, k, i < 2) for i, k in enumerate(keys)])
    write(tmp_path, b, [make_record(plan, b, k, i == 1) for i, k in enumerate(keys[:2])])
    text = report(plan, tmp_path)
    assert "| qwen-small | qwen-medium | 2/60 | 1 | 1 | 0 | 0 | +50.0% |" in text


def test_task_mismatch(plan, tmp_path):
    a, b, _ = plan["models"]
    key = min(expected_keys(plan))
    write(tmp_path, a, [make_record(plan, a, key, True)])
    write(tmp_path, b, [make_record(plan, b, key, False, "a" * 64)])
    with pytest.raises(ValueError, match="task mismatch"):
        report(plan, tmp_path)


def test_manifest_mismatch(plan, tmp_path):
    a, b, _ = plan["models"]
    key = min(expected_keys(plan))
    r = make_record(plan, b, key, False)
    r["manifest"]["runtime"] = {"different": True}
    r["experiment_id"] = fingerprint(r["manifest"])
    write(tmp_path, a, [make_record(plan, a, key, True)])
    write(tmp_path, b, [r])
    with pytest.raises(ValueError, match="shared harness"):
        report(plan, tmp_path)


def test_reject_invalid_reward(plan, tmp_path):
    a = plan["models"][0]
    r = make_record(plan, a, min(expected_keys(plan)), True)
    r["reward"] = 0
    write(tmp_path, a, [r])
    with pytest.raises(ValueError, match="binary reward"):
        report(plan, tmp_path)


def test_docker_required(monkeypatch, tmp_path):
    monkeypatch.setattr("bugforge.study.DockerRunner.available", lambda: False)
    path = Path(__file__).resolve().parents[1] / "study_plan.json"
    with pytest.raises(RuntimeError, match="Docker unavailable"):
        main(["--plan", str(path), "--directory", str(tmp_path), "run", "qwen-small",
              "--max-tokens", "100"])
    assert not list(tmp_path.iterdir())


def test_reject_paid_without_opt_in(plan, tmp_path):
    plan["models"][0]["provider"] = "anthropic"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        main(["--plan", str(path), "run", "qwen-small", "--max-tokens", "100"])
    assert err.value.code == 2


def test_invalid_seeds(plan, tmp_path):
    plan["seeds"] = [0, 0]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        read_plan(path)
