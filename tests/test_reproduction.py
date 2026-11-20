import copy
import json
import statistics
from dataclasses import asdict

import pytest
import torch

from consensusalign.core import ConsensusAlign
from consensusalign.data import read_tensor
from consensusalign.datasets import sha256_file
from consensusalign.reproduce import atomic_json, read_plan, run, summarize, tasks


@pytest.fixture
def plan(corpus, wp):
    path = corpus.parent / "plan.json"
    path.write_text(
        json.dumps(
            {
                "name": "synthetic-test",
                "train_manifest": corpus.name,
                "evaluation_manifest": corpus.name,
                "seed_indices": [0, 1],
                "anchor_budgets": [0, 1],
                "latent_dim": 4,
                "wp": asdict(wp),
                "evaluation_block_size": 16,
            }
        )
    )
    return read_plan(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("weighting", "other"),
        ("latent_dim", 0),
        ("seed_indices", [0, 0]),
        ("seed_indices", [True]),
        ("anchor_budgets", [-1]),
        ("unknown", 1),
    ],
)
def test_invalid_plan_fields(corpus, field, value):
    path = corpus.parent / "invalid.json"
    path.write_text(
        json.dumps({"train_manifest": corpus.name, "evaluation_manifest": corpus.name, field: value})
    )
    with pytest.raises(ValueError):
        read_plan(path)


def test_dry_run_requires_no_data(plan, tmp_path):
    plan["train_manifest"] = tmp_path / "not-downloaded.json"
    result = run(plan, tmp_path / "not-created", dry_run=True)
    assert result["bridge_fits"] == 4
    assert not (tmp_path / "not-created").exists()


def test_run_resume_partial_recovery_and_input_identity(plan, tmp_path, monkeypatch):
    output = tmp_path / "runs"
    summary = run(plan, output)
    assert len(summary["groups"]) == 2
    for group in summary["groups"]:
        assert group["n_seeds"] == 2 and group["pairs_per_seed"] == 9 and group["n_cells"] == 18
        assert group["sd_foscttm"] == pytest.approx(
            statistics.stdev(row["mean"] for row in group["seed_means"])
        )

    def no_fitting(*args, **kwargs):
        pytest.fail("Resumption must reuse the completed or recovered checkpoint")

    monkeypatch.setattr(ConsensusAlign, "fit", no_fitting)
    assert run(plan, output) == summary
    directory = output / tasks(plan)[0]["directory"]
    record = json.loads((directory / "run.json").read_text())
    record["status"] = "fitted"
    atomic_json(directory / "run.json", record)
    (directory / "alignment.json").unlink()
    assert run(plan, output) == summary
    assert not list(output.rglob(".run.lock"))
    complete_record = json.loads((directory / "run.json").read_text())
    checkpoint = directory / "model.pt"
    checkpoint_bytes = checkpoint.read_bytes()
    complete_record["status"] = "fitted"
    atomic_json(directory / "run.json", complete_record)
    checkpoint.write_bytes(checkpoint_bytes + b"changed")
    with pytest.raises(ValueError, match="fitted checkpoint changed"):
        run(plan, output)
    assert not (directory / ".run.lock").exists()
    checkpoint.write_bytes(checkpoint_bytes)
    complete_record["status"] = "complete"
    atomic_json(directory / "run.json", complete_record)
    artifact = directory / "alignment.json"
    artifact.write_text(artifact.read_text() + " ")
    with pytest.raises(ValueError, match="artifact changed"):
        run(plan, output)
    spec = json.loads(plan["train_manifest"].read_text())["vision"]["vision_0"]
    feature_path = plan["train_manifest"].parent / spec
    torch.save(read_tensor(feature_path) + 0.01, feature_path)
    with pytest.raises(ValueError, match="Input data or settings changed"):
        run(plan, output)


def test_lock_prevents_overlapping_fits(plan, tmp_path):
    output = tmp_path / "runs"
    directory = output / tasks(plan)[0]["directory"]
    directory.mkdir(parents=True)
    (directory / ".run.lock").touch()
    with pytest.raises(RuntimeError, match="locked"):
        run(plan, output)
    assert (directory / ".run.lock").exists()


def write_summary_fixtures(plan, output, seed_values):
    for task in tasks(plan):
        directory = output / task["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        values = seed_values[task["seed_index"]]
        rows = [
            {"vision": f"vision_{v}", "text": f"text_{t}", "foscttm": values[v * 3 + t]}
            for v in range(3)
            for t in range(3)
        ]
        atomic_json(directory / "alignment.json", {"task": "alignment", "results": rows})
        atomic_json(
            directory / "run.json",
            {
                **task,
                "status": "complete",
                "protocol": {
                    key: plan[key] for key in ("latent_dim", "weighting", "wp", "evaluation_block_size")
                },
                "evaluation_sha256": sha256_file(directory / "alignment.json"),
            },
        )


def test_summary_uses_sd_of_seed_means_and_complete_pair_coverage(plan, tmp_path):
    plan["anchor_budgets"] = [0]
    output = tmp_path / "summary"
    write_summary_fixtures(plan, output, {0: [0.01] * 8 + [0.91], 1: [0.02] * 9})
    result = summarize(output, plan)["groups"][0]
    assert result["mean_foscttm"] == pytest.approx(0.065)
    assert result["sd_foscttm"] == pytest.approx(statistics.stdev([0.11, 0.02]))
    assert result["failed_cells"] == 1 and result["worst_foscttm"] == 0.91
    directory = output / tasks(plan)[0]["directory"]
    values = json.loads((directory / "alignment.json").read_text())
    values["results"].pop()
    atomic_json(directory / "alignment.json", values)
    record = json.loads((directory / "run.json").read_text())
    record["evaluation_sha256"] = sha256_file(directory / "alignment.json")
    atomic_json(directory / "run.json", record)
    with pytest.raises(ValueError, match="every encoder pair"):
        summarize(output, plan)


def test_single_seed_sd_is_null_not_zero(plan, tmp_path):
    one_seed = copy.deepcopy(plan)
    one_seed["seed_indices"] = [0]
    one_seed["anchor_budgets"] = [0]
    output = tmp_path / "single"
    write_summary_fixtures(one_seed, output, {0: [0.01] * 9})
    assert summarize(output, one_seed)["groups"][0]["sd_foscttm"] is None
