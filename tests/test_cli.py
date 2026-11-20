import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from consensusalign import ConsensusAlign
from consensusalign.data import load_features, read_manifest, seed_at


def cli(directory, *arguments):
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "consensusalign", *map(str, arguments)],
        cwd=directory,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.integration
def test_installed_cli_complete_workflow(corpus, tmp_path):
    model_path = tmp_path / "model.pt"
    cli(
        tmp_path,
        "fit",
        "--manifest",
        corpus,
        "--output",
        model_path,
        "--latent-dim",
        4,
        "--clusters",
        4,
        "--restarts",
        2,
        "--workers",
        2,
        "--batch-size",
        20,
        "--iterations",
        2,
    )
    cli(
        tmp_path,
        "evaluate",
        "--model",
        model_path,
        "--manifest",
        corpus,
        "--output",
        tmp_path / "scores.json",
    )
    scores = json.loads((tmp_path / "scores.json").read_text())
    assert len(scores["results"]) == 9
    assert all(0 <= row["foscttm"] <= 1 for row in scores["results"])
    manifest = read_manifest(corpus)
    attached = tmp_path / "attached.pt"
    cli(
        tmp_path,
        "attach",
        "--model",
        model_path,
        "--modality",
        "vision",
        "--name",
        "new",
        "--features",
        manifest["vision"]["vision_0"]["path"],
        "--output",
        attached,
    )
    assert "new" in ConsensusAlign.load(attached).state["vision"]["factors"]
    cli(tmp_path, "adapt", "--model", model_path, "--manifest", corpus, "--output", tmp_path / "adapted.pt")
    assert torch.equal(
        ConsensusAlign.load(model_path).state["bridge"],
        ConsensusAlign.load(tmp_path / "adapted.pt").state["bridge"],
    )
    cli(
        tmp_path,
        "export",
        "--model",
        model_path,
        "--vision",
        "vision_0",
        "--text",
        "text_0",
        "--output",
        tmp_path / "native.pt",
    )
    compact = torch.load(tmp_path / "native.pt", weights_only=True)
    assert set(compact) == {"mean_x", "mean_y", "weight"}
    assert compact["weight"].shape == (6, 7)
    cli(
        tmp_path,
        "references",
        "--manifest",
        corpus,
        "--output",
        tmp_path / "refs.pt",
        "--clusters",
        4,
        "--restarts",
        1,
        "--workers",
        1,
    )
    cli(
        tmp_path,
        "diagnose",
        "--model",
        model_path,
        "--manifest",
        corpus,
        "--references",
        tmp_path / "refs.pt",
        "--batch-size",
        12,
        "--output",
        tmp_path / "agreement.json",
    )
    agreement = json.loads((tmp_path / "agreement.json").read_text())
    assert len(agreement) == 9 and all(row["reference_count"] == 4 for row in agreement)


@pytest.mark.integration
def test_cli_classification_and_caption_retrieval(corpus, fitted, tmp_path):
    model, _ = fitted
    model_path = tmp_path / "model.pt"
    model.save(model_path)
    manifest = json.loads(corpus.read_text())
    manifest["vision"] = {"vision_0": manifest["vision"]["vision_0"]}
    texts = load_features(read_manifest(corpus)["text"]["text_0"])
    torch.save(texts[:5], tmp_path / "prompts.pt")
    torch.save(torch.arange(49) % 5, tmp_path / "labels.pt")
    manifest.update(task="classification", labels="labels.pt", text={"text_0": "prompts.pt"})
    path = tmp_path / "classification.json"
    path.write_text(json.dumps(manifest))
    cli(
        tmp_path,
        "evaluate",
        "--model",
        model_path,
        "--manifest",
        path,
        "--output",
        tmp_path / "classification_scores.json",
    )
    values = json.loads((tmp_path / "classification_scores.json").read_text())["results"][0]
    assert 0 <= values["top1"] <= values["top5"] <= 1
    captions = texts[:, None, :].repeat(1, 2, 1)
    captions[0, 1] = torch.nan
    torch.save(captions, tmp_path / "captions.pt")
    manifest.update(task="retrieval", text={"text_0": "captions.pt"})
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(manifest))
    cli(
        tmp_path,
        "evaluate",
        "--model",
        model_path,
        "--manifest",
        path,
        "--output",
        tmp_path / "retrieval_scores.json",
    )
    values = json.loads((tmp_path / "retrieval_scores.json").read_text())["results"][0]
    assert 0 <= values["i2t_R@1"] <= values["i2t_R@5"] <= values["i2t_R@10"] <= 1


@pytest.mark.integration
def test_diagnostic_rejects_wrong_seed(corpus, fitted, tmp_path):
    model, _ = fitted
    model.save(tmp_path / "model.pt")
    torch.save({"seed": seed_at(1), "references": {}}, tmp_path / "wrong.pt")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "consensusalign",
            "diagnose",
            "--model",
            str(tmp_path / "model.pt"),
            "--manifest",
            str(corpus),
            "--references",
            str(tmp_path / "wrong.pt"),
            "--output",
            str(tmp_path / "never.json"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0 and "same seed" in result.stderr
    assert not (tmp_path / "never.json").exists()


@pytest.mark.parametrize("module", ["consensusalign", "consensusalign.datasets", "consensusalign.reproduce"])
def test_cli_help_outside_checkout(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=tmp_path, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0 and "usage:" in result.stdout


def test_download_dry_run_requires_no_hub_access(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "consensusalign.datasets",
            "download",
            "--manifest",
            str(root / "configs/coco_train.json"),
            "--revision",
            "test-offline",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["files"]) == 10
