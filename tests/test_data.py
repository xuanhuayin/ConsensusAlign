import json

import numpy as np
import pytest
import torch
from torch.nn.functional import normalize

from consensusalign.data import fitting_data, load_features, read_manifest, read_tensor, seed_at
from consensusalign.datasets import download_plan, fingerprint_manifest, inspect_manifest


@pytest.mark.parametrize("extension", ["pt", "legacy.pt", "npy"])
def test_tensor_formats_and_index_order(tmp_path, extension):
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    path = tmp_path / f"features.{extension}"
    if extension == "npy":
        np.save(path, values.numpy())
    else:
        torch.save(values, path, _use_new_zipfile_serialization=extension != "legacy.pt")
    torch.testing.assert_close(read_tensor(path), values, rtol=0, atol=0)
    indices = torch.tensor([5, 1, 3, 0])
    torch.testing.assert_close(load_features(path, indices), normalize(values[indices], dim=-1))


@pytest.mark.parametrize("prereduce", [False, True])
def test_stored_precision_caption_reduction(tmp_path, prereduce):
    values = torch.randn(6, 3, 5, generator=torch.Generator().manual_seed(1)).bfloat16()
    values[0, 1] = torch.nan
    path = tmp_path / "captions.pt"
    torch.save(values, path)
    expected = normalize(normalize(values, dim=-1).nanmean(1), dim=-1)
    if prereduce:
        expected = normalize(expected, dim=-1)
    actual = load_features({"path": path, "prereduce": prereduce})
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)


def test_invalid_caption_item_is_rejected(tmp_path):
    path = tmp_path / "captions.pt"
    torch.save(torch.full((2, 3, 5), torch.nan), path)
    with pytest.raises(ValueError, match="finite"):
        load_features(path)


@pytest.mark.parametrize("index", range(5))
def test_disjoint_halves_and_nested_anchors(corpus, index):
    manifest = read_manifest(corpus)
    seed = seed_at(index)
    data, rows, anchors = fitting_data(manifest, seed, 5)
    assert len(rows["vision"]) == 24 and len(rows["text"]) == 25
    assert not set(rows["vision"].tolist()) & set(rows["text"].tolist())
    assert set(rows["vision"].tolist()) | set(rows["text"].tolist()) == set(range(49))
    _, repeated, one = fitting_data(manifest, seed, 1)
    for modality, paired in zip(("vision", "text"), anchors, strict=True):
        assert torch.equal(rows[modality], repeated[modality])
        for name, spec in manifest[modality].items():
            torch.testing.assert_close(data[modality][name], load_features(spec, rows[modality]))
            torch.testing.assert_close(paired[name][:1], one[0 if modality == "vision" else 1][name])


def test_unpaired_layout_does_not_resplit(corpus):
    manifest = read_manifest(corpus)
    manifest["layout"] = "unpaired"
    data, rows, anchors = fitting_data(manifest, seed_at(0))
    assert rows == {"vision": None, "text": None} and anchors is None
    assert len(data["vision"]["vision_0"]) == 49
    with pytest.raises(ValueError, match="Anchors"):
        fitting_data(manifest, seed_at(0), 1)


@pytest.mark.parametrize("index", [-1, 5])
def test_invalid_seed(index):
    with pytest.raises(ValueError, match="seed-index"):
        seed_at(index)


def test_manifest_checks_and_path_free_content_identity(corpus):
    manifest = read_manifest(corpus)
    assert inspect_manifest(manifest)["modalities"]["vision"]["vision_0"]["shape"] == [49, 6]
    before = fingerprint_manifest(manifest)
    assert str(corpus.parent) not in json.dumps(before)
    manifest["vision"] = dict(reversed(list(manifest["vision"].items())))
    reordered = fingerprint_manifest(manifest)
    assert reordered != before
    spec = manifest["vision"]["vision_0"]
    torch.save(read_tensor(spec["path"]) + 0.1, spec["path"])
    assert fingerprint_manifest(manifest) != reordered


def test_inspection_rejects_mismatched_item_counts(corpus):
    manifest = read_manifest(corpus)
    torch.save(torch.randn(17, 6), manifest["vision"]["vision_0"]["path"])
    with pytest.raises(ValueError, match="same item count"):
        inspect_manifest(manifest)


def test_public_download_plan_is_exact_and_offline():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    files = download_plan([root / "configs/coco_train.json", root / "configs/coco_val.json"])
    assert len(files) == 20
    assert len({name for name, _ in files}) == 20
    assert all(name.startswith("embeddings/") and name.endswith(".pt") for name, _ in files)


@pytest.mark.parametrize("shape", [(3, 0), (3, 0, 4)])
def test_inspection_rejects_empty_feature_axes(corpus, shape):
    manifest = read_manifest(corpus)
    torch.save(torch.empty(shape), manifest["vision"]["vision_0"]["path"])
    with pytest.raises(ValueError, match="expected floating"):
        inspect_manifest(manifest)


def test_classification_inspection_uses_distinct_image_and_class_counts(corpus):
    manifest = read_manifest(corpus)
    manifest["task"] = "classification"
    for spec in manifest["text"].values():
        torch.save(torch.randn(5, 18, 6), spec["path"])
    labels = corpus.parent / "labels.pt"
    torch.save(torch.arange(49) % 5, labels)
    manifest["labels"] = str(labels)
    assert inspect_manifest(manifest)["labels"]["shape"] == [49]
    torch.save(torch.full((49,), 5, dtype=torch.long), labels)
    with pytest.raises(ValueError, match="integer class label"):
        inspect_manifest(manifest)


def test_retrieval_inspection_requires_caption_axis(corpus):
    manifest = read_manifest(corpus)
    manifest["task"] = "retrieval"
    with pytest.raises(ValueError, match="Retrieval text"):
        inspect_manifest(manifest)
