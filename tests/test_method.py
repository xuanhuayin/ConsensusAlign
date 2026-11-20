import copy

import pytest
import torch
from torch.nn.functional import normalize

from consensusalign import ConsensusAlign
from consensusalign._wp.linalg import center_and_normalize, polar
from consensusalign.core import linear_cka, make_pool, member_weights, project
from consensusalign.data import fitting_data, read_manifest, seed_at
from consensusalign.diagnostics import assignment_agreement, fit_references


def test_pool_projection_and_native_factor_objective():
    g = torch.Generator().manual_seed(21)
    features = {
        "a": normalize(torch.randn(17, 8, generator=g), dim=-1),
        "b": normalize(torch.randn(17, 9, generator=g), dim=-1),
    }
    pool = make_pool(features, 4, False)
    centered = {name: center_and_normalize(x, x.mean(0, keepdim=True)) for name, x in features.items()}
    joined = torch.cat([x / 2**0.5 for x in centered.values()], 1)
    expected = (joined - joined.mean(0, keepdim=True)) @ pool["basis"]
    torch.testing.assert_close(pool["latent"], expected, rtol=0, atol=0)
    torch.testing.assert_close(project(features, pool), expected, rtol=0, atol=0)
    torch.testing.assert_close(pool["basis"].T @ pool["basis"], torch.eye(4), atol=1e-6, rtol=1e-6)
    for name in features:
        torch.testing.assert_close(pool["factors"][name], polar(centered[name].T @ expected))


def test_agreement_weights_use_upper_median_and_squared_amplitudes():
    g = torch.Generator().manual_seed(0)
    shared = torch.randn(32, 4, generator=g)
    features = {
        "a": shared,
        "b": shared + 0.1 * torch.randn(32, 4, generator=g),
        "c": torch.randn(32, 4, generator=g),
        "d": torch.randn(32, 4, generator=g),
    }
    agreement = {
        name: sum(linear_cka(x, y) for other, y in features.items() if other != name) / 3
        for name, x in features.items()
    }
    median = sorted(agreement.values())[2]
    expected = {name: min(1, value / median) ** 2 for name, value in agreement.items()}
    assert member_weights(features, True) == pytest.approx(expected)
    assert member_weights(features, False) == dict.fromkeys(features, 1)


def test_one_anchor_weighted_fit_and_save_round_trip(corpus, tmp_path, wp):
    data, rows, anchors = fitting_data(read_manifest(corpus), seed_at(0), 1)
    model = ConsensusAlign.fit(**data, anchors=anchors, latent_dim=4, weighted=True, wp=wp)
    path = tmp_path / "model.pt"
    model.state["rows"] = rows
    model.save(path)
    restored = ConsensusAlign.load(path)
    torch.testing.assert_close(
        model.state["bridge"].T @ model.state["bridge"], torch.eye(4), atol=2e-5, rtol=2e-5
    )
    original = model.interface("vision_0", "text_0")
    loaded = restored.interface("vision_0", "text_0")
    x, y = data["vision"]["vision_0"][:5], data["text"]["text_0"][:7]
    assert torch.equal(original.similarity(x, y), loaded.similarity(x, y))
    assert torch.isfinite(loaded.weight).all()


def test_native_cosine_normalizes_mapped_images(fitted):
    model, data = fitted
    interface = model.interface("vision_0", "text_0")
    x, y = data["vision"]["vision_0"][:5], data["text"]["text_0"][:7]
    expected = normalize(center_and_normalize(x, interface.mean_x) @ interface.weight, dim=-1)
    expected = expected @ center_and_normalize(y, interface.mean_y).T
    torch.testing.assert_close(interface.similarity(x, y), expected, atol=0, rtol=0)


def test_rectangular_attachment_and_frozen_adaptation(fitted):
    model, data = fitted
    source = copy.deepcopy(model.state)
    for modality in ("vision", "text"):
        n = len(next(iter(data[modality].values())))
        new = normalize(torch.randn(n, 2, generator=torch.Generator().manual_seed(5)), dim=-1)
        model.attach(modality, "attached", new)
        assert model.state[modality]["factors"]["attached"].shape == (2, 4)
        data[modality]["attached"] = new
    assert model.interface("attached", "attached").weight.shape == (2, 2)
    adapted = model.adapt(**{mod: dict(reversed(list(values.items()))) for mod, values in data.items()})
    assert "rows" not in adapted.state
    assert torch.equal(adapted.state["bridge"], source["bridge"])
    for modality in ("vision", "text"):
        for field in ("basis", "mean", "latent"):
            assert torch.equal(model.state[modality][field], source[modality][field])
        assert torch.equal(adapted.state[modality]["basis"], source[modality]["basis"])
        assert adapted.state[modality]["order"] == source[modality]["order"]
        assert adapted.state[modality]["weights"] == source[modality]["weights"]
    torch.testing.assert_close(
        adapted.interface("vision_0", "text_0").weight, model.interface("vision_0", "text_0").weight
    )
    with pytest.raises(ValueError, match="all source encoders"):
        model.adapt({"vision_0": data["vision"]["vision_0"]}, data["text"])


def test_independent_reference_diagnostic_is_disjoint_and_deterministic(fitted, wp):
    model, data = fitted
    refs = fit_references(**data, seed=model.state["seed"], wp=wp)
    first = assignment_agreement(model, **data, references=refs, batch_size=12)
    assert first == assignment_agreement(model, **data, references=refs, batch_size=12)
    assert len(refs) == len(first) == 9
    assert all(row["reference_count"] == 4 and row["batch_size"] == 12 for row in first)
    assert all(0 <= row["agreement"] <= 1 for row in first)
    with pytest.raises(ValueError, match="neither endpoint"):
        assignment_agreement(model, **data, references={("vision_0", "text_0"): refs[("vision_0", "text_0")]})


def test_invalid_fit_and_checkpoint_inputs(corpus, wp, tmp_path):
    data, _, anchors = fitting_data(read_manifest(corpus), seed_at(0), 1)
    with pytest.raises(ValueError, match="latent_dim"):
        ConsensusAlign.fit(**data, latent_dim=100, wp=wp)
    anchors[0]["vision_0"] = anchors[0]["vision_0"].repeat(2, 1)
    with pytest.raises(ValueError, match="same item rows"):
        ConsensusAlign.fit(**data, anchors=anchors, latent_dim=4, wp=wp)
    torch.save({"format_version": 99}, tmp_path / "unsupported.pt")
    with pytest.raises(ValueError, match="Unsupported"):
        ConsensusAlign.load(tmp_path / "unsupported.pt")
