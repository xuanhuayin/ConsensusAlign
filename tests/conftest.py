"""Small synthetic fixtures; no benchmark data or network access is required."""

import json

import pytest
import torch
from torch.nn.functional import normalize

from consensusalign import ConsensusAlign, WPConfig
from consensusalign.data import fitting_data, read_manifest, seed_at


@pytest.fixture(scope="session", autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def corpus(tmp_path):
    generator = torch.Generator().manual_seed(123)
    shared = torch.randn(49, 4, generator=generator) * torch.arange(1, 5)
    manifest = {"layout": "paired-corpus", "vision": {}, "text": {}}
    for modality, widths in [("vision", [6, 8, 7]), ("text", [7, 6, 9])]:
        for index, width in enumerate(widths):
            name = f"{modality}_{index}"
            tensor = normalize(shared @ torch.randn(4, width, generator=generator), dim=-1)
            torch.save(tensor, tmp_path / f"{name}.pt")
            manifest[modality][name] = f"{name}.pt"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def wp():
    return WPConfig(num_clusters=4, num_restarts=1, batch_size=20, num_iterations=2, num_workers=1)


@pytest.fixture
def fitted(corpus, wp):
    data, rows, _ = fitting_data(read_manifest(corpus), seed_at(0))
    model = ConsensusAlign.fit(**data, seed=seed_at(0), latent_dim=4, wp=wp)
    model.state["rows"] = rows
    return model, data
