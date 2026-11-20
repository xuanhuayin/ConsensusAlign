"""Small synthetic exercise of the final method. This is not a paper experiment."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import normalize

from consensusalign import ConsensusAlign, WPConfig
from consensusalign.data import fitting_data, read_manifest
from consensusalign.diagnostics import assignment_agreement, fit_references
from consensusalign.evaluation import alignment_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("smoke_output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(123)
    latent = torch.randn(96, 6, generator=generator) * torch.arange(1, 7)
    manifest = {"layout": "paired-corpus", "vision": {}, "text": {}}
    for modality, widths in [("vision", [8, 10, 12]), ("text", [8, 11, 9])]:
        for i, width in enumerate(widths):
            name = f"{modality}_{i}"
            projection = torch.randn(6, width, generator=generator)
            values = normalize(latent @ projection, dim=-1)
            filename = f"{name}.pt"
            torch.save(values, args.output / filename)
            manifest[modality][name] = filename
    (args.output / "features.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest = read_manifest(args.output / "features.json")
    seed = 734796314
    data, rows, anchors = fitting_data(manifest, seed, num_pairs=1)
    config = WPConfig(num_clusters=6, num_restarts=2, batch_size=40, num_iterations=3, num_workers=1)
    model = ConsensusAlign.fit(data["vision"], data["text"], latent_dim=6, seed=seed, wp=config)
    model.state["rows"] = rows
    model.save(args.output / "model.pt")
    restored = ConsensusAlign.load(args.output / "model.pt")
    original = model.interface("vision_0", "text_0")
    loaded = restored.interface("vision_0", "text_0")
    assert torch.equal(original.weight, loaded.weight)
    assert torch.isfinite(model.state["bridge"]).all()
    torch.testing.assert_close(
        model.state["bridge"].T @ model.state["bridge"], torch.eye(6), atol=1e-5, rtol=1e-5
    )
    anchored = ConsensusAlign.fit(
        data["vision"], data["text"], latent_dim=6, seed=seed, anchors=anchors, weighted=True, wp=config
    )
    assert torch.isfinite(anchored.state["bridge"]).all()
    extra = normalize(torch.randn(len(rows["vision"]), 4, generator=generator), dim=-1)
    restored.attach("vision", "new_vision", extra)
    assert restored.interface("new_vision", "text_0").weight.shape == (4, 8)
    adapted = model.adapt(data["vision"], data["text"])
    assert torch.equal(adapted.state["bridge"], model.state["bridge"])
    refs = fit_references(data["vision"], data["text"], seed, config)
    agreement = assignment_agreement(model, data["vision"], data["text"], refs, batch_size=24)
    assert len(agreement) == 9 and all(row["reference_count"] == 4 for row in agreement)
    metrics = alignment_metrics(loaded, data["vision"]["vision_0"][:24], data["text"]["text_0"][:24], seed, 8)
    assert 0 <= metrics["foscttm"] <= 1
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "data": "synthetic",
                "reference_count_per_target": 4,
                "checks": [
                    "unpaired fitting",
                    "one-anchor weighted fitting",
                    "save/load",
                    "rectangular attachment",
                    "frozen-bridge adaptation",
                    "independent-reference agreement",
                    "alignment metric",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
