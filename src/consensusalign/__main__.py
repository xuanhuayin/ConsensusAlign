"""Command-line entry points for the final method and its evaluation."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import normalize

from .core import ConsensusAlign, WPConfig
from .data import fitting_data, load_features, read_manifest, read_tensor, seed_at
from .diagnostics import assignment_agreement, fit_references
from .evaluation import alignment_metrics, classification_metrics, retrieval_metrics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parser():
    root = argparse.ArgumentParser(description="ConsensusAlign: one pooled bridge and native encoder maps")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ["fit", "adapt", "evaluate", "references", "diagnose"]:
        p = commands.add_parser(name)
        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        if name not in ("fit", "references"):
            p.add_argument("--model", required=True, type=Path)
        if name in ("fit", "references"):
            p.add_argument("--seed-index", type=int, default=0, choices=range(5))
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--clusters", type=int, default=30)
            p.add_argument("--restarts", type=int, default=30)
        if name == "fit":
            p.add_argument("--latent-dim", type=int, default=768)
            p.add_argument("--weighting", choices=["uniform", "agreement"], default="uniform")
            p.add_argument("--anchors", type=int, default=0)
            p.add_argument("--batch-size", type=int, default=10000)
            p.add_argument("--iterations", type=int, default=100)
        if name == "diagnose":
            p.add_argument("--references", required=True, type=Path)
            p.add_argument("--batch-size", type=int, default=5000)
        if name == "evaluate":
            p.add_argument("--block-size", type=int, default=256)
    p = commands.add_parser("attach")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--modality", required=True, choices=["vision", "text"])
    p.add_argument("--name", required=True)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--prereduce", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    p = commands.add_parser("export")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--vision", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", required=True, type=Path)
    return root


def evaluate(model, manifest, block_size):
    if block_size < 1:
        raise ValueError("block-size must be positive")
    task = manifest.get("task", "alignment")
    if task not in ("alignment", "classification", "retrieval"):
        raise ValueError("Evaluation task must be alignment, classification or retrieval")
    if task == "classification" and "labels" not in manifest:
        raise ValueError("Classification manifests must include a labels file")
    labels = read_tensor(manifest["labels"]) if task == "classification" else None
    images = {v: load_features(spec) for v, spec in manifest["vision"].items()}
    result = []
    for t, spec in manifest["text"].items():
        if task == "retrieval":
            raw = read_tensor(spec["path"]).float()
            if raw.ndim != 3 or raw.shape[1] < 1:
                raise ValueError("Retrieval text must have shape [images, captions, dimensions]")
            owners = torch.arange(len(raw)).repeat_interleave(raw.shape[1])
            raw = raw.flatten(0, 1)
            valid = torch.isfinite(raw).all(1)
            captions, owners = normalize(raw[valid], dim=-1), owners[valid]
        else:
            captions = load_features(spec)
        for v, values in images.items():
            interface = model.interface(v, t)
            if task == "alignment":
                metrics = alignment_metrics(interface, values, captions, model.state["seed"], block_size)
            elif task == "classification":
                metrics = classification_metrics(interface, values, captions, labels, block_size)
            else:
                if len(values) != len(read_tensor(spec["path"])):
                    raise ValueError("Retrieval needs the same image count in image and caption files")
                metrics = retrieval_metrics(interface, values, captions, owners, block_size)
            result.append(dict(vision=v, text=t, **metrics))
    return dict(task=task, results=result)


def main(argv=None):
    args = parser().parse_args(argv)
    # The public numerical solver fits on four torch threads. Keep pooling and
    # native factors on the same thread count across command-line runs as well.
    torch.set_num_threads(4)
    if args.command == "attach":
        model = ConsensusAlign.load(args.model)
        rows = model.state.get("rows", {}).get(args.modality)
        features = load_features({"path": str(args.features), "prereduce": args.prereduce}, rows)
        model.attach(args.modality, args.name, features).save(args.output)
    elif args.command == "export":
        interface = ConsensusAlign.load(args.model).interface(args.vision, args.text)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(mean_x=interface.mean_x, mean_y=interface.mean_y, weight=interface.weight), args.output
        )
    else:
        manifest = read_manifest(args.manifest)
        if args.command in ("fit", "references"):
            seed = seed_at(args.seed_index)
            data, rows, anchors = fitting_data(manifest, seed, args.anchors if args.command == "fit" else 0)
            config = WPConfig(
                num_workers=args.workers,
                num_clusters=args.clusters,
                num_restarts=args.restarts,
                batch_size=args.batch_size if args.command == "fit" else 5000,
                num_iterations=args.iterations if args.command == "fit" else 30,
            )
            if args.command == "fit":
                model = ConsensusAlign.fit(
                    data["vision"],
                    data["text"],
                    seed=seed,
                    latent_dim=args.latent_dim,
                    weighted=args.weighting == "agreement",
                    anchors=anchors,
                    wp=config,
                )
                model.state["rows"] = rows
                model.save(args.output)
            else:
                refs = fit_references(data["vision"], data["text"], seed, config)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(seed=seed, references=refs), args.output)
        else:
            model = ConsensusAlign.load(args.model)
            if args.command == "evaluate":
                write_json(args.output, evaluate(model, manifest, args.block_size))
            else:
                data, rows, _ = fitting_data(manifest, model.state["seed"])
                if args.command == "adapt":
                    model = model.adapt(data["vision"], data["text"])
                    model.state["rows"] = rows
                    model.save(args.output)
                else:
                    refs = torch.load(args.references, map_location="cpu", weights_only=True)
                    if refs["seed"] != model.state["seed"]:
                        raise ValueError("Reference maps and candidate model must use the same seed")
                    write_json(
                        args.output,
                        assignment_agreement(
                            model, data["vision"], data["text"], refs["references"], args.batch_size
                        ),
                    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
