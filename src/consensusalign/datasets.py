"""Validate feature manifests and acquire the public base feature files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import load_features, read_manifest, read_tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_manifest(manifest: dict, *, sample_rows: int = 16) -> dict:
    """Check headers and a feature sample. Row identity remains the data owner's contract."""
    import torch

    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    task = manifest.get("task", "alignment")
    if task not in ("alignment", "classification", "retrieval"):
        raise ValueError("task must be alignment, classification or retrieval")
    report = {"layout": manifest.get("layout", "paired-corpus"), "task": task, "modalities": {}}
    min_rows = 2 if task == "alignment" else 1
    for modality in ("vision", "text"):
        records = {}
        for name, spec in manifest[modality].items():
            path = Path(spec["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing feature file for {modality}/{name}: {path}")
            tensor = read_tensor(path)
            if (
                tensor.ndim not in (2, 3)
                or len(tensor) < min_rows
                or not tensor.shape[-1]
                or (tensor.ndim == 3 and not tensor.shape[1])
                or not tensor.is_floating_point()
            ):
                raise ValueError(f"{modality}/{name}: expected floating [N,D] or [N,T,D] features")
            if task == "retrieval" and modality == "text" and tensor.ndim != 3:
                raise ValueError("Retrieval text must have shape [images, captions, dimensions]")
            indices = torch.arange(min(sample_rows, len(tensor)))
            load_features(spec, indices)
            records[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": path.stat().st_size,
            }
        if len({row["shape"][0] for row in records.values()}) != 1:
            raise ValueError(f"All {modality} encoders must share the same item count")
        report["modalities"][modality] = records
    if report["layout"] not in ("paired-corpus", "unpaired"):
        raise ValueError("layout must be paired-corpus or unpaired")
    if (report["layout"] == "paired-corpus" and task == "alignment") or task == "retrieval":
        counts = {row["shape"][0] for values in report["modalities"].values() for row in values.values()}
        if len(counts) != 1:
            raise ValueError("paired-corpus features must share the same corpus item count")
    if task == "classification":
        if "labels" not in manifest:
            raise ValueError("Classification manifests must include a labels file")
        labels = read_tensor(manifest["labels"])
        n_images = next(iter(report["modalities"]["vision"].values()))["shape"][0]
        n_classes = next(iter(report["modalities"]["text"].values()))["shape"][0]
        if (
            labels.shape != (n_images,)
            or labels.is_floating_point()
            or labels.is_complex()
            or labels.dtype == torch.bool
            or labels.min() < 0
            or labels.max() >= n_classes
        ):
            raise ValueError("Classification needs one valid integer class label per image")
        report["labels"] = {"shape": list(labels.shape), "dtype": str(labels.dtype)}
    return report


def fingerprint_manifest(manifest: dict, cache: dict | None = None) -> dict:
    """Content hashes and preprocessing flags without embedding local path names."""
    cache = {} if cache is None else cache
    result = {"layout": manifest.get("layout", "paired-corpus"), "task": manifest.get("task", "alignment")}
    for modality in ("vision", "text"):
        result[modality] = {}
        result[modality + "_order"] = list(manifest[modality])
        for name, spec in manifest[modality].items():
            path = Path(spec["path"])
            if path not in cache:
                cache[path] = sha256_file(path)
            result[modality][name] = {"sha256": cache[path], "prereduce": bool(spec.get("prereduce", False))}
    return result


def download_plan(manifest_paths: list[Path]) -> list[tuple[str, Path]]:
    """Only download public .pt paths already listed in the chosen manifests."""
    entries = {}
    for manifest_path in manifest_paths:
        manifest = read_manifest(manifest_path)
        for modality in ("vision", "text"):
            for spec in manifest[modality].values():
                target = Path(spec["path"])
                positions = [i for i, part in enumerate(target.parts) if part == "embeddings"]
                if not positions:
                    raise ValueError("Public download targets must contain an embeddings/ directory")
                relative = Path(*target.parts[positions[-1] :]).as_posix()
                if target.suffix != ".pt":
                    raise ValueError("The public feature release contains .pt tensors")
                previous = entries.setdefault(relative, target)
                if previous != target:
                    raise ValueError(f"Public file {relative} has conflicting destination directories")
    return sorted(entries.items())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Validate tensor headers and a sample of rows")
    check.add_argument("--manifest", type=Path, required=True)
    download = sub.add_parser("download", help="Download exactly the public files named by manifests")
    download.add_argument("--manifest", type=Path, nargs="+", required=True)
    download.add_argument("--revision", required=True, help="Public dataset commit or tag")
    download.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps(inspect_manifest(read_manifest(args.manifest)), indent=2))
        return
    plan = download_plan(args.manifest)
    if args.dry_run:
        print(json.dumps({"revision": args.revision, "files": [name for name, _ in plan]}, indent=2))
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit("Install data support with: python -m pip install '.[data]'") from error
    for relative, target in plan:
        # local_dir is the directory above embeddings/, independent of the checkout location.
        root = target
        for _ in Path(relative).parts:
            root = root.parent
        actual = Path(
            hf_hub_download(
                "schnaus/unpaired-rosetta-embeddings",
                filename=relative,
                repo_type="dataset",
                revision=args.revision,
                local_dir=root,
            )
        )
        if actual.resolve() != target.resolve():
            raise RuntimeError("Feature download resolved outside the declared target")
        print(f"Available: {relative}")


if __name__ == "__main__":
    main()
