"""Relative-path manifests and stored-precision embedding preprocessing."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from ._wp.randomness import SEEDS, Randomness


def read_tensor(path):
    path = Path(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path, allow_pickle=False).copy())
    if path.suffix != ".pt":
        raise ValueError("Embedding files must be .pt tensors or .npy arrays")
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError as error:
        # Older torch.save tensors are valid .pt inputs but cannot be memory-mapped.
        if "mmap can only be used with files saved with" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def reduce_embeddings(x):
    # Keep stored precision through normalization and caption averaging.
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim == 3:
        x = normalize(x, dim=-1).nanmean(dim=1)
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("Expected floating [items, dim] or [items, captions, dim] embeddings")
    x = normalize(x, dim=-1)
    if not torch.isfinite(x).all():
        raise ValueError("Every item must have at least one finite embedding")
    return x


def load_features(spec, indices=None):
    """Use prereduce=true only for the generative-text alias convention in COCO."""
    path = spec["path"] if isinstance(spec, dict) else spec
    x = read_tensor(path)
    if indices is not None:
        # Match the stored loader's file-order reads and restore requested order.
        order = torch.argsort(indices)
        x = x[indices[order]][torch.argsort(order)]
    if isinstance(spec, dict) and spec.get("prereduce", False):
        x = reduce_embeddings(x).to(x.dtype)
    return reduce_embeddings(x).float()


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    for modality in ("vision", "text"):
        if not manifest.get(modality):
            raise ValueError(f"Manifest needs a nonempty {modality} mapping")
        for name, spec in manifest[modality].items():
            spec = dict(spec) if isinstance(spec, dict) else {"path": spec}
            spec["path"] = str((path.parent / spec["path"]).resolve())
            manifest[modality][name] = spec
    if "labels" in manifest:
        manifest["labels"] = str((path.parent / manifest["labels"]).resolve())
    return manifest


def fitting_data(manifest, seed, num_pairs=0):
    """paired-corpus uses one permutation before drawing the optional anchor subset."""
    layout = manifest.get("layout", "paired-corpus")
    if layout not in ("paired-corpus", "unpaired"):
        raise ValueError("layout must be paired-corpus or unpaired")
    if num_pairs < 0 or (num_pairs and layout != "paired-corpus"):
        raise ValueError("Anchors require paired-corpus input and a nonnegative budget")
    rows = {"vision": None, "text": None}
    pair_rows = None
    if layout == "paired-corpus":
        sizes = {
            read_tensor(spec["path"]).shape[0]
            for mod in ("vision", "text")
            for spec in manifest[mod].values()
        }
        if len(sizes) != 1:
            raise ValueError("paired-corpus files must share one item order and item count")
        n = sizes.pop()
        if num_pairs > n:
            raise ValueError("Anchor budget exceeds corpus size")
        rnd = Randomness(seed)
        order = rnd.permutation(n)
        rows = {"vision": order[: n // 2], "text": order[n // 2 :]}
        pair_rows = rnd.subset(n, num_pairs) if num_pairs else None
    data = {
        mod: {name: load_features(spec, rows[mod]) for name, spec in manifest[mod].items()}
        for mod in ("vision", "text")
    }
    anchors = (
        None
        if pair_rows is None
        else tuple(
            {name: load_features(spec, pair_rows) for name, spec in manifest[mod].items()}
            for mod in ("vision", "text")
        )
    )
    return data, rows, anchors


def seed_at(index):
    if not 0 <= index < len(SEEDS):
        raise ValueError("seed-index must lie between 0 and 4")
    return SEEDS[index]
