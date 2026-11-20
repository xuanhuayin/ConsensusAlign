"""The final concat-PCA, one-WP-bridge, native-distribution construction."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn.functional import normalize

from ._wp.linalg import center_and_normalize, polar
from ._wp.wasserstein_procrustes import WassersteinProcrustes


@dataclass(frozen=True)
class WPConfig:
    num_clusters: int = 30
    num_restarts: int = 30
    batch_size: int = 10000
    num_iterations: int = 100
    num_workers: int = 4
    num_threads: int = 4

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, int) or value < (0 if key == "num_iterations" else 1):
                raise ValueError(f"Invalid {key}: {value}")


def checked_features(features, min_rows=2):
    """Features are loader-normalized float32 rows, aligned within a modality."""
    if not features:
        raise ValueError("A modality must contain at least one encoder")
    result = {}
    for name, value in features.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Encoder names must be nonempty strings")
        x = value.detach().to(device="cpu", dtype=torch.float32)
        if x.ndim != 2 or x.shape[0] < min_rows or not x.shape[1] or not torch.isfinite(x).all():
            raise ValueError(f"{name}: expected a finite [items, dimensions] tensor")
        result[name] = x
    if len({x.shape[0] for x in result.values()}) != 1:
        raise ValueError("Every encoder in a modality must use the same item rows")
    return result


def linear_cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = (a.T @ a).norm() * (b.T @ b).norm()
    return ((a.T @ b).norm().square() / denominator).item() if denominator > 0 else 0.0


def member_weights(features, weighted):
    names = list(features)
    if not weighted or len(names) < 3:
        return {name: 1.0 for name in names}
    index = torch.randperm(len(features[names[0]]), generator=torch.Generator().manual_seed(0))[:8000]
    shared = {name: features[name][index] for name in names}
    agreement = {
        name: sum(linear_cka(shared[name], shared[other]) for other in names if other != name)
        / (len(names) - 1)
        for name in names
    }
    median = sorted(agreement.values())[len(names) // 2]  # upper median
    if median <= 0:
        raise ValueError("Agreement weighting requires nonzero shared geometry")
    return {name: min(1.0, agreement[name] / median) ** 2 for name in names}


def concatenate(features, order, weights):
    return torch.cat([weights[name] * features[name] / len(order) ** 0.5 for name in order], dim=1)


def make_pool(features, dim, weighted):
    means = {name: x.mean(0, keepdim=True) for name, x in features.items()}
    centered = {name: center_and_normalize(x, means[name]) for name, x in features.items()}
    order = list(centered)
    weights = member_weights(centered, weighted)
    joined = concatenate(centered, order, weights)
    if not 1 <= dim <= min(joined.shape):
        raise ValueError(f"latent_dim={dim} exceeds the available matrix shape {tuple(joined.shape)}")
    mean = joined.mean(0, keepdim=True)
    _, _, right = torch.linalg.svd(joined - mean, full_matrices=False)
    basis = right[:dim].T.contiguous()
    latent = (joined - mean) @ basis
    factors = {name: polar(x.T @ latent) for name, x in centered.items()}
    return dict(
        order=order, weights=weights, means=means, mean=mean, basis=basis, latent=latent, factors=factors
    )


def project(features, pool):
    missing = set(pool["order"]) - set(features)
    if missing:
        raise ValueError(f"Missing source encoders: {sorted(missing)}")
    centered = {name: center_and_normalize(features[name], pool["means"][name]) for name in pool["order"]}
    return (concatenate(centered, pool["order"], pool["weights"]) - pool["mean"]) @ pool["basis"]


class NativeMap:
    """A deployed pair needs only these two fitting means and one matrix."""

    def __init__(self, mean_x, mean_y, weight):
        self.mean_x, self.mean_y, self.weight = mean_x, mean_y, weight

    @torch.inference_mode()
    def similarity(self, images, texts):
        images = center_and_normalize(images.float(), self.mean_x)
        texts = center_and_normalize(texts.float(), self.mean_y)
        return normalize(images @ self.weight, dim=-1) @ texts.T


class ConsensusAlign:
    """All public inputs are normalized embedding rows returned by load_features.

    Each modality shares item order. Fit receives disjoint image and text sets.
    Optional anchors are row-aligned across modalities and use the same encoder keys.
    """

    def __init__(self, state):
        self.state = state

    @classmethod
    @torch.inference_mode()
    def fit(cls, vision, text, *, seed=734796314, latent_dim=768, weighted=False, anchors=None, wp=None):
        vision, text = checked_features(vision), checked_features(text)
        wp = wp or WPConfig()
        if (
            min(len(next(iter(vision.values()))), len(next(iter(text.values()))), wp.batch_size)
            < wp.num_clusters
        ):
            raise ValueError("There must be at least num_clusters fitting items in each WP batch")
        pv, pt = make_pool(vision, latent_dim, weighted), make_pool(text, latent_dim, weighted)
        ax = ay = None
        if anchors is not None:
            av, at = (checked_features(values, min_rows=1) for values in anchors)
            if set(av) != set(vision) or set(at) != set(text):
                raise ValueError("Anchor encoders must match fitting encoders")
            ax, ay = project(av, pv), project(at, pt)
            if len(ax) != len(ay):
                raise ValueError("Both modalities must receive the same physical anchors")
        aligner = WassersteinProcrustes(**asdict(wp))
        aligner.fit(pv["latent"], pt["latent"], ax, ay, randomness=int(seed) + 7)
        return cls(
            dict(
                format_version=1,
                vision=pv,
                text=pt,
                bridge=aligner.weight,
                seed=int(seed),
                weighted=bool(weighted),
                wp=asdict(wp),
            )
        )

    def interface(self, vision, text):
        pv, pt = self.state["vision"], self.state["text"]
        return NativeMap(
            pv["means"][vision],
            pt["means"][text],
            pv["factors"][vision] @ self.state["bridge"] @ pt["factors"][text].T,
        )

    @torch.inference_mode()
    def attach(self, modality, name, features):
        if modality not in ("vision", "text"):
            raise ValueError("modality must be vision or text")
        pool = self.state[modality]
        if name in pool["factors"]:
            raise ValueError(f"Encoder already exists: {name}")
        x = checked_features({name: features})[name]
        if len(x) != len(pool["latent"]):
            raise ValueError("Attachment rows must match the cached source fitting rows, in order")
        mean = x.mean(0, keepdim=True)
        factor = polar(center_and_normalize(x, mean).T @ pool["latent"])
        pool["means"][name], pool["factors"][name] = mean, factor
        return self

    @torch.inference_mode()
    def adapt(self, vision, text):
        """Reuse source PCA, weights, concat offsets and C. Refit target native factors."""
        state = dict(self.state)
        state.pop("rows", None)
        for modality, values in [("vision", vision), ("text", text)]:
            values = checked_features(values)
            source = self.state[modality]
            missing = set(source["order"]) - set(values)
            if missing:
                raise ValueError(f"Adaptation requires all source encoders: {sorted(missing)}")
            means = {name: x.mean(0, keepdim=True) for name, x in values.items()}
            centered = {name: center_and_normalize(x, means[name]) for name, x in values.items()}
            joined = concatenate(centered, source["order"], source["weights"])
            latent = (joined - source["mean"]) @ source["basis"]
            factors = {name: polar(x.T @ latent) for name, x in centered.items()}
            state[modality] = dict(source, means=means, factors=factors, latent=latent)
        return ConsensusAlign(state)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        return cls(state)
