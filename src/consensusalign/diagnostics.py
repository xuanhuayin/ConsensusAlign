"""Optional configuration-disjoint agreement using independent WP references."""

from dataclasses import asdict, replace

import torch

from ._wp.assignment import hungarian_matching
from ._wp.linalg import center_and_normalize
from ._wp.wasserstein_procrustes import WassersteinProcrustes
from .core import WPConfig, checked_features


@torch.inference_mode()
def fit_references(vision, text, seed, wp=None):
    vision, text = checked_features(vision), checked_features(text)
    config = replace(wp or WPConfig(), batch_size=5000, num_iterations=30)
    if (
        min(len(next(iter(vision.values()))), len(next(iter(text.values()))), config.batch_size)
        < config.num_clusters
    ):
        raise ValueError("Reference fitting has fewer rows than num_clusters")
    refs = {}
    for v, x in vision.items():
        for t, y in text.items():
            fit = WassersteinProcrustes(**asdict(config))
            fit.fit(x, y, randomness=int(seed) + 7)
            refs[(v, t)] = fit.weight
    return refs


@torch.inference_mode()
def assignment_agreement(model, vision, text, references, batch_size=5000):
    """No threshold is inferred. Agreement is a corpus-dependent diagnostic score."""
    if batch_size < 2:
        raise ValueError("Diagnostic batch size must be at least two")
    pv, pt = model.state["vision"], model.state["text"]
    vision, text = checked_features(vision), checked_features(text)
    g = torch.Generator().manual_seed(model.state["seed"] + 5)
    b = min(batch_size, len(next(iter(vision.values()))), len(next(iter(text.values()))))
    iv = torch.randperm(len(next(iter(vision.values()))), generator=g)[:b]
    it = torch.randperm(len(next(iter(text.values()))), generator=g)[:b]
    x = {v: center_and_normalize(value, pv["means"][v])[iv] for v, value in vision.items()}
    y = {t: center_and_normalize(value, pt["means"][t])[it] for t, value in text.items()}
    result = []
    for v in vision:
        for t in text:
            other = [(a, b_) for a, b_ in references if a != v and b_ != t]
            if not other:
                raise ValueError("Diagnostic needs references sharing neither endpoint configuration")
            scores = torch.zeros((b, b), dtype=torch.float32)
            for a, b_ in other:
                scores += x[a] @ references[(a, b_)] @ y[b_].T
            rows, cols = hungarian_matching(scores / len(other))
            ref = cols[torch.argsort(rows)]
            rows, cols = hungarian_matching(x[v] @ model.interface(v, t).weight @ y[t].T)
            candidate = cols[torch.argsort(rows)]
            result.append(
                dict(
                    vision=v,
                    text=t,
                    batch_size=b,
                    reference_count=len(other),
                    agreement=(candidate == ref).float().mean().item(),
                )
            )
    return result
