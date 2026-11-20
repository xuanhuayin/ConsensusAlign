"""Alignment, zero-shot classification, and bidirectional retrieval metrics."""

import torch

from ._wp.randomness import Randomness


def check_inputs(images, texts, block_size):
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if images.ndim != 2 or texts.ndim != 2 or min(len(images), len(texts)) == 0:
        raise ValueError("Evaluation requires nonempty two-dimensional feature matrices")


@torch.inference_mode()
def alignment_metrics(interface, images, texts, seed=0, block_size=256):
    """Row i is the evaluation partner of row i. Ties receive their midrank.

    The separate shuffles and draw order follow the released evaluation protocol.
    """
    check_inputs(images, texts, block_size)
    n = len(images)
    if n != len(texts) or n < 2:
        raise ValueError("Alignment evaluation requires at least two row-aligned pairs")
    rnd = Randomness(seed)
    ix, iy = rnd.permutation(n), rnd.permutation(n)
    x, y = images[ix], texts[iy]
    partner = torch.argsort(iy)[ix]
    ranks = torch.empty(n, dtype=torch.float64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        scores = interface.similarity(x[start:stop], y).double()
        truth = scores[torch.arange(stop - start), partner[start:stop]]
        higher = (scores > truth[:, None]).sum(1).double()
        tied_others = (scores == truth[:, None]).sum(1).double() - 1
        ranks[ix[start:stop]] = higher + 0.5 * tied_others + 1
    return {"foscttm": ((ranks - 1) / (n - 1)).mean().item(), "mean_rank": ranks.mean().item()}


@torch.inference_mode()
def classification_metrics(interface, images, class_prompts, labels, block_size=256):
    check_inputs(images, class_prompts, block_size)
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("Provide one integer class label per image")
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("Class labels must use an integer dtype")
    labels = labels.long()
    if labels.min() < 0 or labels.max() >= len(class_prompts):
        raise ValueError("Labels must index the supplied class-prompt rows")
    correct1 = correct5 = 0
    for start in range(0, len(images), block_size):
        scores = interface.similarity(images[start : start + block_size], class_prompts)
        ranked = scores.topk(min(5, len(class_prompts)), 1).indices
        truth = labels[start : start + block_size]
        # AVG10 uses argmax, whose first-class tie rule differs from topk.
        correct1 += (scores.argmax(1) == truth).sum().item()
        correct5 += (ranked == truth[:, None]).any(1).sum().item()
    return {"top1": correct1 / len(images), "top5": correct5 / len(images)}


@torch.inference_mode()
def retrieval_metrics(interface, images, captions, owners, block_size=256):
    """owners[j] is the image index of caption j. Compute both query directions."""
    check_inputs(images, captions, block_size)
    if owners.is_floating_point() or owners.is_complex() or owners.dtype == torch.bool:
        raise ValueError("Caption owners must use an integer dtype")
    owners = owners.long()
    if owners.shape != (len(captions),) or owners.min() < 0 or owners.max() >= len(images):
        raise ValueError("Each caption must identify an image row")
    if (torch.bincount(owners, minlength=len(images)) == 0).any():
        raise ValueError("Every evaluated image must have at least one valid caption")
    result = {}
    for text_query in (False, True):
        n = len(captions) if text_query else len(images)
        correct = {k: 0 for k in (1, 5, 10)}
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            score = (
                interface.similarity(images, captions[start:stop]).T
                if text_query
                else interface.similarity(images[start:stop], captions)
            )
            # Keep the original retrieval script's argsort tie convention.
            ranked = score.argsort(1, descending=True)[:, :10]
            matches = (
                ranked == owners[start:stop, None]
                if text_query
                else owners[ranked] == torch.arange(start, stop)[:, None]
            )
            for k in correct:
                correct[k] += matches[:, :k].any(1).sum().item()
        prefix = "t2i" if text_query else "i2t"
        result.update({f"{prefix}_R@{k}": count / n for k, count in correct.items()})
    return result
