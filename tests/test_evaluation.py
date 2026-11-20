import pytest
import torch

from consensusalign.core import NativeMap
from consensusalign.evaluation import alignment_metrics, classification_metrics, retrieval_metrics


class DotProduct:
    def similarity(self, x, y):
        return x @ y.T


def test_alignment_perfect_pairs_and_midrank_ties():
    x = torch.eye(12)
    interface = NativeMap(torch.zeros(1, 12), torch.zeros(1, 12), torch.eye(12))
    assert alignment_metrics(interface, x, x, 42, 5) == {"foscttm": 0, "mean_rank": 1}
    zero = torch.zeros_like(x)
    assert alignment_metrics(interface, zero, zero, 42, 5) == {"foscttm": 0.5, "mean_rank": 6.5}


def test_classification_uses_argmax_for_top1_ties():
    result = classification_metrics(
        DotProduct(), torch.zeros(3, 8), torch.eye(8), torch.zeros(3, dtype=torch.long)
    )
    assert result["top1"] == 1


@pytest.mark.parametrize("seed", range(10))
def test_retrieval_matches_dense_argsort_in_both_directions(seed):
    g = torch.Generator().manual_seed(seed)
    images = torch.randint(-1, 2, (13, 6), generator=g).float()
    captions = torch.randint(-1, 2, (26, 6), generator=g).float()
    owners = torch.arange(13).repeat_interleave(2)
    scores = images @ captions.T
    i_rank = scores.argsort(1, descending=True)
    t_rank = scores.T.argsort(1, descending=True)
    actual = retrieval_metrics(DotProduct(), images, captions, owners, block_size=4)
    for k in (1, 5, 10):
        i_expected = (owners[i_rank[:, :k]] == torch.arange(13)[:, None]).any(1).double().mean().item()
        t_expected = (t_rank[:, :k] == owners[:, None]).any(1).double().mean().item()
        assert actual[f"i2t_R@{k}"] == pytest.approx(i_expected)
        assert actual[f"t2i_R@{k}"] == pytest.approx(t_expected)


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([0.0, 1.0]), torch.tensor([True, False]), torch.tensor([-1, 0]), torch.tensor([0, 8])],
)
def test_invalid_class_labels(labels):
    with pytest.raises(ValueError, match="label|Labels"):
        classification_metrics(DotProduct(), torch.eye(2), torch.eye(2), labels)


@pytest.mark.parametrize("owners", [torch.tensor([0.0, 1.0]), torch.tensor([0, 0]), torch.tensor([0, 2])])
def test_invalid_caption_owners(owners):
    with pytest.raises(ValueError, match="Caption|caption"):
        retrieval_metrics(DotProduct(), torch.eye(2), torch.eye(2), owners)


def test_empty_and_invalid_block_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        alignment_metrics(DotProduct(), torch.empty(0, 2), torch.eye(2))
    with pytest.raises(ValueError, match="block_size"):
        alignment_metrics(DotProduct(), torch.eye(2), torch.eye(2), block_size=0)
