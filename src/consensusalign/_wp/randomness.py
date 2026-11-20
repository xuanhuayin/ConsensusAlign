"""Random generators of a run, all seeded with the run seed.

``torch`` draws permutations and QAP solver seeds, ``numpy`` draws k-means seeds. Both are seeded with the run seed,
which reproduces the paper's numbers.
"""


import numpy as np
import torch
from torch import Tensor


class Randomness:
    """Seeded torch and numpy generators of one run."""

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = int(seed)
        self.torch = torch.Generator().manual_seed(self.seed)
        self.numpy = np.random.RandomState(self.seed)

    def permutation(
        self,
        num_samples: int,
    ) -> Tensor:
        return torch.randperm(num_samples, generator=self.torch)

    def subset(
        self,
        num_samples: int,
        size: int | None,
    ) -> Tensor:
        """A random subset of ``range(num_samples)`` in random order (all indices if ``size`` is None)."""
        return self.permutation(num_samples)[:size]

    def solver_seed(
        self,
    ) -> int:
        return int(torch.randint(0, 2**32 - 1, (1,), generator=self.torch).item())

    def kmeans_seed(
        self,
    ) -> int:
        return int(self.numpy.randint(0, 2**31 - 1))



def derive_seeds(
    seed: int = 42,
    num_seeds: int = 5,
) -> list[int]:
    """The seeds of all experiments.

    ``derive_seeds(42, 5) == [734796314, 576165995, 2197670066, 839703249, 2584932063]``.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2**32 - 1, (num_seeds,), generator=generator).tolist()


SEEDS = derive_seeds(42, 5)
