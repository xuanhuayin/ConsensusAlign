"""Settings for bit-identical results on every machine.

* k-means runs on one OpenMP thread. With more threads, scikit-learn sums partial results in the order the threads
  finish, so the centers change in the last bits from run to run.
* The fit runs on a fixed number of PyTorch threads. PyTorch splits long sums across threads, so their last bits
  depend on the thread count.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

KMEANS_THREADS = 1
FIT_THREADS = 4


@contextmanager
def torch_threads(
    num_threads: int = FIT_THREADS,
) -> Iterator[None]:
    """Run the enclosed code on ``num_threads`` PyTorch threads."""
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


