# Third-party attribution

The numerical WP solver in `src/consensusalign/_wp/` is derived from the MIT-licensed
public implementation at https://github.com/dominik-schnaus/unpaired-rosetta.
It includes the geometric initialization, MPOpt quadratic assignment interface,
Hungarian assignment, polar decomposition, and random streams required by WP.
Imports are package-relative. Unused helpers for external baselines and GPU
experiments have been removed. The numerical fitting algorithm is retained.
The embedding reduction and alignment metric follow the same public implementation.
The original copyright notice and full license are in `src/consensusalign/_wp/LICENSE`.
These credits identify the upstream numerical solver.

MPOpt is accessed through the separately installed `pylibmgm` dependency.
PyTorch, NumPy, SciPy, scikit-learn, threadpoolctl, and tqdm are separately
installed dependencies and retain their own licenses. Their source distributions
and binaries are not included in this archive.

The optional public feature files come from
https://huggingface.co/datasets/schnaus/unpaired-rosetta-embeddings.
They remain subject to their source dataset and encoder licenses. Feature files,
encoder weights, and datasets are not redistributed in this archive.
