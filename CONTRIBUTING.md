# Contributing

Install `python -m pip install -e ".[data,dev]"`, then run `make lint`, `make test`,
and `make smoke` before proposing a change. Tests use small synthetic inputs and
need neither downloaded features nor encoder weights.

Report a reproducibility issue with the Python and dependency versions, command,
configuration, seed index, and the relevant metrics. Remove credentials and local
personal paths from logs. For numerical changes, describe the affected operation
and provide a focused test or a result from the same declared protocol.

Keep the final pooling, bridge and distribution construction easy to follow.
Place comparisons and experimental tooling outside `core.py`. Use the same
row identities within each modality and keep evaluation pairs out of alignment
fitting unless an explicit anchor budget is declared.

The reproducibility runner hashes inputs and artifacts. Use a new output directory
when changing a protocol. Do not edit completed metric files to update a figure.
