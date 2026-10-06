# DataSift

[![Tests](https://github.com/jahid674/DataSift/actions/workflows/tests.yml/badge.svg)](https://github.com/jahid674/DataSift/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10–3.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/EDBT-2026-orange)](https://doi.org/10.48786/edbt.2026.39)

**Selective data expansion for model fairness.** DataSift chooses useful labeled
data from an available pool to expand a classifier's training set. It combines
partition selection through multi-armed bandits with influence-based data
valuation. This repository contains the research implementation, comparison
baselines, and experiment notebooks.

## Publication

Jahid Hasan and Romila Pradhan. **Selective Data Expansion for Model
Performance.** EDBT 2026, pp. 488–502.
[DOI](https://doi.org/10.48786/edbt.2026.39) ·
[Published paper](https://www.openproceedings.org/2026/conf/edbt/paper-146.pdf) ·
[Local published PDF](docs/papers/edbt-2026.pdf) ·
[Earlier arXiv preprint](https://arxiv.org/abs/2412.03009)

The earlier preprint is titled *Data Acquisition for Improving Model Fairness
using Reinforcement Learning*. Publication assets and version information are
listed in [docs/papers/README.md](docs/papers/README.md).

## Installation

Use Python **3.10, 3.11, or 3.12**. The default implementation runs on CPU;
a GPU is not required. Clone the repository to use the bundled data and notebooks.

```bash
git clone https://github.com/jahid674/DataSift.git
cd DataSift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.
Direct runtime dependencies are pinned in [requirements.txt](requirements.txt).
For a requirements-based installation, use `python -m pip install -r requirements.txt`
and run code from the checkout. Editable installation is recommended so imports
also work from the notebook directory.

| Extra | Install | Purpose |
| --- | --- | --- |
| Development | `python -m pip install -e '.[dev]'` | Regression tests and package builds |
| Notebooks | `python -m pip install -e '.[notebooks]'` | JupyterLab and plotting |
| SliceTuner | `python -m pip install -e '.[slicetuner]'` | TensorFlow and CVXPY comparison baseline |

Equivalent requirements files are provided for each extra. TensorFlow is optional
and is not imported by the main DataSift package.

## Quick start

Run the offline example, which constructs synthetic data, creates separate
training/validation/test sets, fits logistic regression, and acquires batches:

```bash
python examples/quickstart.py
```

The example uses the `mab_algorithm` variant with random sampling inside each
partition. To use influence-ranked batches, prepare ordered partitions and call
`mab_inf_algorithm`; the acquisition function consumes that ordering rather than
computing influence scores itself.

The main API retains the existing research arguments and return values:

```python
from datasift.algorithms import mab_algorithm

result = mab_algorithm(
    partition_data, dataset_name, train_df, validation_df, test_df,
    fitted_model, target_column,
    mini_batch_size=10, max_iteration=100, tau=0.01,
    budget=100, alpha=0.1, beta=0.1, metric_label=0,
)
expanded_train = result[0]
accepted_test_fairness = result[3]
accepted_test_accuracy = result[5]
```

All frames must share a numeric feature schema and a binary target column. The
model must return a **one-dimensional array of positive-class probabilities**.
Use `prepare_train_data` to fit the initial model and reset evaluation indices
to contiguous positions. See [examples/quickstart.py](examples/quickstart.py)
for the complete setup and [docs/algorithms.md](docs/algorithms.md) for contracts,
variant differences, and preserved research behavior.

## Algorithms and metrics

| Function | Selection strategy |
| --- | --- |
| `mab_algorithm` | DataSift bandit with random batches inside partitions |
| `mab_inf_algorithm` | DataSift bandit with preordered, influence-ranked batches |
| `mab_algorithm_base` | Bandit reward ablation without distance/accuracy propagation |
| `mab_algorithm_dist` | Neighbor-distance reward propagation baseline |
| `mab_algorithm_acc` | Bandit variant accepting accuracy improvements |
| `random_algorithm` | Random samples from a flat pool |
| `entropy_based_algorithm` | Highest prediction-entropy samples |
| `inf_algorithm` | First samples from a preordered flat pool |
| `random_algorithm_acc` | Random baseline with accuracy-based acceptance |

`metric_label=0` selects statistical parity, `1` selects true-positive-rate
parity, and `2` selects predictive parity. These functions use predicted
probabilities and report **protected minus privileged** group differences;
smaller absolute values mean less disparity. Accuracy uses the strict threshold
`p > 0.5`. The partition-level group-disparity helper instead uses **privileged
minus protected**, with `+1` denominator smoothing. These existing definitions
are preserved.

## Data

The bundled Adult files are the same cleaned inputs used by the original
checkout. German Credit data is retained as a research asset, but there is no
supported German loader. Give Me Some Credit CSVs and downloaded ACS caches are
local inputs, excluded from new commits. See [data/README.md](data/README.md) for
sources, file placement, preprocessing, and dataset licenses.

Supported `load_data` identifiers are `adult`, `credit`, `income`, `employment`,
`public`, and `mobility`. ACS loading downloads 2018 California data through
Folktables by default and therefore requires network access. The travel task has
an existing identifier mismatch; use `load_acs_data("travel")` directly for data
extraction, and see the limitations in the algorithm documentation.

```python
from datasift.datasets import load_data
train_pool, evaluation_data = load_data("adult")
```

Dataset loaders first look for files in the current directory, then in the
checkout's `data/raw/`. Data files are not bundled in the Python wheel.

## Repository layout

```text
datasift/                 Main Python package
  algorithms.py          Acquisition strategies and ablations
  models.py              PyTorch/sklearn model wrappers
  datasets.py            Dataset loading and preprocessing
  partitioning.py        Sampling, clustering, and influence ranking helpers
  metrics.py             Fairness metrics and their derivatives
  influence.py           Gradient, Hessian, and influence computations
  utils.py               Loss functions and tensor utilities
examples/quickstart.py    Supported offline example
notebooks/               Exploratory research and plotting notebooks
SliceTuner/              Optional comparison baseline
data/                    Dataset documentation and bundled UCI inputs
docs/                    Algorithm contracts and publication assets
tests/                   Differential regression tests and original reference
.github/workflows/       Automated tests, example run, and package build
```

Root-level modules such as `Algorithms.py`, `Classifier.py`, and `Misc.py` are
compatibility imports. Existing imports such as `from Algorithms import
mab_algorithm` continue to work. `config.json` keeps the original tensor dtype
setting; imports use the working-directory file when present and the packaged
default otherwise.

## Experiments and verification

```bash
python -m pip install -e '.[dev,notebooks]'
python -m pytest -q
jupyter lab notebooks/
```

The regression suite compares outputs against the unmodified source from commit
`87966fb`, using matched seeds and a deterministic clock. It checks acquisition
results, histories, budget boundaries, exhausted partitions, RNG state, training,
preprocessing, gradients, and Hessians. Runtime measurements naturally change
with optimizations and hardware. Floating-point behavior can also differ between
dependency versions or platforms; use the pinned runtime for comparisons.

The research notebooks retain their exploratory cells, including some older API
calls. Saved execution outputs have been cleared and plot paths made relative.
The supported entry point is the offline example; the notebooks are not an
automated end-to-end reproduction of every paper figure.

## Citation

GitHub's **Cite this repository** action uses [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{hasan2026datasift,
  author    = {Jahid Hasan and Romila Pradhan},
  title     = {Selective Data Expansion for Model Performance},
  booktitle = {Proceedings of the 29th International Conference on
               Extending Database Technology},
  year      = {2026},
  pages     = {488--502},
  doi       = {10.48786/edbt.2026.39}
}
```

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Original project code is released under
the [MIT license](LICENSE). Papers, datasets, and third-party materials retain
their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
