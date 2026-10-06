# Contributing

Create a branch from `main` and install the development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
ruff format --check .
python examples/quickstart.py
```

Changes to acquisition policies, metric definitions, preprocessing, sampling,
or model training must be called out explicitly in the pull request. Performance
refactors must preserve outputs and RNG consumption under the same runtime.
See [docs/algorithms.md](docs/algorithms.md) for existing behavior that must not
be silently corrected.

Keep `tests/reference/` unchanged. Add meaningful regression cases for new edge
conditions. Explain the intended behavior and validation in the pull request.
Avoid committing bytecode, downloaded dataset caches, credentials, generated
figures, or saved notebook execution outputs. Put local experiment artifacts in
`outputs/` and dataset inputs in `data/raw/`.

Report issues with the Python/dependency versions, dataset and model used,
random seed, exact command or minimal example, and the observed result.
