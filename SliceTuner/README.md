# SliceTuner comparison baseline

This directory retains the DataSift experiment's adaptation of the
[Slice Tuner research framework](https://github.com/khtae8250/SliceTuner), with
TensorFlow/Keras models, allocation baselines, and the `System_T` implementation.
See the [Slice Tuner paper](https://arxiv.org/abs/2003.04549) for the method.

Install optional dependencies from the repository root:

```bash
python -m pip install -e '.[slicetuner]'
```

Use package imports such as `from SliceTuner.slicetuner import SliceTunerRunner`
and `from SliceTuner.baseline import Baseline`. The runner receives prepared
training/validation arrays, slice definitions, and per-slice acquisition pools.
The exploratory notebook is [notebooks/SliceTuner_Experiment.ipynb](../notebooks/SliceTuner_Experiment.ipynb).

`DatasetExt.py` delegates to the shared DataSift loader. `Misc.py` retains
baseline-specific research utilities. Earlier development copies are preserved
in `archive/` for reference and excluded from the installed package. Dataset
copies have been consolidated into `data/raw/`.

This optional baseline is separate from the core differential test suite.
