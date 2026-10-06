"""Exact comparisons against commit 87966fb, including RNG and timing histories."""

import copy
import importlib
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from datasift import algorithms, datasets, influence, metrics, partitioning, utils
from datasift.models import LogisticRegression, NeuralNetwork, SVM


ALGORITHMS = [
    "mab_algorithm",
    "mab_inf_algorithm",
    "mab_algorithm_base",
    "mab_algorithm_dist",
    "mab_algorithm_acc",
    "random_algorithm",
    "entropy_based_algorithm",
    "inf_algorithm",
    "random_algorithm_acc",
]


class ControlledModel:
    """Deterministic candidates that exercise both accepted and rejected batches."""

    def __init__(self, mode):
        self.stage = 0
        self.mode = mode

    def fit(self, x, y):
        self.stage += 1

    def predict_proba(self, x):
        x = np.asarray(x)
        if self.mode == "accuracy":
            # The fixture alternates target labels. Stage 1 improves accuracy;
            # stage 2 worsens it, exercising both accuracy acceptance policies.
            bias = [0.25, 0.05, 0.25, 0.05, 0.01][min(self.stage, 4)]
            labels = 2 * (np.arange(len(x)) % 2) - 1
            return (
                0.5 + 0.1 * labels + bias * np.tanh(x[:, 1]) + 0.005 * np.tanh(x[:, 0])
            )
        amplitudes = (
            [0.32, 0.22, 0.38, 0.10, 0.01]
            if self.mode == "reject"
            else [0.32, 0.22, 0.12, 0.04, 0.01]
        )
        amplitude = amplitudes[min(self.stage, 4)]
        return 0.5 + amplitude * np.tanh(x[:, 1]) + 0.005 * np.tanh(x[:, 0])


def frames():
    rng = np.random.RandomState(17)

    def make(size, offset=0):
        return pd.DataFrame(
            {
                "age": rng.normal(size=size),
                "gender": np.tile([0, 0, 1, 1], size // 4),
                "signal": rng.normal(size=size),
                "income": np.tile([0, 1, 0, 1], size // 4),
            },
            index=np.arange(offset, offset + size),
        )

    return make(16), make(12, 100), make(12, 200), make(32, 300)


def compare(actual, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare(left, right)
    else:
        np.testing.assert_array_equal(actual, expected)


def invoke(module, name, model, seed, metric, budget=7, tau=0.02, exhausted=False):
    train, val, test, pool = frames()
    pool_input = (
        [pool.iloc[:12], pool.iloc[12:24], pool.iloc[24:]]
        if name.startswith("mab")
        else pool
    )
    if exhausted and name.startswith("mab"):
        pool_input = [pool.iloc[:0], pool.iloc[:1], pool.iloc[2:10]]
    kwargs = dict(
        dataset_name="adult",
        train_orig=train,
        val_orig=val,
        test_orig=test,
        Model=model,
        target_col="income",
        mini_batch_size=3,
        tau=tau,
        budget=budget,
        metric_label=metric,
    )
    if name.startswith("mab"):
        kwargs.update(max_iteration=7, alpha=0.1)
    if name in {"mab_algorithm", "mab_inf_algorithm"}:
        kwargs["beta"] = 0.2
    if name in {"mab_algorithm_dist", "mab_algorithm_acc"}:
        # Deliberately asymmetric to check the original undirected neighbor rule.
        kwargs["Euclid_normalized_d"] = np.array(
            [[0, 0.0002, 0.8], [0.9, 0, 0.0003], [0.7, 0.6, 0]]
        )
    clock = itertools.count()
    saved_time = module.time
    module.time = SimpleNamespace(time=lambda: float(next(clock)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        result = getattr(module, name)(pool_input, **kwargs)
        rng_state = np.random.get_state()
    finally:
        module.time = saved_time
    # Callers must retain their seed training frame and partition contents.
    compare(train, frames()[0])
    return result, rng_state


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("metric", [0, 1, 2])
@pytest.mark.parametrize("seed", [0, 42, 107])
@pytest.mark.parametrize("mode", ["reject", "improve", "accuracy"])
def test_algorithm_exact_outputs_and_rng(original, name, metric, seed, mode):
    expected = invoke(original["Algorithms"], name, ControlledModel(mode), seed, metric)
    actual = invoke(algorithms, name, ControlledModel(mode), seed, metric)
    compare(actual, expected)


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize(
    "case", ["zero_budget", "threshold", "exhausted", "partial_budget"]
)
def test_algorithm_boundaries(original, name, case):
    kwargs = (
        {"budget": 0}
        if case == "zero_budget"
        else {"tau": 1.0}
        if case == "threshold"
        else {"exhausted": True}
        if case == "exhausted"
        else {"budget": 4}
    )
    compare(
        invoke(algorithms, name, ControlledModel("improve"), 11, 0, **kwargs),
        invoke(
            original["Algorithms"], name, ControlledModel("improve"), 11, 0, **kwargs
        ),
    )


@pytest.mark.parametrize("name", ALGORITHMS)
def test_algorithms_with_real_torch_training(original, name):
    train, _, _, _ = frames()
    x, _, y, _, _, _ = partitioning.prepare_train_data(train, train, "income")
    model = LogisticRegression(3, epoch_num=5)
    model.fit(x, y)
    compare(
        invoke(algorithms, name, copy.deepcopy(model), 42, 0),
        invoke(original["Algorithms"], name, copy.deepcopy(model), 42, 0),
    )


@pytest.mark.parametrize("kind", ["normal", "identical", "empty", "reordered"])
def test_centroid_distances_preserve_alignment_and_nan(original, kind):
    _, _, _, pool = frames()
    clusters = [pool.iloc[:8], pool.iloc[8:16], pool.iloc[16:]]
    if kind == "identical":
        clusters = [pool, pool.copy()]
    elif kind == "empty":
        clusters[0] = pool.iloc[:0]
    elif kind == "reordered":
        clusters[1] = clusters[1][list(reversed(pool.columns))]
    with np.errstate(invalid="ignore", divide="ignore"):
        compare(
            partitioning.compute_normalized_distances(clusters),
            original["Misc"].compute_normalized_distances(clusters),
        )


@pytest.mark.parametrize(
    "dataset,column",
    [
        ("adult", "gender"),
        ("income", "SEX"),
        ("employment", "RAC1P"),
        ("public", "SEX"),
        ("mobility", "RAC1P"),
        ("credit", "age"),
    ],
)
@pytest.mark.parametrize("empty_group", [False, True])
def test_group_disparity(original, dataset, column, empty_group):
    target = datasets.get_target_sensitive_attribute(dataset)[0]
    frame = pd.DataFrame({column: [0, 1, 1, 0, np.nan], target: [1, 0, 1, 0, 1]})
    if empty_group:
        frame[column] = 0
    compare(
        partitioning.calculate_group_disparity(frame, dataset),
        original["Misc"].calculate_group_disparity(frame, dataset),
    )


def test_neighbors_asymmetric(original):
    distance = np.array([[0, 0.1, np.nan], [0.8, 0, 0.05], [0.9, 0.6, 0]])
    assert partitioning.find_neighbors([None] * 3, distance, 0.1) == original[
        "Misc"
    ].find_neighbors([None] * 3, distance, 0.1)


def test_preparation_and_duplicate_index_batches(original):
    train, val, _, _ = frames()
    compare(
        partitioning.prepare_train_data(train, val, "income"),
        original["Misc"].prepare_train_data(train, val, "income"),
    )
    train.index = [i // 2 for i in range(len(train))]
    compare(
        partitioning.get_minibatches(train, 3),
        original["Misc"].get_minibatches(train, 3),
    )


def test_adult_loader_exact(original, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1] / "data" / "raw")
    expected = original["DatasetExt"].load_adult()
    compare(datasets.load_adult(), expected)
    monkeypatch.chdir(Path(__file__).resolve().parents[1] / "notebooks")
    compare(datasets.load_adult(), expected)


def test_credit_preprocessing(original):
    frame = pd.DataFrame(
        {
            "MonthlyIncome": [1.0, 3.0, np.nan, 200.0],
            "NumberOfDependents": [1.0, np.nan, 2.0, 1.0],
            "RevolvingUtilizationOfUnsecuredLines": [0.0, 0.5, 1.0, 5.0],
            "DebtRatio": [1.0, 2.0, 3.0, 200.0],
            "age": [34, 35, 70, 22],
        }
    )
    compare(
        datasets.preprocess_credit(frame.copy()),
        original["DatasetExt"].preprocess_credit(frame.copy()),
    )


@pytest.mark.parametrize("model_class", [LogisticRegression, SVM, NeuralNetwork])
def test_model_training_probabilities_gradients_and_hessians(original, model_class):
    x = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([1, 0, 1, 0])
    old_class = getattr(original["Classifier"], model_class.__name__)
    expected_model = old_class(2, epoch_num=2)
    actual_model = model_class(2, epoch_num=2)
    expected_model.fit(x, y)
    actual_model.fit(x, y)
    compare(actual_model.predict_proba(x), expected_model.predict_proba(x))
    loss_name = {
        LogisticRegression: "logistic_loss_torch",
        SVM: "svm_loss_torch",
        NeuralNetwork: "nn_loss_torch",
    }[model_class]
    expected_loss = getattr(original["utils"], loss_name)
    actual_loss = getattr(utils, loss_name)
    compare(
        influence.get_del_L_del_theta(actual_model, x, y, actual_loss),
        original["influence"].get_del_L_del_theta(expected_model, x, y, expected_loss),
    )
    compare(
        influence.get_hessian_all_points(actual_model, x, y, actual_loss),
        original["influence"].get_hessian_all_points(
            expected_model, x, y, expected_loss
        ),
    )


@pytest.mark.parametrize(
    "legacy,current",
    [
        ("Algorithms", "algorithms"),
        ("Classifier", "models"),
        ("DatasetExt", "datasets"),
        ("Misc", "partitioning"),
        ("metrics", "metrics"),
        ("influence", "influence"),
        ("utils", "utils"),
    ],
)
def test_legacy_imports_share_module_identity(legacy, current):
    assert importlib.import_module(legacy) is importlib.import_module(
        f"datasift.{current}"
    )
