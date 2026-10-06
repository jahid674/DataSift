"""Offline demonstration of the current DataSift API using synthetic data."""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasift.algorithms import mab_algorithm
from datasift.models import LogisticRegression
from datasift.partitioning import prepare_train_data


def main():
    rng = np.random.RandomState(42)
    size = 240
    gender = np.tile([0, 1], size // 2)
    signal = rng.normal(size=size)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "gender": gender,
            "income": (
                signal + 0.7 * gender + rng.normal(scale=0.5, size=size) > 0.4
            ).astype(int),
        }
    )
    train_pool, evaluation = train_test_split(
        frame, test_size=0.3, random_state=42, stratify=frame["income"]
    )
    train, pool = train_test_split(
        train_pool, train_size=48, random_state=42, stratify=train_pool["income"]
    )
    validation, test = train_test_split(
        evaluation, test_size=0.5, random_state=42, stratify=evaluation["income"]
    )
    # Positional indices are required by the fairness functions.
    train, validation, test = [
        part.reset_index(drop=True) for part in [train, validation, test]
    ]
    partitions = [part.copy() for _, part in pool.groupby("gender", sort=True)]
    x_train, _, y_train, _, _, _ = prepare_train_data(train, test, "income")
    model = LogisticRegression(input_size=x_train.shape[1], epoch_num=50)
    model.fit(x_train, y_train)
    np.random.seed(42)
    torch.manual_seed(42)
    result = mab_algorithm(
        partitions,
        "adult",
        train,
        validation,
        test,
        model,
        "income",
        mini_batch_size=4,
        max_iteration=10,
        tau=0.01,
        budget=12,
        alpha=0.1,
        beta=0.1,
        metric_label=0,
    )
    (
        expanded,
        attempts,
        accepted,
        accepted_fairness,
        fairness,
        accuracy,
        _,
        clusters,
        _,
    ) = result
    print(f"Training rows: {len(train)} -> {len(expanded)}")
    print(f"Attempted batches: {attempts[-1]}; accepted batches: {accepted[-1]}")
    print(f"Accepted test fairness: {accepted_fairness}")
    print(f"Accepted test accuracy: {accuracy}")
    print(f"Partition history: {clusters}")


if __name__ == "__main__":
    main()
