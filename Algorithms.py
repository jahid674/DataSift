import numpy as np
import pandas as pd
import time
import copy
from sklearn.preprocessing import StandardScaler
from metrics import *
from Misc import *


def mab_algorithm(partition_data, dataset_name, train_orig, val_orig, test_orig,
                  Model, target_col, mini_batch_size, max_iteration, tau, budget, alpha, beta, metric_label):
    # ----------------------------
    # Prepare TRAIN+TEST (reporting only)
    # ----------------------------
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )

    # ----------------------------
    # Prepare TRAIN+VAL (internal decision only)
    # ----------------------------
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    # ----------------------------
    # Initialize: report on TEST, decide/stop on VAL
    # ----------------------------
    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)

    best_parity_val = ini_parity_val  # stopping based on validation

    clusters = partition_data.copy()
    stat, stat_ex = [ini_parity_test], [ini_parity_test]   # report test
    acc_stat = [acc_ini_test]                              # report test
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha_ = alpha

    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)

    M = copy.deepcopy(Model)
    iteration = 0
    iteration_mono = 0
    remaining_budget = budget
    

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration - 1:
            break
        if iteration_mono >= round(budget / mini_batch_size):
            break

        selected_cluster_indices = np.argmax(U)
        exhausted_clusters = set()
        K = min(mini_batch_size, remaining_budget)

        while len(clusters[selected_cluster_indices]) < K:
            exhausted_clusters.add(selected_cluster_indices)
            U[list(exhausted_clusters)] = -np.inf
            selected_cluster_indices = np.argmax(U)
            if selected_cluster_indices in exhausted_clusters:
                break

        if len(clusters[selected_cluster_indices]) < K:
            break

        iteration += 1
        mini_batch_indices = np.random.choice(len(clusters[selected_cluster_indices]), size=K, replace=False)
        mini_batch = clusters[selected_cluster_indices].iloc[mini_batch_indices]
        Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, X_train_orig_combined, _ = prepare_train_data(
            Ttrain_combined, test_orig, target_col
        )

        # ----------------------------
        # BEFORE/AFTER for DECISION: use VALIDATION
        # ----------------------------
        y_pred_val = M.predict_proba(x_val)
        before_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
        accuracy_bef_val = computeAccuracy(y_val, y_pred_val)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)
        accuracy_1_val = computeAccuracy(y_val, y_pred_val_up)

        del_acc = accuracy_1_val - accuracy_bef_val
        delta = -(np.abs(after_val) - np.abs(before_val))

        # ----------------------------
        # Reporting values: compute on TEST
        # ----------------------------
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        if delta > 0 and np.abs(after_val) < np.abs(best_parity_val):
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget -= K
            i_values_ex_stat.append(iteration_mono)

            stat_ex.append(after_test)          # report test
            acc_stat.append(accuracy_1_test)    # report test
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)

            best_parity_val = after_val

        normalized_distance = compute_normalized_distances(clusters)

        for j in range(num_clusters):
            br[j] = calculate_group_disparity(clusters[j], dataset_name)
            if j == selected_cluster_indices:
                r[iteration, j] = (delta) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)
            else:
                r[iteration, j] = ((delta) / (1 + np.abs(br[j]))) + (beta * (del_acc) * (1 + normalized_distance[selected_cluster_indices, j]))
                n[iteration, j] = n[iteration - 1, j]

            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)
            U[j] = R[iteration, j] + alpha_ * np.sqrt(2 * np.log(np.sum(n[iteration, :]) + 1) / ((n[iteration, j]) + 1))

        cluster_count.append(selected_cluster_indices + 1)

        cluster_df = clusters[selected_cluster_indices]
        cluster_df = cluster_df.drop(cluster_df.index[list(mini_batch_indices)])
        clusters[selected_cluster_indices] = cluster_df

        iteration_time = time.time() - start_time
        i_values_stat.append(iteration)
        stat.append(after_test)  # report test
        iteration_time1 += iteration_time

    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat,
            time_per_iteration_stat, cluster_count, iteration_time1)

def mab_inf_algorithm(
    partition_data, dataset_name,
    train_orig, val_orig, test_orig,     # ✅ added val_orig
    Model, target_col,
    mini_batch_size, max_iteration, tau, budget, alpha, beta, metric_label
):
    # ----------------------------
    # Prepare TRAIN+TEST (reporting only)
    # ----------------------------
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )

    # ----------------------------
    # Prepare TRAIN+VAL (internal decision only)
    # ----------------------------
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    # ----------------------------
    # Initialize: report on TEST, decide/stop on VAL
    # ----------------------------
    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)

    best_parity_val = ini_parity_val  # ✅ stopping/accept logic based on validation

    clusters = partition_data.copy()

    # Reported stats are TEST-based
    stat, stat_ex = [ini_parity_test], [ini_parity_test]
    acc_stat = [acc_ini_test]

    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []

    alpha = alpha
    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)

    M = copy.deepcopy(Model)
    iteration = 0
    iteration_mono = 0
    remaining_budget = budget

    # ✅ stopping criterion uses validation fairness
    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()

        if iteration >= max_iteration - 1:
            break
        if iteration_mono >= round(budget / mini_batch_size):
            break

        iteration += 1
        selected_cluster_indices = np.argmax(U)
        exhausted_clusters = set()
        K = min(mini_batch_size, remaining_budget)

        while len(clusters[selected_cluster_indices]) < K:
            exhausted_clusters.add(selected_cluster_indices)
            U[list(exhausted_clusters)] = -np.inf
            selected_cluster_indices = np.argmax(U)
            if selected_cluster_indices in exhausted_clusters:
                break

        if len(clusters[selected_cluster_indices]) < K:
            break

        mini_batch_indices = np.arange(K)
        mini_batch = clusters[selected_cluster_indices][:K]
        Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        # NOTE: keep your original call signature/behavior here
        X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(
            Ttrain_combined, test_orig, target_col
        )

        # ----------------------------
        # BEFORE/AFTER for DECISION: use VALIDATION
        # ----------------------------
        y_pred_val = M.predict_proba(x_val)
        before_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
        accuracy_bef_val = computeAccuracy(y_val, y_pred_val)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)
        accuracy_1_val = computeAccuracy(y_val, y_pred_val_up)

        del_acc = accuracy_1_val - accuracy_bef_val
        delta = -(np.abs(after_val) - np.abs(before_val))  # ✅ delta from validation

        # ----------------------------
        # Reporting values: compute on TEST
        # ----------------------------
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        # Accept/reject based on VAL (but append TEST)
        if delta > 0 and np.abs(after_val) < np.abs(best_parity_val):
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget = remaining_budget - K

            i_values_ex_stat.append(iteration_mono)

            # ✅ append TEST-based reporting values
            stat_ex.append(after_test)
            acc_stat.append(accuracy_1_test)

            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)

            best_parity_val = after_val  # ✅ update best using validation
            if np.abs(stat_ex[-1]) <= tau:
                break

        # Bandit reward updates unchanged (uses delta and del_acc as before)
        normalized_distance = compute_normalized_distances(clusters)

        for j in range(num_clusters):
            br[j] = calculate_group_disparity(clusters[j], dataset_name)

            if j == selected_cluster_indices:
                r[iteration, j] = (delta) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)
            else:
                r[iteration, j] = ((delta) / (1 + np.abs(br[j]))) + beta * (del_acc) * (1 + normalized_distance[selected_cluster_indices, j])
                n[iteration, j] = n[iteration - 1, j]

            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)
            U[j] = R[iteration, j] + alpha * np.sqrt(
                2 * np.log(np.sum(n[iteration, :]) + 1) / ((n[iteration, j]) + 1)
            )

        cluster_count.append(selected_cluster_indices + 1)

        # Remove used points from selected cluster
        cluster_df = clusters[selected_cluster_indices]
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters[selected_cluster_indices] = cluster_df

        # ✅ per-iteration reporting on TEST
        i_values_stat.append(iteration)
        stat.append(after_test)

        iteration_time = time.time() - start_time
        iteration_time1 += iteration_time

    return (
        Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat,
        time_per_iteration_stat, cluster_count, iteration_time1
    )


def mab_algorithm_base(partition_data, dataset_name, train_orig, val_orig, test_orig,
                       Model, target_col, mini_batch_size, max_iteration, tau, budget, alpha, metric_label):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
    best_parity_val = ini_parity_val

    clusters = partition_data.copy()
    stat, stat_ex = [ini_parity_test], [ini_parity_test]
    acc_stat = [acc_ini_test]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha_ = alpha

    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)

    M = copy.deepcopy(Model)
    iteration = 0
    iteration_mono = 0
    remaining_budget = budget

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration - 1:
            break
        if iteration_mono >= round(budget / mini_batch_size):
            break

        max_indices = np.where(U == np.max(U))[0]
        selected_cluster_indices = np.random.choice(max_indices)

        exhausted_clusters = set()
        K = min(mini_batch_size, remaining_budget)
        while len(clusters[selected_cluster_indices]) < K:
            exhausted_clusters.add(selected_cluster_indices)
            U[list(exhausted_clusters)] = -np.inf
            selected_cluster_indices = np.argmax(U)
            if selected_cluster_indices in exhausted_clusters:
                break

        if len(clusters[selected_cluster_indices]) < K:
            break

        iteration += 1
        mini_batch_indices = np.random.choice(len(clusters[selected_cluster_indices]), size=K, replace=False)
        mini_batch = clusters[selected_cluster_indices].iloc[mini_batch_indices]
        Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, _, _ = prepare_train_data(
            Ttrain_combined, test_orig, target_col
        )

        # decision on VAL
        y_pred_val = M.predict_proba(x_val)
        before_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)

        delta = np.abs(after_val) - np.abs(before_val)

        # report on TEST
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        if delta < 0 and np.abs(after_val) < np.abs(best_parity_val):
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget -= K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after_test)
            acc_stat.append(accuracy_1_test)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            best_parity_val = after_val

        normalized_distance = compute_normalized_distances(clusters)
        for j in range(num_clusters):
            br[j] = calculate_group_disparity(clusters[j], dataset_name)
            if j == selected_cluster_indices:
                r[iteration, j] = -(delta) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)
            else:
                r[iteration, j] = (-(delta)) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)
            U[j] = R[iteration, j] + alpha_ * np.sqrt(2 * np.log(np.sum(n[iteration, :]) + 1) / ((n[iteration, j]) + 1))

        cluster_count.append(selected_cluster_indices + 1)
        cluster_df = clusters[selected_cluster_indices]
        cluster_df = cluster_df.drop(cluster_df.index[list(mini_batch_indices)])
        clusters[selected_cluster_indices] = cluster_df

        iteration_time = time.time() - start_time
        i_values_stat.append(iteration)
        stat.append(after_test)
        iteration_time1 += iteration_time

    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat,
            time_per_iteration_stat, cluster_count, iteration_time1)


def mab_algorithm_dist(partition_data, dataset_name, train_orig, val_orig, test_orig,
                       Model, target_col, mini_batch_size, max_iteration, tau, budget, Euclid_normalized_d, alpha, metric_label):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
    best_parity_val = ini_parity_val

    clusters = partition_data.copy()
    stat, stat_ex = [ini_parity_test], [ini_parity_test]
    acc_stat = [acc_ini_test]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    Ttrain_updated = copy.deepcopy(train_orig)

    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)

    M = copy.deepcopy(Model)
    iteration = 0
    iteration_mono = 0
    remaining_budget = budget

    # sequential before tracking should be on VAL for decision
    before_val = ini_parity_val

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration - 1:
            break
        if iteration_mono >= round(budget / mini_batch_size):
            break

        iteration += 1
        max_indices = np.where(U == np.max(U))[0]
        selected_cluster_indices = np.random.choice(max_indices)

        exhausted_clusters = set()
        K = min(mini_batch_size, remaining_budget)
        while len(clusters[selected_cluster_indices]) < K:
            exhausted_clusters.add(selected_cluster_indices)
            U[list(exhausted_clusters)] = -np.inf
            selected_cluster_indices = np.argmax(U)
            if selected_cluster_indices in exhausted_clusters:
                break

        if len(clusters[selected_cluster_indices]) < K:
            break

        mini_batch_indices = np.random.choice(len(clusters[selected_cluster_indices]), size=K, replace=False)
        mini_batch = clusters[selected_cluster_indices].iloc[mini_batch_indices]
        Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, _, _ = prepare_train_data(
            Ttrain_combined, test_orig, target_col
        )

        # decision on VAL (note: original code used rolling "before"; keep same behavior but on VAL)
        accuracy_bef_val = computeAccuracy(y_val, M.predict_proba(x_val))

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)
        accuracy_1_val = computeAccuracy(y_val, y_pred_val_up)

        delta = np.abs(after_val) - np.abs(before_val)
        del_acc = accuracy_1_val - accuracy_bef_val
        before_val = after_val

        # report on TEST
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        if delta < 0:
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget -= K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after_test)
            acc_stat.append(accuracy_1_test)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)

            cluster_count.append(selected_cluster_indices + 1)
            cluster_df = clusters[selected_cluster_indices]
            cluster_df = cluster_df.drop(cluster_df.index[list(mini_batch_indices)])
            clusters[selected_cluster_indices] = cluster_df
            best_parity_val = after_val

        dist = 0.001
        for j in range(num_clusters):
            n_dist = Euclid_normalized_d[j, selected_cluster_indices]
            if j in find_neighbors(clusters, Euclid_normalized_d, dist)[selected_cluster_indices]:
                r[iteration, j] = delta * (1 - n_dist / dist)
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)
                R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)
            else:
                r[iteration, j] = 0
                n[iteration, j] = n[iteration - 1, j]
                R[iteration, j] = R[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)
            U[j] = R[iteration, j] + alpha * np.sqrt(2 * np.log(np.sum(n[iteration, :]) + 1) / ((n[iteration, j]) + 1))

        iteration_time = time.time() - start_time
        time_per_iteration_stat.append(time_per_iteration_stat[-1] + iteration_time)
        i_values_stat.append(iteration)
        stat.append(after_test)
        iteration_time1 += iteration_time

    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat,
            time_per_iteration_stat, cluster_count, iteration_time1)


def random_algorithm(clusters, dataset_name, train_orig, val_orig, test_orig, Model, target_col, mini_batch_size, tau, budget, metric_label):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    # report init on TEST
    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    # stop/decision on VAL
    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
    best_parity_val = ini_parity_val

    stat_ran, stat_ex_ran = [ini_parity_test], [ini_parity_test]
    acc_ran = [acc_ini_test]
    acc_ex, i_values_ran, i_values_ex_ran = [], [0], [0]
    time_per_iteration_ran, iteration_time1 = [0], 0

    Ttrain_updated = copy.deepcopy(train_orig)
    train_shape = []
    M = copy.deepcopy(Model)
    iteration = 0
    remaining_budget = budget

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration >= round(budget / mini_batch_size):
            break

        iteration += 1
        mini_batch_indices = np.random.choice(len(clusters), size=K, replace=False)
        mini_batch = clusters.iloc[mini_batch_indices]
        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, _, _ = prepare_train_data(
            Ttrain_updated_combined, test_orig, target_col
        )

        # decision delta on VAL
        y_pred_val = M.predict_proba(x_val)
        before_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
        accuracy_bef_val = computeAccuracy(y_val, y_pred_val)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)

        # report on TEST
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        delta = np.abs(after_val) - np.abs(before_val)

        if delta < 0 or delta > 0:
            remaining_budget -= K
            acc_ex.append(accuracy_1_test)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            i_values_ex_ran.append(iteration)
            stat_ex_ran.append(after_test)
            train_shape.append(Ttrain_updated.shape)

        if np.abs(after_val) < np.abs(best_parity_val):
            best_parity_val = after_val

        cluster_df = clusters.drop(clusters.index[list(mini_batch_indices)])
        clusters = cluster_df

        i_values_ran.append(iteration)
        stat_ran.append(after_test)
        acc_ran.append(accuracy_1_test)

        iteration_time1 += (time.time() - start_time)

    return Ttrain_updated, i_values_ex_ran, i_values_ran, stat_ran, stat_ex_ran, acc_ran, iteration_time1


def entropy_based_algorithm(clusters, dataset_name, train_orig, val_orig, test_orig, Model, target_col, mini_batch_size, tau, budget, metric_label):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
    best_parity_val = ini_parity_val

    stat_ent, stat_ex_ent = [ini_parity_test], [ini_parity_test]
    acc_ent = [acc_ini_test]
    acc_ex, i_values_ent, i_values_ex_ent = [], [0], [0]
    time_per_iteration_ent, iteration_time1 = [0], 0

    Ttrain_updated = copy.deepcopy(train_orig)
    train_shape = []
    M = copy.deepcopy(Model)
    iteration = 0
    remaining_budget = budget

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration >= round(budget / mini_batch_size):
            break

        iteration += 1

        y_pred_clusters = M.predict_proba(clusters.drop([target_col], axis=1).values)
        p = y_pred_clusters
        entropy = - (p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12))
        mini_batch_indices = np.argsort(entropy)[-K:]
        mini_batch = clusters.iloc[mini_batch_indices]

        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, _, _ = prepare_train_data(
            Ttrain_updated_combined, test_orig, target_col
        )

        # decision on VAL
        y_pred_val = M.predict_proba(x_val)
        before_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)
        delta = np.abs(after_val) - np.abs(before_val)

        # report on TEST
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        if delta < 0 or delta > 0:
            remaining_budget -= K
            i_values_ex_ent.append(iteration)
            stat_ex_ent.append(after_test)
            acc_ex.append(accuracy_1_test)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            train_shape.append(Ttrain_updated.shape)

        if np.abs(after_val) < np.abs(best_parity_val):
            best_parity_val = after_val

        clusters = clusters.drop(clusters.index[list(mini_batch_indices)])

        i_values_ent.append(iteration)
        stat_ent.append(after_test)
        acc_ent.append(accuracy_1_test)

        iteration_time1 += (time.time() - start_time)

    return Ttrain_updated, i_values_ex_ent, i_values_ent, stat_ent, stat_ex_ent, acc_ent, iteration_time1


def inf_algorithm(clusters, dataset_name, train_orig, val_orig, test_orig, Model, target_col, mini_batch_size, tau, budget, metric_label):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
    best_parity_val = ini_parity_val

    stat_ran, stat_ex_ran = [ini_parity_test], [ini_parity_test]
    acc_ran = [acc_ini_test]
    acc_ex, i_values_ran, i_values_ex_ran = [], [0], [0]
    iteration_time1 = 0

    Ttrain_updated = copy.deepcopy(train_orig)
    train_shape = []
    M = copy.deepcopy(Model)
    iteration = 0
    remaining_budget = budget

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration >= round(budget / mini_batch_size):
            break

        iteration += 1
        mini_batch_indices = np.arange(K)
        mini_batch = clusters[:K]

        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, _, _ = prepare_train_data(
            Ttrain_updated_combined, test_orig, target_col
        )

        # decision on VAL
        y_pred_val = M.predict_proba(x_val)
        before_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)
        delta = np.abs(after_val) - np.abs(before_val)

        # report on TEST
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        if delta < 0 or delta > 0:
            remaining_budget -= K
            i_values_ex_ran.append(iteration)
            stat_ex_ran.append(after_test)
            acc_ex.append(accuracy_1_test)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            train_shape.append(Ttrain_updated.shape)

        if np.abs(after_val) < np.abs(best_parity_val):
            best_parity_val = after_val

        clusters = clusters.drop(clusters.index[list(mini_batch_indices)])

        i_values_ran.append(iteration)
        stat_ran.append(after_test)
        acc_ran.append(accuracy_1_test)

        iteration_time1 += (time.time() - start_time)

    return Ttrain_updated, i_values_ex_ran, i_values_ran, stat_ran, stat_ex_ran, acc_ran, iteration_time1


def mab_algorithm_acc(partition_data, dataset_name, train_orig, val_orig, test_orig,
                      Model, target_col, mini_batch_size, max_iteration, tau, budget, Euclid_normalized_d, alpha, metric_label):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
    best_parity_val = ini_parity_val

    clusters = partition_data.copy()
    stat, stat_ex = [ini_parity_test], [ini_parity_test]
    acc_stat = [acc_ini_test]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []

    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)

    M = copy.deepcopy(Model)
    iteration = 0
    iteration_mono = 0
    remaining_budget = budget

    # rolling before should be on VAL for decision
    before_val = ini_parity_val

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration - 1:
            break
        if iteration_mono >= round(budget / mini_batch_size):
            break

        iteration += 1
        max_indices = np.where(U == np.max(U))[0]
        selected_cluster_indices = np.random.choice(max_indices)

        exhausted_clusters = set()
        K = min(mini_batch_size, remaining_budget)
        while len(clusters[selected_cluster_indices]) < K:
            exhausted_clusters.add(selected_cluster_indices)
            U[list(exhausted_clusters)] = -np.inf
            selected_cluster_indices = np.argmax(U)
            if selected_cluster_indices in exhausted_clusters:
                break

        if len(clusters[selected_cluster_indices]) < K:
            break

        mini_batch_indices = np.random.choice(len(clusters[selected_cluster_indices]), size=K, replace=False)
        mini_batch = clusters[selected_cluster_indices].iloc[mini_batch_indices]
        Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, _, _ = prepare_train_data(
            Ttrain_combined, test_orig, target_col
        )

        # decision on VAL (accuracy-based)
        y_pred_val = M.predict_proba(x_val)
        accuracy_bef_val = computeAccuracy(y_val, y_pred_val)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)
        accuracy_1_val = computeAccuracy(y_val, y_pred_val_up)

        delta = np.abs(after_val) - np.abs(before_val)
        acc_del = accuracy_1_val - accuracy_bef_val
        before_val = after_val

        # report on TEST
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        if acc_del > 0:
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget -= K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after_test)
            acc_stat.append(accuracy_1_test)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            cluster_count.append(selected_cluster_indices + 1)

            cluster_df = clusters[selected_cluster_indices]
            cluster_df = cluster_df.drop(cluster_df.index[list(mini_batch_indices)])
            clusters[selected_cluster_indices] = cluster_df

            best_parity_val = after_val

        dist = 0.001
        for j in range(num_clusters):
            n_dist = Euclid_normalized_d[j, selected_cluster_indices]
            if j in find_neighbors(clusters, Euclid_normalized_d, dist)[selected_cluster_indices]:
                r[iteration, j] = delta * (1 - n_dist / dist)
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)
                R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)
            else:
                r[iteration, j] = 0
                n[iteration, j] = n[iteration - 1, j]
                R[iteration, j] = R[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)
            U[j] = R[iteration, j] + alpha * np.sqrt(2 * np.log(np.sum(n[iteration, :]) + 1) / ((n[iteration, j]) + 1))

        iteration_time = time.time() - start_time
        time_per_iteration_stat.append(time_per_iteration_stat[-1] + iteration_time)
        i_values_stat.append(iteration)
        stat.append(after_test)
        iteration_time1 += iteration_time

    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat,
            time_per_iteration_stat, cluster_count, iteration_time1)


def random_algorithm_acc(clusters, dataset_name, train_orig, val_orig, test_orig,
                         Model, target_col, mini_batch_size, tau, budget, metric_label):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(
        train_orig, test_orig, target_col
    )
    _, x_val, _, y_val, _, X_val_original = prepare_train_data(
        train_orig, val_orig, target_col
    )

    y_pred_test = Model.predict_proba(x_test)
    ini_parity_test = computeFairness(y_pred_test, X_test_original, y_test, metric_label, dataset_name)
    acc_ini_test = computeAccuracy(y_test, y_pred_test)

    y_pred_val = Model.predict_proba(x_val)
    ini_parity_val = computeFairness(y_pred_val, X_val_original, y_val, metric_label, dataset_name)
    best_parity_val = ini_parity_val

    stat_ran, stat_ex_ran = [ini_parity_test], [ini_parity_test]
    acc_ran = [acc_ini_test]
    acc_ex, i_values_ran, i_values_ex_ran = [], [0], [0]
    iteration_time1 = 0

    Ttrain_updated = copy.deepcopy(train_orig)
    train_shape = []
    M = copy.deepcopy(Model)
    iteration = 0
    remaining_budget = budget

    while np.abs(best_parity_val) > tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration >= round(budget / mini_batch_size):
            break

        iteration += 1
        mini_batch_indices = np.random.choice(len(clusters), size=K, replace=False)
        mini_batch = clusters.iloc[mini_batch_indices]

        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, _, _ = prepare_train_data(
            Ttrain_updated_combined, test_orig, target_col
        )

        # decision on VAL (accuracy delta)
        y_pred_val = M.predict_proba(x_val)
        accuracy_bef_val = computeAccuracy(y_val, y_pred_val)

        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)

        y_pred_val_up = model_up.predict_proba(x_val)
        after_val = computeFairness(y_pred_val_up, X_val_original, y_val, metric_label, dataset_name)
        accuracy_1_val = computeAccuracy(y_val, y_pred_val_up)

        # report on TEST
        y_pred_test_up = model_up.predict_proba(x_test)
        after_test = computeFairness(y_pred_test_up, X_test_original, y_test, metric_label, dataset_name)
        accuracy_1_test = computeAccuracy(y_test, y_pred_test_up)

        del_acc = accuracy_1_val - accuracy_bef_val

        if del_acc < 0 or del_acc > 0:
            remaining_budget -= K
            acc_ex.append(accuracy_1_test)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            i_values_ex_ran.append(iteration)
            stat_ex_ran.append(after_test)
            train_shape.append(Ttrain_updated.shape)

        if np.abs(after_val) < np.abs(best_parity_val):
            best_parity_val = after_val

        clusters = clusters.drop(clusters.index[list(mini_batch_indices)])

        i_values_ran.append(iteration)
        stat_ran.append(after_test)
        acc_ran.append(accuracy_1_test)

        iteration_time1 += (time.time() - start_time)

    return Ttrain_updated, i_values_ex_ran, i_values_ran, stat_ran, stat_ex_ran, acc_ran, iteration_time1



# def mab_inf_algorithm(partition_data, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, max_iteration, tau, budget, alpha, beta):
#     X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
#     y_pred_test=Model.predict_proba(x_test)
#     ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
#     best_parity = ini_parity       
#     #ini_parity=[ini_parity]
#     clusters = partition_data.copy()   
#     stat, stat_ex = [ini_parity], [ini_parity]
#     acc_ini=computeAccuracy(y_test, y_pred_test)
#     #acc_ini=[acc_ini]
#     acc_stat=[acc_ini]
#     acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
#     time_per_iteration_stat, iteration_time1 = [0], 0
#     cluster_count, train_shape = [], []
#     alpha=alpha
#     Ttrain_updated = copy.deepcopy(train_orig)
#     num_clusters = len(clusters)
#     k = max_iteration
#     r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
#     br, U = np.zeros(num_clusters), np.zeros(num_clusters)
    
#     M = copy.deepcopy(Model)
#     iteration=0
#     iteration_mono=0
#     remaining_budget=budget
#     while np.abs(best_parity)>tau and remaining_budget > 0:
#         start_time = time.time()
#         if iteration >= max_iteration-1:
#             break
#         if iteration_mono >= round(budget/mini_batch_size):
#             break
#         iteration +=1
#         #max_indices = np.where(U == np.max(U))[0]
#         #selected_cluster_indices = np.random.choice(max_indices)
#         selected_cluster_indices = np.argmax(U)
#         exhausted_clusters = set()
#         K = min(mini_batch_size, remaining_budget)
#         while len(clusters[selected_cluster_indices]) < K:
#             exhausted_clusters.add(selected_cluster_indices)
#             U[list(exhausted_clusters)] = -np.inf
#             selected_cluster_indices = np.argmax(U)
#             if selected_cluster_indices in exhausted_clusters:
#                 break
        
#         if len(clusters[selected_cluster_indices]) < K:
#             break

#         mini_batch_indices = np.arange(K)
#         mini_batch = clusters[selected_cluster_indices][:K]
#         Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

#         X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_combined, test_orig, target_col)
        
#         y_pred_test = M.predict_proba(x_test)
#         before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
#         accuracy_bef = computeAccuracy(y_test, y_pred_test)
        
#         model_up = copy.deepcopy(M)
#         model_up.fit(X_train_combined, y_train_combined)
        
#         y_pred_test = model_up.predict_proba(x_test)
#         after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
#         accuracy_1 = computeAccuracy(y_test, y_pred_test)
#         del_acc=accuracy_1-accuracy_bef
#         delta = -(np.abs(after) - np.abs(before))
#         if delta > 0 and np.abs(after) < np.abs(best_parity):                                                   
#             iteration_mono += 1
#             Ttrain_updated, M = Ttrain_combined, model_up
#             remaining_budget=remaining_budget-K
#             i_values_ex_stat.append(iteration_mono)
#             stat_ex.append(after)
#             acc_stat.append(accuracy_1)
#             train_shape.append(Ttrain_updated.shape)
#             cluster_count.append(selected_cluster_indices + 1)
#             best_parity = after
#             #print(f'These indices drwan from{mini_batch_indices} drawn from{selected_cluster_indices+1}')
        
        
#         for j in range(num_clusters):
#             br[j]=calculate_group_disparity(clusters[j], dataset_name)
#             normalized_distance=compute_normalized_distances(clusters)
#             if j==selected_cluster_indices:
#                 r[iteration, j] = (delta) /(1 + np.abs(br[j]))
#                 n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)                   
#             else:
#                 r[iteration, j] = ((delta) / ((1 + np.abs(br[j]))))+ beta*(del_acc)*(1+normalized_distance[selected_cluster_indices, j])
#                 n[iteration, j] = n[iteration - 1, j]
#             R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j]+1)
#             U[j] = R[iteration, j] + alpha * np.sqrt(2 * np.log(np.sum(n[iteration, :])+1) / ((n[iteration, j]) + 1))
        
#         cluster_count.append(selected_cluster_indices + 1)

#         cluster_df = clusters[selected_cluster_indices]
#         mini_batch_indices = list(mini_batch_indices)
#         cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
#         clusters[selected_cluster_indices] = cluster_df

#         i_values_stat.append(iteration)
#         stat.append(after)

#         iteration_time = time.time() - start_time
#         iteration_time1+=iteration_time
        

        
    
#     return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat, 
#             time_per_iteration_stat, cluster_count, iteration_time1)

