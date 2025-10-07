import numpy as np
import pandas as pd
import time
import copy
from sklearn.preprocessing import StandardScaler
from metrics import *
from Misc import *


def mab_algorithm(partition_data, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, max_iteration, tau, budget, alpha, beta):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity       
    clusters = partition_data.copy()   
    stat, stat_ex = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_stat=[acc_ini]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha_=alpha
    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)
    remaining_budget=budget
    
    M = copy.deepcopy(Model)
    iteration=0
    iteration_mono=0
    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration-1:
            break
        if iteration_mono >= round(budget/mini_batch_size):
            break
        #max_indices = np.where(U == np.max(U))[0]
        #selected_cluster_indices = np.random.choice(max_indices)
        selected_cluster_indices = np.argmax(U)
        exhausted_clusters = set()
        K = min(mini_batch_size, remaining_budget)
        while len(clusters[selected_cluster_indices]) < K:
            exhausted_clusters.add(selected_cluster_indices)
            U[list(exhausted_clusters)] = -np.inf
            selected_cluster_indices = np.argmax(U)
            if selected_cluster_indices in exhausted_clusters:
                break
        #print('Selected Cluster', selected_cluster_indices)
        if len(clusters[selected_cluster_indices]) < K:
            break

        iteration +=1
        mini_batch_indices = np.random.choice(len(clusters[selected_cluster_indices]), size=K, replace=False)
        mini_batch = clusters[selected_cluster_indices].iloc[mini_batch_indices]
        Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_combined, test_orig, target_col)
        
        y_pred_test = M.predict_proba(x_test)
        before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef = computeAccuracy(y_test, y_pred_test)
        
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        
        y_pred_test = model_up.predict_proba(x_test)
        after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        del_acc=accuracy_1-accuracy_bef
        del_perce=((after-before)/before)*100
        delta = -(np.abs(after) - np.abs(before))
        if delta > 0 and np.abs(after) < np.abs(best_parity):
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget=remaining_budget-K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after)
            acc_stat.append(accuracy_1)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            best_parity = after
            #print(Ttrain_updated.shape)
            #print(f'These indices drwan from{from{selected_cluster_indices+1}')
        
        for j in range(num_clusters):
            br[j]=calculate_group_disparity(clusters[j], dataset_name)
            normalized_distance=compute_normalized_distances(clusters)
            if j==selected_cluster_indices:
                r[iteration, j] = ((delta)) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)                   
            else:
                r[iteration, j] = ((delta) / ((1 + np.abs(br[j]))))+ (beta*(del_acc)*(1+normalized_distance[selected_cluster_indices, j]))
                n[iteration, j] = n[iteration - 1, j]
        
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j]+1)
            U[j] = R[iteration, j] + alpha_ * np.sqrt(2 * np.log(np.sum(n[iteration, :])+1) / ((n[iteration, j]) + 1))

        #print('r', r[iteration])
        #print('n', R[iteration])
        #print('R', R[iteration])
        #print('U shape', U.shape)
        #print('U', U)
        cluster_count.append(selected_cluster_indices + 1)
        cluster_df = clusters[selected_cluster_indices]
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters[selected_cluster_indices] = cluster_df
        iteration_time = time.time() - start_time
        i_values_stat.append(iteration)
        stat.append(after)
        iteration_time1 += iteration_time
   
    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat, 
            time_per_iteration_stat, cluster_count, iteration_time1)

def mab_algorithm_base(partition_data, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, max_iteration, tau, budget, alpha):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity       
    clusters = partition_data.copy()   
    stat, stat_ex = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_stat=[acc_ini]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha_=alpha
    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)
    remaining_budget=budget
    
    M = copy.deepcopy(Model)
    iteration=0
    iteration_mono=0
    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration-1:
            break
        if iteration_mono >= round(budget/mini_batch_size):
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

        iteration +=1
        mini_batch_indices = np.random.choice(len(clusters[selected_cluster_indices]), size=K, replace=False)
        mini_batch = clusters[selected_cluster_indices].iloc[mini_batch_indices]
        Ttrain_combined = pd.concat([Ttrain_updated, mini_batch], axis=0).reset_index(drop=True)

        X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_combined, test_orig, target_col)
        
        y_pred_test = M.predict_proba(x_test)
        before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef = computeAccuracy(y_test, y_pred_test)
        
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        
        y_pred_test = model_up.predict_proba(x_test)
        after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        
        delta = np.abs(after) - np.abs(before)
        del_acc=accuracy_1-accuracy_bef
        if delta < 0 and np.abs(after) < np.abs(best_parity):
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget=remaining_budget-K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after)
            acc_stat.append(accuracy_1)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            best_parity = after
        
        for j in range(num_clusters):
            br[j]=calculate_group_disparity(clusters[j], dataset_name)
            normalized_distance=compute_normalized_distances(clusters)
            if j==selected_cluster_indices:
                r[iteration, j] = -(delta) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)                   
            else:
                r[iteration, j] = (-(delta)) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j]+1)
            U[j] = R[iteration, j] + alpha_ * np.sqrt(2 * np.log(np.sum(n[iteration, :])+1) / ((n[iteration, j]) + 1))
        cluster_count.append(selected_cluster_indices + 1)
        cluster_df = clusters[selected_cluster_indices]
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters[selected_cluster_indices] = cluster_df
        iteration_time = time.time() - start_time
        i_values_stat.append(iteration)
        stat.append(after)
        iteration_time1 += iteration_time
   
    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat, 
            time_per_iteration_stat, cluster_count, iteration_time1)


def mab_algorithm_dist(partition_data, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, max_iteration, tau, budget, Euclid_normalized_d, alpha):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity       
    clusters = partition_data.copy()   
    stat, stat_ex = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_stat=[acc_ini]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha=alpha
    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)
    
    M = copy.deepcopy(Model)
    iteration=0
    iteration_mono=0
    remaining_budget=budget
    before=ini_parity
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration-1:
            break
        if iteration_mono >= round(budget/mini_batch_size):
            break
        iteration +=1
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

        X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_combined, test_orig, target_col)
        
        y_pred_test = M.predict_proba(x_test)
        #before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef = computeAccuracy(y_test, y_pred_test)
        
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        
        y_pred_test = model_up.predict_proba(x_test)
        after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        
        delta = np.abs(after) - np.abs(before)
        del_acc=accuracy_1-accuracy_bef
        before=after
        if delta < 0:# and np.abs(after) < np.abs(best_parity):
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget=remaining_budget-K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after)
            acc_stat.append(accuracy_1)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            cluster_count.append(selected_cluster_indices + 1)
            cluster_df = clusters[selected_cluster_indices]
            mini_batch_indices = list(mini_batch_indices)
            cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
            clusters[selected_cluster_indices] = cluster_df
            best_parity = after
        dist=0.001
        for j in range(num_clusters):
            n_dist = Euclid_normalized_d[j, selected_cluster_indices]
            if j in find_neighbors(clusters,Euclid_normalized_d, dist)[selected_cluster_indices]:
                r[iteration, j] = delta * (1 - n_dist / dist)
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)
                R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)                  
            else:
                r[iteration, j] = 0
                n[iteration, j] = n[iteration - 1, j]
                R[iteration, j] = R[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j]+1)
            U[j] = R[iteration, j] + alpha * np.sqrt(2 * np.log(np.sum(n[iteration, :])+1) / ((n[iteration, j]) + 1))
        

        iteration_time = time.time() - start_time
        time_per_iteration_stat.append(time_per_iteration_stat[-1] + iteration_time)
        i_values_stat.append(iteration)
        stat.append(after)
        iteration_time1+=iteration_time
   
    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat, 
            time_per_iteration_stat, cluster_count, iteration_time1)


def mab_inf_algorithm(partition_data, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, max_iteration, tau, budget, alpha, beta):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity       
    #ini_parity=[ini_parity]
    clusters = partition_data.copy()   
    stat, stat_ex = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    #acc_ini=[acc_ini]
    acc_stat=[acc_ini]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha=alpha
    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)
    
    M = copy.deepcopy(Model)
    iteration=0
    iteration_mono=0
    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration-1:
            break
        if iteration_mono >= round(budget/mini_batch_size):
            break
        iteration +=1
        #max_indices = np.where(U == np.max(U))[0]
        #selected_cluster_indices = np.random.choice(max_indices)
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

        X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_combined, test_orig, target_col)
        
        y_pred_test = M.predict_proba(x_test)
        before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef = computeAccuracy(y_test, y_pred_test)
        
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        
        y_pred_test = model_up.predict_proba(x_test)
        after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        del_acc=accuracy_1-accuracy_bef
        delta = -(np.abs(after) - np.abs(before))
        if delta > 0 and np.abs(after) < np.abs(best_parity):                                                   
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget=remaining_budget-K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after)
            acc_stat.append(accuracy_1)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            best_parity = after
            #print(f'These indices drwan from{mini_batch_indices} drawn from{selected_cluster_indices+1}')
        
        
        for j in range(num_clusters):
            br[j]=calculate_group_disparity(clusters[j], dataset_name)
            normalized_distance=compute_normalized_distances(clusters)
            if j==selected_cluster_indices:
                r[iteration, j] = (delta) /(1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)                   
            else:
                r[iteration, j] = ((delta) / ((1 + np.abs(br[j]))))+ beta*(del_acc)*(1+normalized_distance[selected_cluster_indices, j])
                n[iteration, j] = n[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j]+1)
            U[j] = R[iteration, j] + alpha * np.sqrt(2 * np.log(np.sum(n[iteration, :])+1) / ((n[iteration, j]) + 1))
        
        cluster_count.append(selected_cluster_indices + 1)

        cluster_df = clusters[selected_cluster_indices]
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters[selected_cluster_indices] = cluster_df

        i_values_stat.append(iteration)
        stat.append(after)

        iteration_time = time.time() - start_time
        iteration_time1+=iteration_time
        

        
    
    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat, 
            time_per_iteration_stat, cluster_count, iteration_time1)

def mab_inf_algorithm_base(partition_data, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, max_iteration, tau, budget, alpha):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity       
    #ini_parity=[ini_parity]
    clusters = partition_data.copy()   
    stat, stat_ex = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    #acc_ini=[acc_ini]
    acc_stat=[acc_ini]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha=alpha
    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)
    
    M = copy.deepcopy(Model)
    iteration=0
    iteration_mono=0
    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration-1:
            break
        if iteration_mono >= round(budget/mini_batch_size):
            break
        iteration +=1
        #max_indices = np.where(U == np.max(U))[0]
        #selected_cluster_indices = np.random.choice(max_indices)
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

        X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_combined, test_orig, target_col)
        
        y_pred_test = M.predict_proba(x_test)
        before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef = computeAccuracy(y_test, y_pred_test)
        
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        
        y_pred_test = model_up.predict_proba(x_test)
        after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        del_acc=accuracy_1-accuracy_bef
        delta = -(np.abs(after) - np.abs(before))
        if delta > 0 and np.abs(after) < np.abs(best_parity):                                                   
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget=remaining_budget-K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after)
            acc_stat.append(accuracy_1)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            best_parity = after
            #print(f'These indices drwan from{mini_batch_indices} drawn from{selected_cluster_indices+1}')
        
        
        for j in range(num_clusters):
            br[j]=calculate_group_disparity(clusters[j], dataset_name)
            normalized_distance=compute_normalized_distances(clusters)
            if j==selected_cluster_indices:
                r[iteration, j] = -(delta) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)                   
            else:
                r[iteration, j] = (-(delta)) / (1 + np.abs(br[j]))
                n[iteration, j] = n[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j]+1)
            U[j] = R[iteration, j] + 0.1 * np.sqrt(2 * np.log(np.sum(n[iteration, :])+1) / ((n[iteration, j]) + 1))
        
        cluster_count.append(selected_cluster_indices + 1)

        cluster_df = clusters[selected_cluster_indices]
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters[selected_cluster_indices] = cluster_df

        i_values_stat.append(iteration)
        stat.append(after)

        iteration_time = time.time() - start_time
        iteration_time1+=iteration_time
        

        
    
    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat, 
            time_per_iteration_stat, cluster_count, iteration_time1)


def random_algorithm(clusters, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, tau, budget):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity        
    stat_ran, stat_ex_ran = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_ran=[acc_ini]
    acc_ex, i_values_ran, i_values_ex_ran = [], [0], [0]
    time_per_iteration_ran, iteration_time1 = [0], 0

    Ttrain_updated = copy.deepcopy(train_orig) 
    cumulative_time = 0
    train_shape=[]
    time_per_iteration = [0]
    time_per_iteration_ran = [0]

    M = copy.deepcopy(Model)
    iteration=0

    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration>= round(budget/mini_batch_size):
            break
        iteration +=1
        mini_batch_indices = np.random.choice(len(clusters), size=K, replace=False)
        mini_batch = clusters.iloc[mini_batch_indices]
        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0)
        Ttrain_updated_combined = Ttrain_updated_combined.reset_index(drop=True)
        X_train_combined, X_test_combined, y_train_combined, y_test_combined, X_train_orig_combined, X_test_original=prepare_train_data(Ttrain_updated_combined, test_orig, target_col)
        y_pred_test=M.predict_proba(x_test)
        before= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef =computeAccuracy(y_test, y_pred_test)
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        y_pred_test=model_up.predict_proba(x_test)
        after= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        delta = np.abs(after) - np.abs(before)
        if delta<0 or delta>0:
            remaining_budget=remaining_budget-K
            acc_ex.append(accuracy_1)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            i_values_ex_ran.append(iteration)
            stat_ex_ran.append(after)
            train_shape.append(Ttrain_updated.shape)
        
        if after > best_parity:
            best_parity = after
            iteration_no = iteration
        cluster_df = clusters
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters = cluster_df
        end_time = time.time()
        
        i_values_ran.append(iteration)
        stat_ran.append(after)
        acc_ran.append(accuracy_1)
        iteration_time = end_time - start_time
        iteration_time1+=iteration_time

    return Ttrain_updated, i_values_ex_ran, i_values_ran, stat_ran, stat_ex_ran, acc_ran, iteration_time1

def entropy_based_algorithm(clusters, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, tau, budget):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity        
    stat_ent, stat_ex_ent = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_ent=[acc_ini]
    acc_ex, i_values_ent, i_values_ex_ent = [], [0], [0]
    time_per_iteration_ent, iteration_time1 = [0], 0

    Ttrain_updated = copy.deepcopy(train_orig) 
    cumulative_time = 0
    train_shape=[]
    time_per_iteration = [0]
    time_per_iteration_ent = [0]

    M = copy.deepcopy(Model)
    iteration=0

    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration>= round(budget/mini_batch_size):
            break
        iteration +=1
        y_pred_clusters = M.predict_proba(clusters.drop([target_col], axis=1).values)
        p = y_pred_clusters
        entropy = - (p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12))
        mini_batch_indices = np.argsort(entropy)[-K:]
        mini_batch = clusters.iloc[mini_batch_indices]   
        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0)
        Ttrain_updated_combined = Ttrain_updated_combined.reset_index(drop=True)
        X_train_combined, X_test_combined, y_train_combined, y_test_combined, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_updated_combined, test_orig, target_col)
        y_pred_test = M.predict_proba(x_test)
        before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef = computeAccuracy(y_test, y_pred_test)
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        y_pred_test = model_up.predict_proba(x_test)
        after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        delta = np.abs(after) - np.abs(before)
        
        #print(f'The statistical parity after {iteration}th iteration is {after:.4f}')

        if delta<0 or delta>0:
            remaining_budget=remaining_budget-K
            i_values_ex_ent.append(iteration)
            stat_ex_ent.append(after)
            acc_ex.append(accuracy_1)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            train_shape.append(Ttrain_updated.shape)
        if after > best_parity:
            best_parity = after
            iteration_no = iteration
        cluster_df = clusters
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters = cluster_df
        

        i_values_ent.append(iteration)
        stat_ent.append(after)
        acc_ent.append(accuracy_1)
        end_time = time.time()
        iteration_time = end_time - start_time
        iteration_time1+=iteration_time
    return Ttrain_updated, i_values_ex_ent, i_values_ent, stat_ent, stat_ex_ent, acc_ent, iteration_time1

def inf_algorithm(clusters, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, tau, budget):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity        
    stat_ran, stat_ex_ran = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_ran=[acc_ini]
    acc_ex, i_values_ran, i_values_ex_ran = [], [0], [0]
    time_per_iteration_ran, iteration_time1 = [0], 0

    Ttrain_updated = copy.deepcopy(train_orig) 
    cumulative_time = 0
    train_shape=[]
    time_per_iteration = [0]
    time_per_iteration_ran = [0]

    M = copy.deepcopy(Model)
    iteration=0

    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration>= round(budget/mini_batch_size):
            break
        iteration +=1
        mini_batch_indices = np.arange(K)
        mini_batch = clusters[:K]
        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0)
        Ttrain_updated_combined = Ttrain_updated_combined.reset_index(drop=True)
        X_train_combined, X_test_combined, y_train_combined, y_test_combined, X_train_orig_combined, X_test_original=prepare_train_data(Ttrain_updated_combined, test_orig, target_col)
        y_pred_test=M.predict_proba(x_test)
        before= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef =computeAccuracy(y_test, y_pred_test)
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        y_pred_test=model_up.predict_proba(x_test)
        after= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        delta = np.abs(after) - np.abs(before)
        if delta<0 or delta>0:
            remaining_budget=remaining_budget-K
            i_values_ex_ran.append(iteration)
            stat_ex_ran.append(after)
            acc_ex.append(accuracy_1)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            train_shape.append(Ttrain_updated.shape)
        if after > best_parity:
            best_parity = after
            iteration_no = iteration
        cluster_df = clusters
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters = cluster_df
        
    
        i_values_ran.append(iteration)
        stat_ran.append(after)
        acc_ran.append(accuracy_1)
        end_time = time.time()
        iteration_time = end_time - start_time
        iteration_time1+=iteration_time
    return Ttrain_updated, i_values_ex_ran, i_values_ran, stat_ran, stat_ex_ran, acc_ran, iteration_time1

def mab_algorithm_acc(partition_data, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, max_iteration, tau, budget, Euclid_normalized_d, alpha):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity       
    clusters = partition_data.copy()   
    stat, stat_ex = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_stat=[acc_ini]
    acc_ex, i_values_stat, i_values_ex_stat = [], [0], [0]
    time_per_iteration_stat, iteration_time1 = [0], 0
    cluster_count, train_shape = [], []
    alpha=alpha
    Ttrain_updated = copy.deepcopy(train_orig)
    num_clusters = len(clusters)
    k = max_iteration
    r, R, n = np.zeros((k, num_clusters)), np.zeros((k, num_clusters)), np.zeros((k, num_clusters))
    br, U = np.zeros(num_clusters), np.zeros(num_clusters)
    
    M = copy.deepcopy(Model)
    iteration=0
    iteration_mono=0
    remaining_budget=budget
    before=ini_parity
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        if iteration >= max_iteration-1:
            break
        if iteration_mono >= round(budget/mini_batch_size):
            break
        iteration +=1
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

        X_train_combined, _, y_train_combined, _, X_train_orig_combined, X_test_original = prepare_train_data(Ttrain_combined, test_orig, target_col)
        y_pred_test = M.predict_proba(x_test)
        #before = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef = computeAccuracy(y_test, y_pred_test)
        
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        
        y_pred_test = model_up.predict_proba(x_test)
        after = computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        
        delta = np.abs(after) - np.abs(before)
        acc_del=accuracy_1-accuracy_bef
        before=after
        if acc_del > 0: # and np.abs(after) < np.abs(best_parity):
            iteration_mono += 1
            Ttrain_updated, M = Ttrain_combined, model_up
            remaining_budget=remaining_budget-K
            i_values_ex_stat.append(iteration_mono)
            stat_ex.append(after)
            acc_stat.append(accuracy_1)
            train_shape.append(Ttrain_updated.shape)
            cluster_count.append(selected_cluster_indices + 1)
            cluster_count.append(selected_cluster_indices + 1)
            cluster_df = clusters[selected_cluster_indices]
            mini_batch_indices = list(mini_batch_indices)
            cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
            clusters[selected_cluster_indices] = cluster_df
            best_parity = after
        dist=0.001
        for j in range(num_clusters):
            n_dist = Euclid_normalized_d[j, selected_cluster_indices]
            if j in find_neighbors(clusters,Euclid_normalized_d, dist)[selected_cluster_indices]:
                r[iteration, j] = delta * (1 - n_dist / dist)
                n[iteration, j] = n[iteration - 1, j] + (r[iteration, j] > 0).astype(int)
                R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j] + 1)                  
            else:
                r[iteration, j] = 0
                n[iteration, j] = n[iteration - 1, j]
                R[iteration, j] = R[iteration - 1, j]
            R[iteration, j] = np.sum(r[:, j]) / (n[iteration, j]+1)
            U[j] = R[iteration, j] + alpha * np.sqrt(2 * np.log(np.sum(n[iteration, :])+1) / ((n[iteration, j]) + 1))
        

        iteration_time = time.time() - start_time
        time_per_iteration_stat.append(time_per_iteration_stat[-1] + iteration_time)
        i_values_stat.append(iteration)
        stat.append(after)
        iteration_time1+=iteration_time
   
    return (Ttrain_updated, i_values_stat, i_values_ex_stat, stat_ex, stat, acc_stat, 
            time_per_iteration_stat, cluster_count, iteration_time1)

def random_algorithm_acc(clusters, dataset_name, train_orig, test_orig, Model, target_col, mini_batch_size, tau, budget):
    X_train, x_test, y_train_combined, y_test, X_train_orig, X_test_original = prepare_train_data(train_orig, test_orig, target_col)
    y_pred_test=Model.predict_proba(x_test)
    ini_parity= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
    best_parity = ini_parity        
    stat_ran, stat_ex_ran = [ini_parity], [ini_parity]
    acc_ini=computeAccuracy(y_test, y_pred_test)
    acc_ran=[acc_ini]
    acc_ex, i_values_ran, i_values_ex_ran = [], [0], [0]
    time_per_iteration_ran, iteration_time1 = [0], 0

    Ttrain_updated = copy.deepcopy(train_orig) 
    cumulative_time = 0
    train_shape=[]
    time_per_iteration = [0]
    time_per_iteration_ran = [0]

    M = copy.deepcopy(Model)
    iteration=0

    remaining_budget=budget
    while np.abs(best_parity)>tau and remaining_budget > 0:
        start_time = time.time()
        K = min(mini_batch_size, remaining_budget)
        if len(clusters) < K:
            break
        if iteration>= round(budget/mini_batch_size):
            break
        iteration +=1
        mini_batch_indices = np.random.choice(len(clusters), size=K, replace=False)
        mini_batch = clusters.iloc[mini_batch_indices]
        Ttrain_updated_combined = pd.concat([Ttrain_updated, mini_batch], axis=0)
        Ttrain_updated_combined = Ttrain_updated_combined.reset_index(drop=True)
        X_train_combined, X_test_combined, y_train_combined, y_test_combined, X_train_orig_combined, X_test_original=prepare_train_data(Ttrain_updated_combined, test_orig, target_col)
        y_pred_test=M.predict_proba(x_test)
        before= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_bef =computeAccuracy(y_test, y_pred_test)
        model_up = copy.deepcopy(M)
        model_up.fit(X_train_combined, y_train_combined)
        y_pred_test=model_up.predict_proba(x_test)
        after= computeFairness(y_pred_test, X_test_original, y_test, 0, dataset_name)
        accuracy_1 = computeAccuracy(y_test, y_pred_test)
        delta = np.abs(after) - np.abs(before)
        del_acc=accuracy_1-accuracy_bef
        if delta<0 or delta>0:
            remaining_budget=remaining_budget-K
            acc_ex.append(accuracy_1)
            Ttrain_updated = Ttrain_updated_combined
            M = model_up
            i_values_ex_ran.append(iteration)
            stat_ex_ran.append(after)
            train_shape.append(Ttrain_updated.shape)
        if after > best_parity:
            best_parity = after
            iteration_no = iteration
        cluster_df = clusters
        mini_batch_indices = list(mini_batch_indices)
        cluster_df = cluster_df.drop(cluster_df.index[mini_batch_indices])
        clusters = cluster_df
        end_time = time.time()
        
        i_values_ran.append(iteration)
        stat_ran.append(after)
        acc_ran.append(accuracy_1)
        iteration_time = end_time - start_time
        iteration_time1+=iteration_time

    return Ttrain_updated, i_values_ex_ran, i_values_ran, stat_ran, stat_ex_ran, acc_ran, iteration_time1
