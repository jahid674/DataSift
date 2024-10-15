import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt 
from kneed import KneeLocator
from DatasetExt import *
np.random.seed(42)

def compute_normalized_distances(clustered_data):
  
    num_clusters = len(clustered_data)
    d = np.zeros((num_clusters, num_clusters))
    reg_term=1e-3
    for i in range(num_clusters):
        for j in range(num_clusters):
            mean1 = np.mean(clustered_data[i], axis=0)   
            mean2 = np.mean(clustered_data[j], axis=0)
            distance = np.linalg.norm(mean1 - mean2) ** 2 + reg_term
            d[i, j] = distance
            
    # Normalize the distances
    min_distance = np.min(d)
    max_distance = np.max(d)
    normalized_distance = (d - min_distance) / (max_distance - min_distance)
    
    return normalized_distance

def find_neighbors(clusters, normalized_dist, dist):
    num_clusters = len(clusters)
    neighbors = {i: set() for i in range(num_clusters)}
    
    for i in range(num_clusters):
        for j in range(num_clusters):
                distance=normalized_dist[i,j]
                if distance <= dist:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
    
    return neighbors

def calculate_group_disparity(data_df, dataset_name):
    positive_outcome=1
    target_attribute, protected_attribute, protected_group, privileged_group= get_target_sensitive_attribute(dataset_name)
    priv_positive = data_df[(data_df[protected_attribute] == privileged_group) & (data_df[target_attribute] == positive_outcome)]
    prot_positive = data_df[(data_df[protected_attribute] == protected_group) & (data_df[target_attribute] == positive_outcome)]
    priv_positive_rate = len(priv_positive) / (len(data_df[data_df[protected_attribute] == privileged_group])+1)
    prot_positive_rate = len(prot_positive) / (len(data_df[data_df[protected_attribute] == protected_group])+1)
    group_disparity = priv_positive_rate - prot_positive_rate
    return group_disparity

def get_group_count(data, dataset_name):
    positive_outcome=1
    negative_outcome=0
    target_attribute, protected_attribute, protected_group, privileged_group= get_target_sensitive_attribute(dataset_name)
    privileged_positive = data[(data[protected_attribute] == privileged_group) & (data[target_attribute] == positive_outcome)]
    privileged_negative = data[(data[protected_attribute] == privileged_group) & (data[target_attribute] == negative_outcome)]
    protected_positive = data[(data[protected_attribute] == protected_group) & (data[target_attribute] == positive_outcome)]
    protected_negative = data[(data[protected_attribute] == protected_group) & (data[target_attribute] == negative_outcome)]
    privileged_data_full = data[data[protected_attribute] == privileged_group]
    protected_data_full = data[data[protected_attribute] == protected_group]   
    return privileged_positive, privileged_negative, protected_positive, protected_negative, privileged_data_full, protected_data_full

def smart_sample(data_df, total_size, privileged_pct, privileged_positive_pct, dataset_name):
    num_privileged = int(total_size * (privileged_pct / 100))
    num_privileged_positive = int(num_privileged * (privileged_positive_pct / 100))
    num_privileged_negative = num_privileged - num_privileged_positive
    num_protected = total_size - num_privileged
    privileged_positive, privileged_negative, protected_positive, protected_negative, privileged_data, protected_data=get_group_count(data_df, dataset_name)

    if len(privileged_positive) < num_privileged_positive or len(privileged_negative) < num_privileged_negative:
        raise ValueError("Not enough data in one or more required categories to perform the requested sampling.")

    sampled_privileged_positive = privileged_positive.sample(n=num_privileged_positive, random_state=42)
    sampled_privileged_negative = privileged_negative.sample(n=num_privileged_negative, random_state=42)
    sampled_protected = protected_data.sample(n=num_protected, random_state=42)
    sampled_df = pd.concat([sampled_privileged_positive, sampled_privileged_negative, sampled_protected])

    return sampled_df

def prepare_train_data(train_df, test_df, target_attribute):
    X_train = train_df.drop(columns=[target_attribute])
    y_train = train_df[target_attribute]
    X_test = test_df.drop(columns=[target_attribute])
    y_test = test_df[target_attribute]
    duplicates = 1
    make_duplicates = lambda x, d: pd.concat([x]*d, axis=0).reset_index(drop=True)
    X_train = make_duplicates(X_train, duplicates)
    X_test = make_duplicates(X_test, duplicates)
    y_train = make_duplicates(y_train, duplicates)
    y_test = make_duplicates(y_test, duplicates)
    X_train_orig = copy.deepcopy(X_train)
    X_test_orig = copy.deepcopy(X_test)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    return X_train, X_test, y_train, y_test, X_train_orig, X_test_orig

def get_minibatches(df, mini_batch_size):
    batches = []
    while len(df) > 0:
        batch = df[:mini_batch_size]
        batches.append(batch)
        df = df.drop(batch.index)
        if df.empty:
            break
    return batches

def find_optimal_gmm_components(data):
    n_components_range = range(1, 11)
    random_state = 42
    bic_scores = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
    kneedle_gmm = KneeLocator(list(n_components_range), bic_scores, curve="convex", direction="decreasing")
    optimal_num_components_gmm = kneedle_gmm.knee
    '''plt.plot(n_components_range, bic_scores, marker='o')
    plt.title('GMM Knee Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.axvline(optimal_num_components_gmm, color='r', linestyle='--', label='Optimal Number of Components')
    plt.show()'''

    return optimal_num_components_gmm

def gmm_clustering(data, optimal_num_components):
    random_state = 42
    gmm = GaussianMixture(n_components=optimal_num_components, random_state=random_state)
    gmm.fit(data)
    clusters_gmm = gmm.predict(data)

    clustered_data = []
    for i in range(optimal_num_components):
        cluster_i_indices = np.where(clusters_gmm == i)[0]
        cluster_i_data = data.iloc[cluster_i_indices, :]
        clustered_data.append(cluster_i_data)
        print(f"Cluster {i+1} shape: {cluster_i_data.shape}")

    return clustered_data

def sorted_influence_KNN(train_df, test_df, influences, k):
    def similarity_metric(test_point, train_data):
        distances = np.linalg.norm(train_data - test_point, axis=1)
        return distances

    def estimate_test_influence(test_data, train_data, influences, k):
        estimated_influences = []
        for test_point in test_data:
            similarities = similarity_metric(test_point, train_data)
            nearest_indices = np.argsort(similarities)[:k]
            nearest_influences = influences[nearest_indices]
            estimated_influence = np.mean(nearest_influences)
            estimated_influences.append(estimated_influence)
        estimated_influences = np.array(estimated_influences)
        return estimated_influences
    train_orig_array = train_df.to_numpy()
    test_data_array = test_df.to_numpy()
    test_influences = estimate_test_influence(test_data_array, train_orig_array, influences, k)
    test_data_with_influence = pd.DataFrame(test_data_array, columns=test_df.columns)
    test_data_with_influence['influence'] = test_influences
    sorted_cluster_data = test_data_with_influence.sort_values(by='influence', ascending=True).reset_index(drop=True)
    sorted_cluster_data_array = sorted_cluster_data.drop(columns=['influence'])
    sorted_influences = sorted_cluster_data['influence']
    return sorted_cluster_data_array, sorted_influences


def train_reg_model(data_df, influence_fair, target_column, degree=2, alpha=1.0, test_size=0.9, random_state=42):
    data_orig_inf = data_df.copy()
    data_orig_inf['Influence'] = influence_fair
    Target_inf = target_column
    train_orig_inf, test_orig_inf = train_test_split(data_orig_inf, test_size=test_size, random_state=random_state)

    X_train_inf, X_test_inf, y_train_inf, y_test_inf, X_train_orig_inf, y_train_orig_inf = prepare_train_data(
        train_orig_inf, test_orig_inf, Target_inf
    )

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_inf)
    X_test_poly = poly.transform(X_test_inf)

    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X_train_poly, y_train_inf)

    y_pred = model_ridge.predict(X_test_poly)

    r2_ridge = r2_score(y_test_inf, y_pred)
    print(f"Ridge R-squared (R²): {r2_ridge}")
    return model_ridge, X_test_poly, y_test_inf

def sorted_influences_reg(test_df, model):
    sc = StandardScaler()
    test_df_scale=sc.fit_transform(test_df)
    poly = PolynomialFeatures(degree=2)
    test_poly=poly.fit_transform(test_df_scale)

    estimate_test_influence= model.predict(test_poly)
    test_data_array = test_df.to_numpy()

    test_data_with_influence = pd.DataFrame(test_data_array, columns=test_df.columns)
    test_data_with_influence['influence'] = estimate_test_influence
    sorted_cluster_data = test_data_with_influence.sort_values(by='influence', ascending=False).reset_index(drop=True)
    sorted_cluster_data_array = sorted_cluster_data.drop(columns=['influence'])
    sorted_influences = sorted_cluster_data['influence']

    return sorted_cluster_data_array, sorted_influences

def generate_synthetic_data(datapool, n):
    synthetic_data = datapool.sample(n=n, replace=True, random_state=42)
    return synthetic_data
