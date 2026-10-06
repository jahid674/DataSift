import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score
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


# -----------------------------
# UPDATED: find_optimal_components
# Adds method="birch" (scalable hierarchical) using silhouette on a sample
# -----------------------------
def find_optimal_components(
    data,
    method="gmm",
    k_range=range(2, 11),
    gmm_range=range(1, 11),
    random_state=42,
    # BIRCH knobs
    birch_threshold=0.5,
    birch_branching_factor=50,
    birch_sample_size=10000
):
    """
    Finds an 'optimal' number of components/clusters depending on method.

    - method="gmm": uses BIC + knee on n_components in gmm_range
    - method="kmeans": uses inertia + knee on k in k_range
    - method="birch": chooses k in k_range by silhouette on standardized features,
                      computed on a random sample for scalability.
    """
    method = method.lower().strip()

    if method == "gmm":
        scores = []
        x_vals = list(gmm_range)
        for n_components in x_vals:
            gmm = GaussianMixture(n_components=n_components, random_state=random_state)
            gmm.fit(data)
            scores.append(gmm.bic(data))
        kneedle = KneeLocator(x_vals, scores, curve="convex", direction="decreasing")
        return kneedle.knee

    elif method == "kmeans":
        inertias = []
        x_vals = list(k_range)
        for k in x_vals:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            km.fit(data)
            inertias.append(km.inertia_)
        kneedle = KneeLocator(x_vals, inertias, curve="convex", direction="decreasing")
        return kneedle.knee

    elif method == "birch":
        rng = np.random.RandomState(random_state)

        # Standardize first (important for BIRCH)
        X = StandardScaler().fit_transform(data.values)
        n = X.shape[0]
        sample_n = min(int(birch_sample_size), n)
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        Xs = X[sample_idx]

        best_k, best_s = None, -np.inf
        for k in list(k_range):
            model = Birch(
                threshold=float(birch_threshold),
                branching_factor=int(birch_branching_factor),
                n_clusters=int(k)
            )
            labels = model.fit_predict(Xs)

            # Need at least 2 clusters for silhouette
            if len(set(labels)) < 2:
                continue

            s = silhouette_score(Xs, labels)
            if s > best_s:
                best_s, best_k = s, int(k)

        return best_k

    else:
        raise ValueError("method must be one of: 'gmm', 'kmeans', 'birch'")


# -----------------------------
# UPDATED: clustering
# Adds method="birch" returning list of per-cluster DataFrames
# -----------------------------
def clustering(
    data,
    method="gmm",
    optimal_num_components=None,
    random_state=42,
    # BIRCH knobs
    birch_threshold=0.5,
    birch_branching_factor=50,
    scale_for_birch=True
):
    """
    Runs clustering and returns a list of per-cluster DataFrames.

    - method="gmm": uses optimal_num_components (required)
    - method="kmeans": uses optimal_num_components (required)
    - method="birch": uses optimal_num_components (required)
        * scalable hierarchical clustering
        * recommended: scale_for_birch=True
    """
    method = method.lower().strip()

    if method in ("gmm", "kmeans", "birch") and (optimal_num_components is None or optimal_num_components < 1):
        raise ValueError(f"For '{method}', provide optimal_num_components >= 1.")

    if method == "gmm":
        model = GaussianMixture(n_components=optimal_num_components, random_state=random_state)
        model.fit(data)
        labels = model.predict(data)

    elif method == "kmeans":
        model = KMeans(n_clusters=optimal_num_components, random_state=random_state, n_init="auto")
        labels = model.fit_predict(data)

    elif method == "birch":
        X = data.values
        if scale_for_birch:
            X = StandardScaler().fit_transform(X)

        model = Birch(
            threshold=float(birch_threshold),
            branching_factor=int(birch_branching_factor),
            n_clusters=int(optimal_num_components)
        )
        labels = model.fit_predict(X)

    else:
        raise ValueError("method must be one of: 'gmm', 'kmeans', 'birch'")

    clustered_data = []
    for lab in sorted(set(labels)):
        idx = np.where(labels == lab)[0]
        cluster_df = data.iloc[idx, :]
        clustered_data.append(cluster_df)
        print(f"Cluster {len(clustered_data)} shape: {cluster_df.shape}")

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
