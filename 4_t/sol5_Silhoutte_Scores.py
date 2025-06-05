import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Încărcăm și scalăm datele
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Aplicăm cele 3 metode de clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Obținem etichetele de cluster
kmeans_labels = kmeans.labels_
dbscan_labels = dbscan.labels_
agg_labels = agg.labels_


# 4. Calculăm scorul Silhouette pentru fiecare (doar dacă avem >1 cluster)
def get_silhouette(X, labels, method_name):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        print(f"{method_name}: Prea puține clustere pentru a calcula silhouette score.")
        return None
    else:
        score = silhouette_score(X, labels)
        print(f"{method_name}: Silhouette Score = {score:.4f}")
        return score


# 5. Afișăm scorurile
get_silhouette(X_scaled, kmeans_labels, "KMeans")
get_silhouette(X_scaled, dbscan_labels, "DBSCAN")
get_silhouette(X_scaled, agg_labels, "Agglomerative Clustering")
