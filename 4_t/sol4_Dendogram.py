import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Simulăm df_scaled
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 2. Aplicăm Agglomerative Clustering (K=3 clustere)
agglo = AgglomerativeClustering(n_clusters=3)
cluster_labels = agglo.fit_predict(df_scaled)

# 3. Adăugăm etichetele în DataFrame
df_scaled['Cluster'] = cluster_labels

# 4. Afișăm distribuția pe clustere
print("Distribuția pe clustere:")
print(df_scaled['Cluster'].value_counts())

# 5. Creăm matricea de linkage pentru dendrogramă
linked = linkage(df_scaled.drop(columns='Cluster'), method='ward')

# 6. Afișăm dendrograma
plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False)
plt.title("Dendrogramă - Agglomerative Clustering")
plt.xlabel("Puncte de date")
plt.ylabel("Distanță")
plt.show()
