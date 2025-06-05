import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# 1. Încărcăm și scalăm datele
iris = load_iris()
X = iris.data
y = iris.target  # Referință, dar nu o folosim în clustering

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Aplicăm cele 3 metode de clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Reducem datele la 2 dimensiuni pentru vizualizare
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4. Combinăm rezultatele într-un DataFrame
df_final = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_final['KMeans'] = kmeans.labels_
df_final['DBSCAN'] = dbscan.labels_
df_final['Agglo'] = agg.labels_

# 5. Afișăm 3 grafice unul lângă altul cu rezultatele
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# KMeans
axs[0].scatter(df_final['PC1'], df_final['PC2'], c=df_final['KMeans'], cmap='viridis', edgecolor='k')
axs[0].set_title('KMeans Clustering')

# DBSCAN
axs[1].scatter(df_final['PC1'], df_final['PC2'], c=df_final['DBSCAN'], cmap='plasma', edgecolor='k')
axs[1].set_title('DBSCAN Clustering')

# Agglomerative
axs[2].scatter(df_final['PC1'], df_final['PC2'], c=df_final['Agglo'], cmap='cool', edgecolor='k')
axs[2].set_title('Agglomerative Clustering')

for ax in axs:
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

plt.tight_layout()
plt.show()

# 6. Raport sumar: distribuția pe clustere
print("=== Distribuția punctelor în fiecare cluster ===")
print("KMeans:\n", pd.Series(kmeans.labels_).value_counts(), "\n")
print("DBSCAN:\n", pd.Series(dbscan.labels_).value_counts(), "\n")
print("Agglomerative:\n", pd.Series(agg.labels_).value_counts())
