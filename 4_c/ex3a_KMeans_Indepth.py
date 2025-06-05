import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    rand_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score
)

#########################
# 1. Generate Synthetic Data with True Labels
#########################
# Let's create a dataset of 150 samples, 3 centers, in 2D
X, y_true = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)

#########################
# 2. Cluster Using K-Means
#########################
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

#########################
# 3. Internal Metrics
#########################
# These metrics do NOT require true labels.
sil = silhouette_score(X, y_pred)
db = davies_bouldin_score(X, y_pred)
ch = calinski_harabasz_score(X, y_pred)

#########################
# 4. External Metrics
#########################
# These metrics compare predicted labels (y_pred) to the true labels (y_true).
rand_idx = rand_score(y_true, y_pred)
adj_rand_idx = adjusted_rand_score(y_true, y_pred)


# Purity calculation:
#  - For each cluster, find which true label appears most often
#  - Sum the max counts
#  - Divide by total number of samples
def purity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


purity = purity_score(y_true, y_pred)

# F-measure (F1-score) at the cluster level.
# We'll treat each cluster as a "predicted class" and compare to the true labels.
# We can compute a macro-averaged F1 to treat classes equally.
f_measure = f1_score(y_true, y_pred, average='macro')

#########################
# 5. Print the Results
#########################
print("=== Internal Metrics (no true labels needed) ===")
print(f"Silhouette Score:       {sil:.3f} (range: -1 to 1, higher is better)")
print(f"Davies-Bouldin Index:   {db:.3f} (lower is better)")
print(f"Calinski-Harabasz Index: {ch:.3f} (higher is better)")

print("\n=== External Metrics (compare y_pred to y_true) ===")
print(f"Rand Index:             {rand_idx:.3f} (range: 0 to 1, higher is better)")
print(f"Adjusted Rand Index:    {adj_rand_idx:.3f} (range: -1 to 1, higher is better)")
print(f"Purity:                 {purity:.3f} (range: 0 to 1, higher is better)")
print(f"F-Measure (F1, macro):  {f_measure:.3f} (range: 0 to 1, higher is better)")

'''
Acest cod face o evaluare completă a unui algoritm de clustering (K-Means) folosind atât metode interne (care nu necesită etichete reale), cât și metode externe (care compară predicțiile cu etichetele reale y_true). Iată ce face pas cu pas:

🔧 1. Generare date sintetice
python
Copy
Edit
X, y_true = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
Se creează un set de date cu 150 puncte, distribuite în jurul a 3 centre (clustere), fiecare cu 2 caracteristici (2D). y_true sunt etichetele reale — folosite doar pentru metricile externe.

📊 2. Aplicarea K-Means
python
Copy
Edit
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)
Se aplică algoritmul K-Means cu 3 clustere. y_pred sunt etichetele atribuite de model (fără să știe y_true).

📈 3. Metrici interne (nu folosesc y_true)
Evaluarea calității clustere-lor doar pe baza formei și densității lor:

silhouette_score: măsoară cât de bine este un punct în interiorul clusterului său vs. celelalte (valori între -1 și 1; mai mare e mai bine).

davies_bouldin_score: un raport între distanțele dintre clustere și lățimea lor (mai mic e mai bine).

calinski_harabasz_score: raport între dispersia inter-clustere și intra-cluster (mai mare e mai bine).

🧪 4. Metrici externe (folosesc y_true)
Compară grupările făcute de K-Means (y_pred) cu etichetele reale (y_true):

rand_score: procentul de decizii corecte la nivel de perechi de puncte (sunt sau nu în același cluster).

adjusted_rand_score: varianta ajustată a Rand Index, penalizează rezultatele obținute întâmplător.

purity_score: pentru fiecare cluster prezis, se ia clasa reală dominantă și se calculează cât de “pure” sunt clusterele.

f1_score (macro): media scorurilor F1 pentru fiecare clasă — adică un compromis între precizie și acoperire (recall).

🖨️ 5. Afișare rezultate'''
