import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generăm date „normale”
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))  # 200 puncte distribuite normal

# 2. Generăm câteva anomalii evidente
outliers = np.array([
    [100, 100],
    [10, 90],
    [90, 10],
    [120, 40],
    [40, 120]
])

# 3. Combinăm cele două seturi
X = np.vstack((normal_data, outliers))

# 4. Aplicăm DBSCAN pentru a detecta clustere și anomalii
dbscan = DBSCAN(eps=8, min_samples=5)
labels = dbscan.fit_predict(X)

# 5. Identificăm outlierii (eticheta -1 înseamnă „zgomot”)
n_outliers = list(labels).count(-1)
print(f"Număr de anomalii detectate: {n_outliers}")

# 6. Vizualizare
plt.figure(figsize=(8, 6))
colors = ['black' if label == -1 else f'C{label}' for label in labels]
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
plt.title('Anomaly Detection cu DBSCAN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 7. Raportare
n_total = len(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Total puncte: {n_total}")
print(f"Clustere găsite: {n_clusters}")
print(f"Anomalii detectate: {n_outliers}")
