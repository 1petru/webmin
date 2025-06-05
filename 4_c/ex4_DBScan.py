# [An example describing how unusual network traffic is flagged as an anomaly]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generate synthetic "normal" network traffic data
#    For example, each point has (avg_packet_size, connection_speed)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))  # 200 data points

# 2. Generate synthetic "unusual" or anomalous traffic data
#    These points are scattered more randomly and far from the normal data range
outliers = np.random.uniform(low=0, high=100, size=(10, 2))  # 10 outlier points

# 3. Combine normal data and outliers
X = np.vstack((normal_data, outliers))

# 4. Apply DBSCAN for anomaly detection
dbscan = DBSCAN(eps=3, min_samples=5)
labels = dbscan.fit_predict(X)

# 5. Identify outliers (labeled -1 by DBSCAN)
outlier_points = X[labels == -1]
normal_points = X[labels != -1]

# 6. Visualize the results
#    - Normal points are shown with colors according to their cluster label
#    - Anomalies (outliers) are plotted with red "x" markers
plt.scatter(normal_points[:, 0], normal_points[:, 1], c=labels[labels != -1])
plt.scatter(outlier_points[:, 0], outlier_points[:, 1], marker='x', s=100, color='red', label='Anomaly')
plt.title("Network Traffic Anomaly Detection with DBSCAN")
plt.xlabel("Feature 1 (e.g. Avg Packet Size)")
plt.ylabel("Feature 2 (e.g. Connection Speed)")
plt.legend()
plt.show()

# 7. Print summary of anomalies
print("Number of anomalies detected:", len(outlier_points))
print("Sample anomaly points:\n", outlier_points[:5])
'''
🧠 Ce face modelul DBSCAN aici?
📌 1. Datele "normale"
Se generează 200 de puncte (simulând conexiuni normale) cu o distribuție normală în jurul valorii 50.

Fiecare punct are două caracteristici (ex: dimensiunea medie a pachetelor, viteza conexiunii).

📌 2. Datele "anomale"
Se adaugă 10 puncte distribuite aleator (uniform) pe o zonă mult mai largă — simulează trafic neobișnuit sau posibil periculos (ex: atacuri, bug-uri, scanări de porturi).

📌 3. DBSCAN = Density-Based Spatial Clustering of Applications with Noise
Caută zone dense de puncte → le definește ca clustere.

Punctele care nu au suficienți vecini apropiați sunt considerate zgomot sau anomalii → primesc eticheta -1.

📌 4. Detecția de anomalii
Orice punct cu eticheta -1 este tratat ca anomalie.

În plot: punctele roșii cu x sunt cele detectate de DBSCAN ca fiind anormale.

📊 Interpretare rezultate
DBSCAN a găsit clustere de trafic „normal” și a etichetat punctele rare ca fiind anomalii.

print("Number of anomalies detected:", ...) îți arată câte puncte sunt considerate „suspecte”.

Poți vedea și care sunt primele 5 puncte etichetate ca anomalii.'''