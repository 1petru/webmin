## Exercise 3 (10 minutes): DBSCAN Clustering
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we'll again simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate DBSCAN with chosen parameters
#    eps defines the neighborhood radius, min_samples is the minimum number of points
#    for a region to be considered dense.
dbscan = DBSCAN(eps=0.6, min_samples=5)

# 3. Fit the model to the data
dbscan.fit(df_scaled)

# 4. Extract cluster labels
labels = dbscan.labels_

# 5. Identify outliers (DBSCAN labels outliers as -1)
n_outliers = list(labels).count(-1)
print(f"Number of outliers: {n_outliers}")

# 6. (Optional) Add the labels to the DataFrame
df_scaled['Cluster'] = labels

# 7. Print the cluster label counts
print(df_scaled['Cluster'].value_counts())

# 8. Optional quick visualization (for 2D only)
plt.figure(figsize=(8, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 2], c=labels, cmap='plasma')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
#    Choose two features to plot, coloring by DBSCAN labels
