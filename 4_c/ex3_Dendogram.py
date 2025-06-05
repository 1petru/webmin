# [An example illustrating how to read a dendrogram for a dataset of images grouped by visual similarity]
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Generate synthetic feature vectors
#    For demonstration, imagine each row is a feature vector extracted from an image.
#    In a real project, replace this random data with actual image descriptors.
np.random.seed(42)
num_images = 10
num_features = 64  # e.g., a flattened 8x8 grayscale image
image_features = np.random.rand(num_images, num_features)

# 2. Perform hierarchical clustering (linkage)
#    "ward" attempts to minimize the variance within clusters;
#    other methods: "single", "complete", "average", etc.
Z = linkage(image_features, method='ward')

# 3. Plot the dendrogram
plt.figure(figsize=(8, 6))
dendrogram(Z, labels=[f"Img_{i}" for i in range(num_images)])
plt.title("Dendrogram of Synthetic Image Feature Vectors")
plt.xlabel("Images")
plt.ylabel("Distance (Ward linkage)")
plt.show()

# 4. How to interpret the dendrogram:
#    - Each leaf corresponds to an image.
#    - The branches show how clusters merge at increasing distances.
#    - A horizontal 'cut' at a certain distance can be used to decide the number of clusters.

'''
 Ce face mai exact:
Generare date sintetice:

num_images = 10: avem 10 "imagini".

Fiecare imagine este reprezentată de un vector de 64 de caracteristici (ex: un 8x8 patch de imagine grayscale aplatizat).

Clustering ierarhic:

linkage(image_features, method='ward'): calculează distanțele dintre vectori și construiește o ierarhie de clustere.

Metoda Ward minimizează variația internă când două clustere sunt unite (e bună pentru clustere compacte).

Plot dendrogramă:

Afișează vizual modul în care imaginile sunt grupate pe baza similarității vectorilor de caracteristici.

📊 Cum interpretezi dendrograma:
Frunzele (leaf nodes) = imaginile individuale (Img_0, Img_1, etc.).

Ramurile = cum sunt grupate imaginile între ele.

Înălțimea la care două ramuri se unesc = distanța/diferența între cele două clustere.

Poți "tăia" dendrograma orizontal, la o anumită înălțime, ca să obții un anumit număr de clustere.

🧩 Pe scurt:
Termen	Înseamnă...
linkage	Calculează pașii de fuzionare a clusterelor
ward	Minimizează variația în interiorul clusterelor
dendrogram	Diagramă care arată cum se formează ierarhia
Tăiere orizontală	Număr de clustere dorit (decupat vizual)'''