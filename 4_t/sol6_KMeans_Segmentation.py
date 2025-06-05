import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Generăm date sintetice pentru clienți
np.random.seed(42)
num_customers = 50

df_customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 15, num_customers),  # între 1 și 14 achiziții/lună
    'average_spent': np.random.randint(10, 500, num_customers),     # între 10 și 500 RON
    'loyalty_score': np.random.randint(1, 6, num_customers)         # scor de fidelitate între 1 și 5
})

print("=== Raw Customer Data (first 5 rows) ===")
print(df_customers.head(), "\n")

# 2. Scalăm datele
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_customers)

# 3. Aplicăm K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 4. Adăugăm etichetele în DataFrame
df_customers['Segment'] = labels

# 5. Analizăm fiecare segment
print("=== Segment Summary ===")
print(df_customers.groupby('Segment').mean())

# 6. Interpretare rapidă (opțională)
print("\nInterpretare posibilă:")
for segment in df_customers['Segment'].unique():
    group = df_customers[df_customers['Segment'] == segment]
    avg_freq = group['purchase_frequency'].mean()
    avg_spent = group['average_spent'].mean()
    avg_loyalty = group['loyalty_score'].mean()
    print(f"Segment {segment}:")
    print(f"  - Achiziții/lună: {avg_freq:.1f}")
    print(f"  - Cheltuieli medii: {avg_spent:.2f} RON")
    print(f"  - Scor fidelitate: {avg_loyalty:.2f}")
    print("")
