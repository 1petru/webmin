# [Dealing with class imbalance in fraud detection datasets.]
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Step 1: Create a sample imbalanced dataset
np.random.seed(42)
data = {
    "TransactionID": range(1, 21),
    "Amount": np.random.randint(10, 1000, 20),
    "IsFraud": [0] * 17 + [1] * 3  # Imbalanced: 17 non-fraud, 3 fraud cases
}

df = pd.DataFrame(data)

# Step 2: Separate features and target
X = df[["Amount"]]  # Features
y = df["IsFraud"]   # Target variable

print("Original Class Distribution:", Counter(y))

# Step 3: Apply Undersampling (reduce majority class)
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)

print("Class Distribution After Undersampling:", Counter(y_under))

# Step 4: Apply Oversampling (SMOTE with reduced n_neighbors)
smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=1)  # Reduce k_neighbors
X_smote, y_smote = smote.fit_resample(X, y)

print("Class Distribution After Oversampling (SMOTE):", Counter(y_smote))
'''
📊 Distribuția originală
python
Copy
Edit
Counter({0: 17, 1: 3})
17 tranzacții legitime (0)

3 fraude (1)

Situație clasică de desechilibru sever: doar 15% din date sunt fraude → un model nativ ar putea învăța să zică mereu "nu e fraudă" și tot ar avea ~85% acuratețe... dar ar fi complet inutil în practică.

🔻 După Random Undersampling
python
Copy
Edit
Counter({0: 6, 1: 3})
Clasa majoritară (0) a fost redusă la 6 exemple.

Clasa minoritară (1) a rămas cu 3 exemple.

Obții un raport 2:1, conform sampling_strategy=0.5.

✅ Avantaj: modelul va învăța să acorde atenție și clasei minoritare.
⚠️ Atenție: ai pierdut 11 exemple legitime (6 din 17 păstrate) → risc să pierzi informații utile.

🔺 După SMOTE Oversampling
python
Copy
Edit
Counter({0: 17, 1: 13})
Clasa 1 (fraudă) a fost mărită artificial de la 3 → 13 prin generare de puncte sintetice.

Clasa 0 (legitim) a rămas neschimbată la 17.

Aproape echilibru: raport de 13:17 (76% în loc de 80% exact din cauza rotunjirii).

✅ Avantaj: ai păstrat toate datele originale și ai crescut puterea de învățare pentru fraude.
⚠️ Atenție: dacă datele originale de fraudă sunt zgomotoase sau prea puține, SMOTE poate introduce artefacte irelevante (de aceea ai folosit k_neighbors=1 – ceea ce e ok pentru seturi mici).
'''