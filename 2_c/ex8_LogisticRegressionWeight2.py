# [Example: Outline how a bank uses logistic regression to identify potential fraudulent transactions]
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Generate a synthetic imbalanced dataset
#    Weights ~ 99% legitimate, 1% fraudulent
X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.99, 0.01], # 99% legit, 1% frauda
    random_state=42
)

# Convert to a DataFrame for clarity (optional)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["is_fraudulent"] = y

# 2. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train Logistic Regression (with class weighting to handle imbalance)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. Predict on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)

# Classification report shows precision, recall, and F1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. (Optional) Print feature coefficients to see which features drive fraud detection
coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0]
})
print("\nFeature Coefficients (Logistic Regression):")
print(coefficients)

'''
Ce face codul, pas cu pas:
🧪 1. Generează un set de date dezechilibrat
2000 de tranzacții, cu doar 1% frauduloase (is_fraudulent = 1)

6 caracteristici (feature_0 până la feature_5)

Setul simulează realitatea: majoritatea tranzacțiilor sunt legitime

✂️ 2. Împarte datele în training și test
70% pentru antrenare

30% pentru testare

🧠 3. Antrenează un model Logistic Regression
Include: class_weight='balanced'

Această opțiune spune modelului să acorde mai multă importanță clasei minoritare (fraudă), compensând dezechilibrul.

📈 4. Face predicții pe setul de test
🧮 5. Evaluează modelul:
Confusion Matrix → cum s-au împărțit cazurile corecte/greșite.

Classification Report → include:

Precision: cât de sigur e modelul când spune "fraudă"

Recall: cât de bine prinde toate fraudele

F1-score: media armonică între precision și recall

📊 6. Afișează coeficienții modelului
Permite interpretarea: ce caracteristici contribuie la predicția fraudei.

Coeficient pozitiv → crește probabilitatea de fraudă

Coeficient negativ → scade probabilitatea'''