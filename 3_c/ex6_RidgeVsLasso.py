# [Demonstrate how the same dataset can lead to different weight values with Lasso vs. Ridge]
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

# 1. Create a small synthetic dataset
np.random.seed(42)

# Features (X) with 5 columns, some of which might be correlated
X = np.random.rand(100, 5) * 10
# True coefficients (some are zero for demonstration)
true_coefs = np.array([1.5, 0.0, -2.0, 0.0, 3.0])
# Generate target with some noise
y = X.dot(true_coefs) + np.random.normal(0, 2, size=100)

# 2. Split into training and test sets (to mimic a typical ML workflow)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit a Ridge regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_coefs = ridge.coef_
ridge_intercept = ridge.intercept_

# 4. Fit a Lasso regression model
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_coefs = lasso.coef_
lasso_intercept = lasso.intercept_

# 5. Compare the coefficients
print("True coefficients:", true_coefs)
print("\nRidge coefficients:", ridge_coefs)
print("Ridge intercept:", ridge_intercept)
print("\nLasso coefficients:", lasso_coefs)
print("Lasso intercept:", lasso_intercept)

# 6. Evaluate on test data (optional, to see performance)
ridge_score = ridge.score(X_test, y_test)
lasso_score = lasso.score(X_test, y_test)
print(f"\nRidge R^2 on test data: {ridge_score:.3f}")
print(f"Lasso R^2 on test data: {lasso_score:.3f}")

'''
🔍 Ce face codul:
Generează un set sintetic de date:

5 variabile independente (X), unele posibil corelate.

Coeficienții reali (true_coefs) sunt [1.5, 0.0, -2.0, 0.0, 3.0] → doar 3 din 5 variabile au un impact real.

Variabila dependentă y este creată cu o combinație liniară a lui X și puțin zgomot.

Separa datele în train/test (80% / 20%).

Antrenează două modele:

Ridge → Regresie L2: penalizează coeficienții mari, dar nu îi anulează.

Lasso → Regresie L1: penalizează coeficienții și îi poate face exact 0, deci face și selecție de variabile.

Compară coeficienții obținuți vs cei reali.

Evaluează performanța (R²) pe datele de test.

🧠 Diferențe între Lasso și Ridge:
Aspect	Ridge (L2)	Lasso (L1)
Penalizare	Proporțională cu suma pătratelor coeficienților	Proporțională cu suma valorilor absolute
Efect	Reduce coeficienții mari	Poate anula complet coeficienții
Selecție de variabile	❌ Nu elimină variabile	✅ Elimină variabile irelevante
Când e util?	Când toate variabilele contează puțin	Când doar unele sunt relevante

'''