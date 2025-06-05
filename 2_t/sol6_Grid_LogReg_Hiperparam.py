import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Define the parameter grid for Logistic Regression
#    - 'C' is the inverse of regularization strength (smaller => stronger regularization)
#    - 'penalty' controls L1 vs. L2 regularization
#    - 'solver' must support the selected penalty; 'saga' works for both l1 and l2.
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['saga']
}

# 4. Set up the GridSearchCV with 5-fold cross-validation
#    n_jobs=-1 uses all CPU cores to speed up the search
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# 5. Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# 6. Retrieve the best parameters and the corresponding score
print("Best Parameters found by grid search:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.3f}")

# 7. Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Model: {test_accuracy:.3f}")


'''
🔍 Ce face codul:
1. Încarcă setul de date:
Setul de date wine conține caracteristici chimice ale unor vinuri și tipul lor (3 clase: 0, 1, 2).

2. Împarte datele în train/test:
80% pentru antrenare, 20% pentru testare.

3. Definește o grilă de căutare (param_grid):
C: [0.01, 0.1, 1, 10] → controlează cât de puternică este regularizarea. Valori mici = regularizare mai puternică.

penalty: l1 sau l2 → tipul de regularizare (L1 = Lasso, L2 = Ridge).

solver: saga → este unul dintre puținii algoritmi care suportă și l1, și l2.

4. Caută combinația optimă cu GridSearchCV:
5-fold cross-validation = antrenează modelul de 5 ori pe subseturi diferite.

n_jobs=-1 = folosește toate nucleele procesorului pentru viteză.

5. Antrenează modelul cu toate combinațiile și alege cea mai bună pe baza acurateții medii.
6. Afișează parametrii cei mai buni și acuratețea medie obținută la cross-validation.
7. Evaluează modelul optim pe datele de testare, pentru a vedea performanța reală.
📊 Ce urmărește modelul Logistic Regression aici?
Modelul învață să clasifice vinurile în cele 3 clase (0, 1, 2) pe baza a 13 trăsături chimice, iar GridSearchCV ajustează hiperparametrii C și penalty pentru a maximiza acuratețea.'''