import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Define a parameter grid for 'max_depth' (tuning from 2 to 10)
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# 4. Set up GridSearchCV
#    - Use a DecisionTreeClassifier
#    - 5-fold cross-validation
#    - Evaluate using accuracy
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1   # utilize all CPU cores if available
)

# 5. Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# 6. Obtain the best estimator, its parameters, and cross-validation score
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Val Score: {grid_search.best_score_:.3f}")

# 7. Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.3f}")

'''
🔧 Cum funcționează GridSearchCV în 5 pași:
1. Tu definești o grilă de hiperparametri

param_grid = {
    'max_depth': [2, 3, 4, 5, 6]
}
Vrei să afli ce valoare de max_depth merge cel mai bine pentru DecisionTreeClassifier.

2. Pentru fiecare valoare posibilă, se antrenează și evaluează un model
Exemplu cu cv=5:
Pentru max_depth=2 → modelul se antrenează de 5 ori pe 80% din train, și se testează pe celelalte 20%.

Se calculează scorul mediu (ex: acuratețe).

Se repetă același proces pentru max_depth=3, 4, 5 etc.

3. Se compară scorurile medii pentru toate valorile testate
Exemplu rezultat:

max_depth	Cross-Val Accuracy
2	88%
3	92.2% ✅ (cel mai bun)
4	90%
5	85%

4. GridSearchCV reține modelul cu scorul cel mai bun
Accesezi astfel:


grid_search.best_params_        # {'max_depth': 3}
grid_search.best_score_         # 0.922
grid_search.best_estimator_     # modelul complet antrenat cu max_depth=3


5. Tu folosești best_estimator_ pe testul final
Testezi modelul selectat pe date complet nevăzute:

y_pred = grid_search.best_estimator_.predict(X_test)
🔁 De ce e util GridSearchCV?
Fără el, ai face manual:

for depth in [2, 3, 4, 5]:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    ...
Cu GridSearchCV, totul e automatizat și mai eficient.'''