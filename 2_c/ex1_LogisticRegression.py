# Acest program construiește un model de învățare automată care poate prezice
# dacă o persoană va cumpăra un produs sau nu, pe baza a două caracteristici: varsta si venit


# Importăm bibliotecile necesare
import pandas as pd  # pentru manipularea datelor în DataFrame-uri
from sklearn.model_selection import train_test_split  # pentru împărțirea datelor în train/test
from sklearn.linear_model import LogisticRegression  # modelul de regresie logistică

# 1. Creăm un set mic de date cu vârstă, venit și dacă a cumpărat sau nu (0/1)
data = {
    'age': [25, 40, 35, 50, 28, 60, 45],  # vârsta
    'income': [50000, 70000, 60000, 80000, 52000, 100000, 75000],  # venitul
    'purchased': [0, 1, 0, 1, 0, 1, 1]  # eticheta: 0 = nu a cumpărat, 1 = a cumpărat
}
df = pd.DataFrame(data)  # creăm un DataFrame din dicționarul de mai sus

# 2. Separăm caracteristicile (X) de eticheta țintă (y)
X = df[['age', 'income']]  # X = variabile independente (features)
y = df['purchased']        # y = variabilă dependentă (target)

# 3. Împărțim datele în set de antrenare și set de testare (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                  # datele de împărțit
    test_size=0.2,         # 20% din date vor fi folosite pentru testare
    random_state=42        # pentru a obține aceleași rezultate la fiecare rulare
)

# 4. Inițializăm și antrenăm modelul de regresie logistică
model = LogisticRegression()      # creăm instanța modelului
model.fit(X_train, y_train)       # antrenăm modelul pe datele de antrenament

# 5. Facem predicții pe setul de testare
y_pred = model.predict(X_test)    # prezicem etichetele pentru datele de test

# 6. Calculăm acuratețea predicțiilor (procentul de predicții corecte)
accuracy = (y_pred == y_test).mean()  # comparăm predicțiile cu valorile reale
print(f"Test Accuracy: {accuracy:.2f}")  # afișăm acuratețea cu 2 zecimale

# Optional: Dacă vrei să vezi întregul dataset
# print("\nDataset:")
# print(df)
