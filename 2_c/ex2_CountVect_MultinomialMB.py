# [Example: Classify reviews as positive or negative based on word frequency]
# Acest program învață un model de clasificare de text care poate prezice
# dacă o recenzie este pozitivă sau negativă, pe baza cuvintelor conținute în text.

# Importăm bibliotecile necesare
import pandas as pd  # pentru lucrul cu datele sub formă de tabel (DataFrame)
from sklearn.feature_extraction.text import CountVectorizer  # pentru transformarea textului în vectori numerici
from sklearn.model_selection import train_test_split  # pentru împărțirea datelor în train/test
from sklearn.naive_bayes import MultinomialNB  # model de clasificare bazat pe Naive Bayes (potrivit pentru text)

# 1. Cream un set mic de date cu recenzii de filme și etichete (1 = pozitivă, 0 = negativă)
data = [
    ("I absolutely loved this movie, it was fantastic!", 1),
    ("Horrible plot and terrible acting, wasted my time.", 0),
    ("An instant classic, superb in every aspect!", 1),
    ("I wouldn't recommend this film to anyone.", 0),
    ("It was just okay, nothing special or groundbreaking.", 0),
    ("Brilliant! I enjoyed every minute of it!", 1)
]
# Convertim lista într-un DataFrame cu două coloane: "text" (recenzia) și "label" (eticheta)
df = pd.DataFrame(data, columns=["text", "label"])

# 2. Transformăm textul în reprezentare numerică folosind modelul Bag-of-Words
vectorizer = CountVectorizer()  # inițializăm vectorizatorul
X = vectorizer.fit_transform(df["text"])  # transformăm textele în matrice de frecvențe de cuvinte
y = df["label"]  # etichetele (target-ul) rămân neschimbate

# 3. Împărțim datele în set de antrenament (70%) și testare (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Antrenăm un model Naive Bayes pe datele de antrenament
model = MultinomialNB()  # creăm modelul de clasificare
model.fit(X_train, y_train)  # antrenăm modelul pe datele de training

# 5. Facem predicții pe setul de test
y_pred = model.predict(X_test)  # prezicem etichetele pentru recenziile din test

# 6. Evaluăm acuratețea modelului
accuracy = (y_pred == y_test).mean()  # calculăm proporția de predicții corecte
print(f"Test Accuracy: {accuracy:.2f}")  # afișăm acuratețea cu două zecimale

# Opțional: Afișăm o comparație între etichetele reale și cele prezise
comparison = pd.DataFrame({
    "Review": df["text"].iloc[y_test.index],         # recenziile corespunzătoare setului de test
    "Actual Label": y_test,                          # etichetele reale
    "Predicted Label": y_pred                        # etichetele prezise de model
})
print("\nPredictions vs. Actual:")
print(comparison)
