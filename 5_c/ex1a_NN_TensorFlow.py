# [An example showing equivalent code for defining a neural net in both TensorFlow and PyTorch]
import tensorflow as tf
import numpy as np

# Convert to NumPy arrays
X = np.array([
    [0.1, 0.2],
    [0.4, 0.3],
    [0.6, 0.8],
    [0.9, 0.5]
])

y = np.array([1, 0, 1, 0])

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X, y, epochs=10)

'''
1. Definirea datelor de intrare (X) și etichetelor (y):
python
Copy
Edit
X = np.array([
    [0.1, 0.2],
    [0.4, 0.3],
    [0.6, 0.8],
    [0.9, 0.5]
])
y = np.array([1, 0, 1, 0])
Ai 4 exemple de antrenare, fiecare cu 2 caracteristici. Etichetele (y) sunt binare (0 sau 1) — ceea ce înseamnă că e o problemă de clasificare binară.

2. Crearea modelului:
python
Copy
Edit
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Este o rețea secvențială (straturi unul după altul).

Primul strat (Dense(4, activation='relu')):

4 neuroni,

funcție de activare ReLU,

primește 2 intrări (deoarece fiecare rând din X are 2 valori).

Al doilea strat (Dense(1, activation='sigmoid')):

un singur neuron pentru ieșire (clasificare binară),

funcție de activare sigmoid pentru a produce o probabilitate între 0 și 1.

3. Compilarea modelului:
python
Copy
Edit
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
Optimizer: adam — optimizator eficient pentru probleme generale.

Loss: binary_crossentropy — potrivit pentru clasificare binară.

Metrics: accuracy — măsoară cât de des modelul face predicții corecte.

4. Antrenarea modelului:
python
Copy
Edit
model.fit(X, y, epochs=10)
Rulează antrenamentul pentru 10 epoci.

Pe acest set mic de date, modelul va învăța rapid, dar poate suprapotrivi.'''