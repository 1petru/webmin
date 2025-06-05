# [Example: Demonstrate training an LSTM on sentiment analysis with a small dataset]
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample dataset
texts = [
    "I love this movie",
    "This film was terrible",
    "Amazing storyline and great acting",
    "I hated the ending",
    "The best movie I have seen",
    "Not worth watching at all"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Tokenization
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')
# Transformă fiecare propoziție într-o listă de indici (corespunzători cuvintelor).
# Apoi, toate propozițiile sunt umplute cu zerouri la final (post-padding) pentru a avea aceeași lungime.


# Convert labels to numpy array
labels = np.array(labels)

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Embedding Layer: transformă fiecare index într-un vector dens (ex: cuvânt → vector 16D).
# LSTM Layer: învață secvența cuvintelor (ordine, context).
# Dense cu sigmoid: ieșire între 0 și 1, interpretabilă ca probabilitate.

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, verbose=1)

# Test prediction
sample_text = ["This movie was fantastic"]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_padded = pad_sequences(sample_seq, maxlen=padded_sequences.shape[1], padding='post')
prediction = model.predict(sample_padded)
print("Sentiment Prediction (closer to 1 = positive, closer to 0 = negative):", prediction[0][0])
