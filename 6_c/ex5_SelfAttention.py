#  [Example: Show a diagram highlighting encoder self-attention layers and how they attend to different words in a sentence]
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Simulated attention scores for a sentence (5 words)
attention_scores = torch.rand(5, 5)  # (sequence_length, sequence_length)
# Simulează un matrice de scoruri de atenție între 5 cuvinte dintr-o propoziție.
# Dimensiunea este (5, 5) deoarece fiecare cuvânt (query) poate "privi" spre toate celelalte (inclusiv spre el însuși).


attention_weights = F.softmax(attention_scores, dim=-1)
# Transformă scorurile în ponderi de atenție (valori între 0 și 1), care sumează la 1 pe linie.
# Aceste ponderi determină câtă atenție acordă fiecare cuvânt altora.

# Example words in a sentence
words = ["The", "cat", "sat", "on", "mat"]

# Visualizing the attention weights
plt.figure(figsize=(6,6))
plt.imshow(attention_weights.detach().numpy(), cmap="Blues")
plt.xticks(ticks=np.arange(len(words)), labels=words)
plt.yticks(ticks=np.arange(len(words)), labels=words)
plt.xlabel("Words Attended To")
plt.ylabel("Query Words")
plt.title("Self-Attention Visualization")
plt.colorbar()
plt.show()