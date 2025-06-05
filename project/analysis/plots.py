import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_csv("../cleaned_steam_games.csv")

# Plot 1: Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Price (Euro)"], bins=30)
plt.title("Price Distribution")
plt.xlabel("Price (Euro)")
plt.ylabel("Number of Games")
plt.show()

# Plot 2: Review Scores Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Review Score")
plt.title("Review Scores Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Games")
plt.show()

# Plot 3: Correlation Matrix
plt.figure(figsize=(10, 5))
corr = df.drop(columns=["Title"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
