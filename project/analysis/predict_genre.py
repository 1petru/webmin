from transformers import pipeline
import pandas as pd

df = pd.read_csv("../cleaned_steam_games.csv")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-cnn")

genres = ["RPG", "Shooter", "Strategy", "Adventure", "Survival", "Racing", "Fighting", "MMO", "Simulation", "Horror"]

predicts = []
titles = df["Title"][:10]
for title in titles:
    result = classifier(title, candidate_labels=genres)
    predicts.append([title, result['labels'][0]])

predict_df = pd.DataFrame(predicts, columns=["Title", "Prediction"])
predict_df.to_csv("../genre_pred.csv", index=False)
