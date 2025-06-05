import pandas as pd

df = pd.read_csv("./steam_games.csv")
df = df.dropna().reset_index(drop=True)


def convert_price(price):
    try:
        return float(price.replace("€", "").replace(",", ".").strip())
    except ValueError:
        return 0.0


def extract_review_text(r):
    return r.split("<br>")[0].strip()


def extract_review_percent(r):
    return r.split("<br>")[1].strip().split("%")[0].strip()


def extract_review_number(r):
    text = r.split("of the")[1].strip().split("user reviews")[0].strip()
    try:
        return int(text.replace(",", "").strip())
    except ValueError:
        return 0


def extract_year(y):
    try:
        return int(y.split(",")[1].strip())
    except ValueError:
        return 0


df["Year"] = df["Release Date"].apply(extract_year)
df["Price (Euro)"] = df["Price"].apply(convert_price)
df["Review Category"] = df["Review"].apply(extract_review_text)
df["Review Percent"] = df["Review"].apply(extract_review_percent)
df["Number of Reviews"] = df["Review"].apply(extract_review_number)

df = df.drop(["Release Date", "Price", "Review"], axis=1)


score_mapping = {
    "Overwhelmingly Positive": 9,
    "Very Positive": 8,
    "Positive": 7,
    "Mostly Positive": 6,
    "Mixed": 5,
    "Mostly Negative": 4,
    "Negative": 3,
    "Very Negative": 2,
    "Overwhelmingly Negative": 1
}

df["Review Score"] = df["Review Category"].map(score_mapping)
df = df.drop(["Review Category"], axis=1)

df["Rank"] = df.index + 1

df.to_csv("cleaned_steam_games.csv", index=False)
