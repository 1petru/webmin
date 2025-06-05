from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


df = pd.read_csv("../cleaned_steam_games.csv")

# Select features and target
X = df[["Number of Reviews", "Price (Euro)", "Year", "Rank"]]
y = df["Review Percent"]

# Split into train and test (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

coefficients = dict(zip(X.columns, model.coef_))
intercept = model.intercept_
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")