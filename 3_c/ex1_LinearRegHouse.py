# [Show a short scenario about predicting housing prices based on square footage, location, etc.]
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Create a toy dataset
#    In reality, you would load real data from a file or database.
data = {
    'sqft':      [1500, 2000, 1100, 2500, 1400, 2300],
    'bedrooms':  [3,    4,    2,    5,    3,    4],
    'location':  ['cityA', 'cityB', 'cityA', 'cityB', 'cityA', 'cityB'],
    'price':     [300000, 400000, 200000, 500000, 280000, 450000]
}
df = pd.DataFrame(data)

# 2. Separate features (X) from the target (y)
X = df[['sqft', 'bedrooms', 'location']]
y = df['price']

# 3. Convert the categorical 'location' feature into dummy (one-hot) variables
#    'drop_first=True' avoids the dummy variable trap by dropping one category.
X_encoded = pd.get_dummies(X, columns=['location'], drop_first=True)

# 4. Train a Linear Regression model
model = LinearRegression()
model.fit(X_encoded, y)

# 5. Print out the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 6. Example prediction
#    Suppose we want to predict the price for a new house with:
#      - 1600 sqft
#      - 3 bedrooms
#      - location = 'cityB'
new_house = pd.DataFrame({
    'sqft': [1600],
    'bedrooms': [3],
    'location': ['cityB']
})

# One-hot encode the new data (same columns as training)
new_house_encoded = pd.get_dummies(new_house, columns=['location'], drop_first=True)

# Ensure both have matching columns by reindexing the new data
new_house_encoded = new_house_encoded.reindex(columns=X_encoded.columns, fill_value=0)

predicted_price = model.predict(new_house_encoded)
print("Predicted price for the new house:", predicted_price[0])
'''
🧮 Coeficienți model:

Coefficients: [  170. 23000. -7000.]
Aceștia corespund ordinii coloanelor din X_encoded, care sunt:
sqft → 170
bedrooms → 23.000
location_cityB → -7.000

Deci:
✅ Pentru fiecare 1 sqft în plus, prețul crește cu 170.
✅ Pentru fiecare dormitor în plus, prețul crește cu 23.000.
✅ Dacă locația este cityB, prețul scade cu 7.000 comparativ cu cityA.

🧾 Intercept:
Intercept: -28000
Acesta este un fel de „bază” teoretică a modelului atunci când toate variabilele sunt 0. În practică, nu are o interpretare utilă directă (nu există casă cu 0 mp și 0 dormitoare), dar este necesar în formula matematică.

🏡 Predicția pentru noua casă:
sqft = 1600
bedrooms = 3
location = cityB (=> location_cityB = 1)
Predicția este calculată astfel:

price = (1600 * 170) + (3 * 23000) + (1 * -7000) + (-28000)
      = 272000 + 69000 - 7000 - 28000
      = 313000
🟩 Predicted price: 312999.99'''