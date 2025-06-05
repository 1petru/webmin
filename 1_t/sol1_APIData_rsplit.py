# Exercise 1: Extracting and Cleaning Data from an API
import requests
import pandas as pd

# Step 1: Define cities to fetch weather data for
cities = ["New York", "London", "Tokyo", "Paris", "Berlin"]
#  url = f"https://wttr.in/{city}?format=%C+%t"

# Step 2: Fetch weather data from wttr.in (public API, no API key needed)
weather_data = []

for city in cities:
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.text.strip()
        condition, temperature = data.rsplit(" ", 1)
        temperature = temperature.replace("°C", "")
        weather_data.append([city, condition, temperature])

    else:
        print(f"Failed to retrieve data for {city}")

# step 3: convert data to pd dataframe

df = pd.DataFrame(weather_data, columns=["City", "Weather Condition", "Temperature (°C)"])
df.to_csv("cleaned_weather_data.csv", index=False)
print(df)

