# Exercise 2: Web Scraping a Product Listings Page
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Define the target URL (public e-commerce test site)
URL = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"

# Step 2: Fetch the webpage content
response = requests.get(URL)

# Step 3: Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")
print(soup)

# Step 4: Extract product names and prices
products = []
for product in soup.find_all("div", class_="thumbnail"):  # locate product containers
    name = product.find("a", class_="title").text.strip() # extrage textul vizibil
    price = product.find("h4", class_="price").text.strip()
    products.append([name, price])

print(products)

# Step 5: Convert to a Pandas DataFrame
df = pd.DataFrame(products, columns=["Product Name", "Price"])

# Step 6: Remove duplicates (if any)
df_cleaned = df.drop_duplicates()

# Step 7: Display the cleaned data
print("Scraped Product Listings:\n")
print(df_cleaned.head(10))  # Show the first 10 products

# Step 8: Save the cleaned data to a CSV file
df_cleaned.to_csv("scraped_products.csv", index=False)
