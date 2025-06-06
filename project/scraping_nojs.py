import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

chromedriver = "./chromedriver.exe"

# Configure Chrome Options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-gpu")

# Initialize driver
service = Service(chromedriver)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Accessing the webpage
url = "https://store.steampowered.com/search/?filter=topsellers"
driver.get(url)

# Scroll to load more games
for _ in range(3):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# Find all game rows
rows = driver.find_elements(By.CLASS_NAME, "search_result_row")

data = []

for row in rows:
    try:
        title = row.find_element(By.CLASS_NAME, "title").text
    except:
        title = "N/A"

    try:
        date = row.find_element(By.CLASS_NAME, "search_released").text
    except:
        date = "N/A"

    try:
        price = row.find_element(By.CLASS_NAME, "discount_final_price").text
    except:
        price = "N/A"

    try:
        review = row.find_element(By.CLASS_NAME, "search_review_summary").get_attribute("data-tooltip-html")
    except:
        review = "N/A"

    data.append({
        "Title": title,
        "Release Date": date,
        "Price": price,
        "Review": review
    })

driver.quit()

# Save to CSV
df = pd.DataFrame(data)
df = df[["Title", "Release Date", "Price", "Review"]]
df.to_csv("steam_games.csv", index=False)
