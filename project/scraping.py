import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
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

# Safety
# time.sleep(5)

# Scroll once to load more games
for _ in range(3):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# JavaScript Scraping
script = """
const rows = document.querySelectorAll('.search_result_row');
let data = [];

rows.forEach(row => {
    const title = row.querySelector('.title')?.innerText || 'N/A';
         date = row.querySelector('.search_released')?.innerText || 'N/A';
    const price = row.querySelector('.discount_final_price')?.innerText || 'N/A';
    const review = row.querySelector('.search_review_summary')?.getAttribute('data-tooltip-html') || 'N/A';
    data.push({
    "Title": title,
    "Release Date": date,
    "Price": price,
    "Review": review
    });
});

return data;
"""
results = driver.execute_script(script)
driver.quit()

# Converting data to a Pandas Dataframe
df = pd.DataFrame(results)
df = df[["Title", "Release Date", "Price", "Review"]]
df.to_csv("steam_games.csv", index=False)
