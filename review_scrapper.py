import os
import random
import re
import threading
import time

import pandas as pd
import requests
import requests_random_user_agent
from bs4 import BeautifulSoup
import urllib3
from fp.fp import FreeProxy

reviewsText = []
ratings = []
enableProxy = False


def store_reviews():
    badChars = [
        ('\[\(\'', ''),
        ('\,\)\]', ''),
        ('\[\]', ''),
        ('\[\'', ''),
        ('\'', ''),
        ('/\"', ''),
        (', \)\]', ''),
        ('\]', ''),
        ('\&amp\;#039\;', '\'')
    ]
    for old, new in badChars:
        ratings[:] = [re.sub(old, new, str(x)) for x in ratings]
        reviewsText[:] = [re.sub(old, new, str(x)) for x in reviewsText]
    reviewsText[:] = [str(x).split(",") for x in reviewsText]
    reviewsText[:] = [x[0] for x in reviewsText]
    ratings[:] = [str(x).split(",") for x in ratings]
    ratings[:] = [x[0] for x in ratings]

    df = pd.DataFrame({"ReviewText": reviewsText, "Rating": ratings})
    df = df[df.ReviewText != '']
    df.set_index("ReviewText", inplace=True)

    if os.path.exists("Scraped reviews.xlsx"):
        with pd.ExcelWriter("Scraped reviews.xlsx", engine='openpyxl', if_sheet_exists='overlay', mode='a') as writer:
            df.to_excel(writer, sheet_name="Sheet1", startrow=writer.sheets["Sheet1"].max_row, header=False)
    else:
        df.to_excel("Scraped reviews.xlsx")


def scrape_reviews(i):
    global reviewsText
    global ratings
    try:
        # If you own a proxy you can add it below and use it
        # If you want to use a free proxy make the enableProxy parameter true
        if enableProxy:
            proxy1 = FreeProxy().get()
            proxy2 = FreeProxy(https=True).get()
            proxyObj = {"http": proxy1,
                        "https": proxy2}
            user_agent = str(requests.get('https://httpbin.org/user-agent'))
            data = requests.get(url=url, headers={"user-agent": user_agent}, proxies=proxyObj, verify=False)
        else:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
            data = requests.get(url=url, headers={"user-agent": user_agent})
        soup = BeautifulSoup(data.text, "html.parser")
        reviews = [x for x in soup.find_all("div", {"class": "product-review-body"})]
        for x in reviews:
            stars = x.find("div", {"class": "star-rating"})
            textDiv = x.find("div", {"class": "review-body-container"})
            if stars is not None and textDiv is not None and textDiv.text != "":
                reviewsText.append(textDiv.text)
                ratings.append(stars.attrs.get("class")[2][6])
    except Exception as er:
        print(er)
        scrape_reviews(url)


urllib3.disable_warnings()
if os.path.exists("Scraped products.xlsx"):
    df = pd.read_excel(r'Scraped products.xlsx')
    # Below you can modify the batch size and waiting time after each batch for collecting product reviews
    # Batches are saved to file after each one is completed
    batch_size = 200
    batch = 200
    threads = []
    for index, row in df.iterrows():
        print(f"Scraping row {index}")
        url = row['Link']
        threadObj = threading.Thread(target=scrape_reviews, args=(url,))
        threads.append(threadObj)
        threadObj.start()
        if index > batch:
            for thread in threads:
                thread.join()
            threads = []
            store_reviews()
            batch = batch + batch_size
            reviewsText = []
            ratings = []
            waitTime = random.randrange(60, 240)
            print(f"Batch saved. Waiting {waitTime} seconds")
            time.sleep(waitTime)
    for thread in threads:
        thread.join()
    store_reviews()
else:
    print("Scrape products first")
