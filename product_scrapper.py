import os
import re
import ctypes
import pandas as pd
import requests
import threading
import requests_random_user_agent
from bs4 import BeautifulSoup

productsTitle = []
productsLink = []


def scrape_products(i):
    global productsTitle
    global productsLink
    urlIterate = url + f"/p{i}/c"
    user_agent = str(requests.get('https://httpbin.org/user-agent'))
    data = requests.get(url=urlIterate, headers={"user-agent": user_agent})
    soup = BeautifulSoup(data.text, "html.parser")
    products = [x for x in soup.find_all("div", {"class": "card-item"})]
    print(f"Accessing page {urlIterate}")
    if len(products) == 0:
        print("Page: " + str(int(i) - 1))
        return

    for x in products:
        link = x.find('a', href=True)
        if x.attrs.get('data-name') is not None and link['href'] is not None and x.attrs.get('data-name') != "" and \
                link['href'] != "" and x.attrs.get('data-name') not in productsTitle:
            productsTitle.append(x.attrs.get('data-name'))
            productsLink.append(link['href'])


while True:
    userInput = input("Enter Emag link here: ")
    if userInput != "":
        break

try:
    url = userInput
    user_agent = str(requests.get('https://httpbin.org/user-agent'))
    data = requests.get(url=url, headers={"user-agent": user_agent})
    soup = BeautifulSoup(data.text, "html.parser")

    pageCount = soup.find("span", {"class": "visible-xs"})
    pageCount = re.findall(r'((?<=\">1 din ).*([^\r]*)(?=</span>))', str(pageCount))
    pageCount = str(pageCount[0]).strip("('")
    pageCount = int(pageCount.strip("', '')"))
    pageCount = min(pageCount, 150)
    threads = []
    for i in range(1, pageCount + 1):
        threadObj = threading.Thread(target=scrape_products, args=(i,))
        threads.append(threadObj)
        threadObj.start()

    for thread in threads:
        thread.join()

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
        productsLink[:] = [re.sub(old, new, str(x)) for x in productsLink]
        productsTitle[:] = [re.sub(old, new, str(x)) for x in productsTitle]
    productsTitle[:] = [str(x).split(",") for x in productsTitle]
    productsTitle[:] = [x[0] for x in productsTitle]
    productsLink[:] = [str(x).split(",") for x in productsLink]
    productsLink[:] = [x[0] for x in productsLink]

    df = pd.DataFrame({"Title": productsTitle, "Link": productsLink})
    df = df[df.Title != '']
    df.set_index("Title", inplace=True)
    if os.path.exists("Scraped products.xlsx"):
        with pd.ExcelWriter("Scraped products.xlsx", engine='openpyxl', if_sheet_exists='overlay', mode='a') as writer:
            df.to_excel(writer, sheet_name="Sheet1", startrow=writer.sheets["Sheet1"].max_row, header=False)
    else:
        df.to_excel("Scraped products.xlsx")
    ctypes.windll.user32.MessageBoxW(0, "Products have been scraped", "Warning", 0)
except:
    ctypes.windll.user32.MessageBoxW(0, "There was an error or the link is not valid", "Warning", 0)
