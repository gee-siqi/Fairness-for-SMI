import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import json


# get top 1000 names and statistics from listing
def get_list(url):
    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='top-charts')

        titles = []
        data_values = []

        # Loop through each row (tr) in the table
        for row in table.find_all('tr'):
            # Find all cells (td) in the row
            cells = row.find_all('th')
            datas = row.find_all('td')
            if cells and len(cells) == 7:
                for i in range(7):
                    title = cells[i].text.strip()
                    titles.append(title)
            if datas:
                data = [datas[j].text.strip() for j in range(len(datas))]

                data_values.append(data)
        df = pd.DataFrame(data_values, columns=titles)

    else:
        print('Failed to fetch data. Status Code:', response.status_code)
        df = None
    return df


# get published time from YouTube api (not accurate)
def get_time(name):
    url_api = f"https://youtube.googleapis.com/youtube/v3/channels?key=AIzaSyBHXi5MEagFtCdHnOZP41vBztWktZfUfak&part=snippet,statistics&forUsername={name}"

    payload = {}
    headers = {}
    response = requests.request("GET", url_api, headers=headers, data=payload)

    data = json.loads(response.text)
    # print(data)
    if "items" in data:
        published_at = data["items"][0]["snippet"].get("publishedAt")
    else:
        published_at = ""
    return published_at


# preprocess
def process(df):
    # delete data with empty info
    df = df[(df['video views'] != 0) & (df['video count'] != 0)]
    df["video count"] = df["video count"].map(lambda x: int(x.replace(',', '')))
    df["subscribers"] = df["subscribers"].map(lambda x: int(x.replace(',', '')))
    df["video views"] = df["video views"].map(lambda x: int(x.replace(',', '')))

    # assume the beginning time is the first day of the year(not accurate)
    df["year"] = df["started"].map(lambda x: 2023 - int(x) + 0.5)
    df["freq_m"] = df["video count"] / df["year"] / 12

    # use youtuber api to get start time(!!! not accurate, only 185/1000 can get value)
    # df["published"] = df["Youtuber"].map(get_time)
    # df['published'].eq('').sum()  # count empty value from api
    return df



