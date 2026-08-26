import requests
import aiohttp
import asyncio
import pandas as pd
import re

def get_url(text):
    try:
        url = re.findall(r'(https?://\S+)', text)
    except:
        url = []
    return ' '.join(url)

class WebScraper(object):
    def __init__(self, urls):
        self.urls = urls
        # Global Place To Store The Data:
        self.all_data  = []
        # Run The Scraper:
        asyncio.run(self.main())

    async def fetch(self, session, url):
        try:
            async with session.get(url, allow_redirects=False) as response:
                Location = str(response.headers["Location"]) #.split("Location': \'")#[1].split("\'")[0]
                return Location
        except Exception as e:
            print(str(e))

    async def main(self):
        tasks = []
        headers = {
            "user-agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
        async with aiohttp.ClientSession(headers=headers) as session:
            for url in self.urls:
                tasks.append(self.fetch(session, url))

            htmls = await asyncio.gather(*tasks)
            self.all_data.extend(htmls)

def get_long_url(urls):
    urls = list(urls.split(' '))
    scraper = WebScraper(urls = urls)
    return scraper.all_data

def remove_mentions(text):
    # Removal of mentions
    text = re.sub("@[^\s]*", "@user", text)
    return text

def preprocess_long_urls(data):

    data['out_link'] = ''
    data['twitter_link'] = ''
    for index, row in data.iterrows():
        ids = row['id_str']
        urls = row['long_urls']
        out_urls = []
        for u in urls:
            if ids in u:
                pass
            else:
                out_urls.append(u)

        data.loc[index, 'out_url'] = ' '.join(out_urls)
        twitter = 'False'
        for u in out_urls:
            if 'twitter.com' in u:
                twitter = 'True'
        data.loc[index, 'twitter_url'] = twitter

    return data


# 1. Create a list of URLs for our scraper to get the data for:
#urls = ["https://t.co/5ATb826FKR", "https://aka.ms/portal"]

# 2. Create the scraper class instance, this will automatically create a new event loop within the __init__ method:
#scraper = WebScraper(urls = urls)

# 3. Notice how we have a list length of 2:
#print(scraper.all_data)

data = pd.read_csv('../../data/np_without_duplicates.csv', dtype={'id_str':'str', 'parent_id_str':'str',  'in_reply_to_status_id_str':'str', })

data = data[data['post_type'].isin(['post', 'comment'])].reset_index(drop=True)
data['urls'] = data['text'].apply(get_url)
data = data[data['urls']!=''].reset_index(drop=True)
print(len(data))

data['long_urls'] = data['urls'].apply(get_long_url)
data = preprocess_long_urls(data)
data['long_urls'] = data['long_urls'].apply(lambda x: ' '.join(x))

data['text'] = data['text'].apply(remove_mentions)
data[['id_str', 'text', 'urls', 'long_urls', 'out_url', 'twitter_url']].to_csv('../../data/np_urls.csv', index=None)