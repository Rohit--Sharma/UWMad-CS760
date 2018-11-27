# Install newsapi for python with "pip install newsapi-python"
from newsapi import NewsApiClient

# A temporary API Key
newsapi = NewsApiClient(api_key='cc1da4ceb6eb4cf7952f78387f7af276')

top_headlines = newsapi.get_top_headlines(q='bitcoin', sources='bbc-news,the-verge', category='business', language='en', country='us')       # doesn't work. Try below one
top_headlines = newsapi.get_top_headlines(q='bitcoin', sources='bbc-news', language='en')
newsapi.get_top_headlines(sources='bbc-news')
newsapi.get_top_headlines(q='India', sources='bbc-news')
newsapi.get_everything(q='India')
