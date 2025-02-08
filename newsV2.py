import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, timedelta
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import concurrent.futures
import re
from datetime import datetime
from word2number import w2n

load_dotenv()

CATEGORIES = ["business","entertainment","general","health","sports","science","technology"]

NEWS_APIS = [
    {
        "name":"NewsApi",
        "url":"https://newsapi.org/v2/top-headlines",
        "categoryParam":"category",
        "apiKeyParam":"apiKey",
        "apiKey":os.getenv("NEWSAPI_API_KEY")
    },
    {
        "name":"MediaStack",
        "url":"http://api.mediastack.com/v1/news",
        "categoryParam":"categories",
        "apiKeyParam":"access_key",
        "apiKey":os.getenv("MEDIA_STACK_KEY")
    },
    {
        "name":"GNews",
        "url":"https://gnews.io/api/v4/top-headlines",
        "categoryParam":"category",
        "apiKeyParam":"apikey",
        "apiKey":os.getenv("G_NEWS_KEY")
    }    
]

def chunk_articles(articles, maxTokens=30000):
    words = articles.split(' ')
    chunks = []
    chunk = []
    tokenCount = 0

    for word in words:
        tokenCount += len(word.split())
        if tokenCount >= maxTokens:
            chunks.append(' '.join(chunk))
            chunk = [word]
            tokenCount = len(word.split())
        else:
            chunk.append(word)
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def invoke_articles(articlesText, prompt):
    model = ChatOpenAI(model="gpt-4o")
    chunks = chunk_articles(articlesText)
    responses = []
    for chunk in chunks:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are helpful assistant with news articles."),
                ("human", "{input}\n\n{text}"),
            ]
        )
        chain =  prompt_template | model | StrOutputParser()
        response = chain.invoke({"input":prompt, "text":chunk})
        responses.append(response)
    return '\n'.join(responses)

def make_request(news, category):
    params = {
        news["apiKeyParam"]: news["apiKey"],
        news["categoryParam"]: category
    }
    if news["name"] == "NewsApi":
        response = requests.get(news["url"], params)
        articles = response.json().get("articles", [])
        return '\n\n'.join([f'''Author: {article['author']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}
                            \nContent: {article['content']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articles])
    elif news["name"] == "MediaStack":
        response = requests.get(news["url"], params)
        articles = response.json().get("data", [])
        return '\n\n'.join([f'''Author: {article['author']}\nTitle: {article['title']}\nPublished At: {article['published_at']}
                            \nContent: {article['description']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articles])
    elif news["name"] == "GNews":
        response = requests.get(news["url"], params)
        articles = response.json().get("articles", [])
        return '\n\n'.join([f'''Author: {article['source']['name']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}
                            \nContent: {article['content']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articles])

def fetch_news(news, categories):
    combinedArticles = ""
    if len(categories) == 0:
        categories = CATEGORIES
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(make_request, news, category) for category in categories]
        for future in concurrent.futures.as_completed(futures):
            combinedArticles += future.result() + '\n\n'
    return combinedArticles

def process_request(userInput):
    categories = findCategory(userInput)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_news, news, categories) for news in NEWS_APIS]
        for future in concurrent.futures.as_completed(futures):
            combinedArticles += future.result() + '\n\n'
    summary = invoke_articles(combinedArticles, userInput)
    return summary
    
def findCategory(userInput):
    filteredCategories = [category for category in CATEGORIES if category in userInput]
    return filteredCategories

def findDates(userInput):
    dateNumPattern = '''(\b(?:\d{1,2}[-/]){1}\d{1,2}[-/]\d{4}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b)'''
    dateWordPattern = r'''((?:[january|february|march|april|may|june|july|august|september|october|november|december]+(?:\s\d{1,2})(?:st|nd|rd|th)?,\s\d{4})|
    \b(?:[january|february|march|april|may|june|july|august|september|october|november|december]+(?:\s\d{1,2})(?:st|nd|rd|th)?))'''
    dateNumMatches = re.match(dateNumPattern, userInput)
    dateWordMatches = re.match(dateWordPattern, userInput.lower())
    matches = dateNumMatches + dateWordMatches
    if len(matches) == 0:
        if "today" in userInput:
            return None, None
        daysAgoPattern = r'(\d+|\b\w+\b) (days|day) ago'
        daysAgoMatch = re.match(daysAgoPattern, userInput)
        if daysAgoMatch[0].isnumeric():
            fromDate = (datetime.today() - timedelta(days=int(daysAgoMatch[0]))).strftime("%Y-%m-%d")
            return fromDate, None
        else:
            try:
                daysAgo = w2n.word_to_num(daysAgoMatch[0])
            except Exception as e:
                if daysAgoMatch[0].lower() == "a":
                    daysAgo = 1
                elif daysAgoMatch[0].lower() == "couple":
                    daysAgo = 2
                elif daysAgoMatch[0].lower() == "few":
                    daysAgo = 3
                else:
                    return None, None
            fromDate = (datetime.today() - timedelta(days=daysAgo)).strftime("%Y-%m-%d")
    elif len(dateNumMatches) == 1:
        print('''CONTINUE FROM HERE''')
        

def main():
    print("This is the News Summarizer")
    while True:
        userInput = input("Enter a request about the news (type 'exit' to quit):")
        if userInput == "quit":
           print("Goodbye")
           break
        summary = process_request(userInput)
        print("AI Response: "+ summary)

if __name__ == "__main__":
    main()