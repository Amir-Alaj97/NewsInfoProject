import os
from dotenv import load_dotenv
import requests
import openai
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
        "headLineUrl":"https://newsapi.org/v2/top-headlines",
        "storiesUrl":"https://newsapi.org/v2/everything",
        "categoryParam":"category",
        "apiKeyParam":"apiKey",
        "fromDateParam":"from",
        "toDateParam":"to",
        "apiKey":os.getenv("NEWSAPI_API_KEY")
    },
    {
        "name":"MediaStack",
        "headLineUrl":"http://api.mediastack.com/v1/news",
        "storiesUrl":"http://api.mediastack.com/v1/news",
        "dateParam":"date",
        "categoryParam":"categories",
        "apiKeyParam":"access_key",
        "apiKey":os.getenv("MEDIA_STACK_KEY")
    },
    {
        "name":"GNews",
        "headLineUrl":"https://gnews.io/api/v4/top-headlines",
        "storiesUrl":"https://gnews.io/api/v4/top-headlines",
        "categoryParam":"category",
        "apiKeyParam":"apikey",
        "fromDateParam":"from",
        "toDateParam":"to",
        "apiKey":os.getenv("G_NEWS_KEY")
    }    
]

def get_filters(userInput) :
    today = datetime.today()

    prompt = f"""
    Extract the news category or categories based on these categories: ["business","entertainment","general","health","sports","science","technology"] 
    and date from the following query, convert the date to YYYY-MM-DD format, if multiple dates put in a list and return them in JSON format. If no category is found,
    try to find the closest match, otherwise return 'None Found'. If no date os found use todays date.
    
    Example: Say today's date is {today}
    Input: "Show me sports articles from five days ago"
    Output: {{"category": "sports", "date": "{(today - timedelta(days=5)).strftime("%Y-%m-%d")}"}}

    Input: "{userInput}"
    Output:
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


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

def make_request(news, category, date, userInput):
    params = {
        news["apiKeyParam"]: news["apiKey"],
        news["categoryParam"]: category
    }
    if (("headline" in userInput.lower() or "top stories" in userInput.lower()) and 
        (date == datetime.today().strftime("%Y-%m-%d"))):
        url = news["headLineUrl"]
    else:
        url = news["storiesUrl"]
        if news['name'] == "MediaStack":
            if isinstance(date, list):
                params[news["dateParam"]] = date[0]+","+date[1]
            else:
                params[news["dateParam"]] = date
        else:
            if isinstance(date, list):
                params[news["fromDateParam"]] = date[0]
                params[news["toDateParam"]] = date[1]
            else:
                params[news["fromDateParam"]] = date
    
    print(f"***YOU ARE HERE***:\n{url}\n{params}\n")
    if news["name"] == "NewsApi":
        response = requests.get(url, params)
        articles = response.json().get("articles", [])
        return '\n\n'.join([f'''Author: {article['author']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}\nContent: {article['content']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articles])
    elif news["name"] == "MediaStack":
        response = requests.get(url, params)
        articles = response.json().get("data", [])
        return '\n\n'.join([f'''Author: {article['author']}\nTitle: {article['title']}\nPublished At: {article['published_at']}\nContent: {article['description']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articles])
    elif news["name"] == "GNews":
        response = requests.get(url, params)
        articles = response.json().get("articles", [])
        return '\n\n'.join([f'''Author: {article['source']['name']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}\nContent: {article['content']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articles])

def fetch_news(news, categories, date, userInput):
    combinedArticles = ""
    if isinstance(categories, list):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(make_request, news, category, date, userInput) for category in categories]
            for future in concurrent.futures.as_completed(futures):
                combinedArticles += future.result() + '\n\n'
    elif categories == "None Found":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(make_request, news, category, date, userInput) for category in CATEGORIES]
            for future in concurrent.futures.as_completed(futures):
                combinedArticles += future.result() + '\n\n'
    else:
        combinedArticles = make_request(news, categories, date, userInput) + "\n\n"
    return combinedArticles

def process_request(userInput):
    combinedArticles = ""
    response = get_filters(userInput)
    response = json.loads(response)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_news, news, response["category"], response["date"], userInput) for news in NEWS_APIS]
        for future in concurrent.futures.as_completed(futures):
            combinedArticles += future.result() + '\n\n'
    print(combinedArticles)
    summary = invoke_articles(combinedArticles, userInput)
    return summary

def main():
    chatHistory = []
    print("This is the News Summarizer")
    while True:
        userInput = input("Enter a request about the news (type 'exit' to quit):")
        if userInput.lower() == "exit":
           print("Goodbye")
           break
        summary = process_request(userInput)
        print("AI Response: "+ summary)

if __name__ == "__main__":
    main()