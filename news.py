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

def store_articles(articles, prompt):
    articlesText = '\n'.join([f'''Author: {article['author']}\nTitle: {article['title']}\nDescription: {article['description']}
                                \nPublished At: {article['publishedAt']}\nContent: {article['content']}\n''' for article in articles])
    model = ChatOpenAI(model="gpt-4o")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are helpful assistant with news articles."),
            ("human", "{input}\n\n{text}"),
        ]
    )
    chain =  prompt_template | model | StrOutputParser()
    return chain.invoke({"input":prompt, "text":articlesText})

def get_news(category=None, dateRange=None):
    baseUrl = "https://newsapi.org/v2/top-headlines"
    newsApiKey = os.getenv("NEWSAPI_API_KEY")
    requestParams = {
        "apiKey":newsApiKey,
        "country":"us"
    }
    '''
    if dateRange is not None:
        today = datetime.today()
        fromDate = (today - timedelta(days=dateRange)).strftime("%Y-%m-%d")
        requestParams["from"] = fromDate
    '''
    response = requests.get(baseUrl, params=requestParams)
    if response.status_code == 200:
        return response.json().get('articles',[])
    else:
        print("Error getting the news")
        print(response.json())
        return []

def process_request(userInput):
    if "headline" in userInput.lower() or "top stories" in userInput.lower() or "top story" in userInput.lower():
        articles = get_news()
        #print(articles)
        summary = store_articles(articles, userInput)
    return summary

def main():
    load_dotenv()
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
