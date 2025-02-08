import os
from dotenv import load_dotenv
import requests
import openai
import io
import json
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
import concurrent.futures
import re
from datetime import datetime
from word2number import w2n
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import speech_recognition as sr
import tiktoken
import time

load_dotenv()
model = ChatOpenAI(model="gpt-4o", verbose=True)

TOTAL_TOKENS = 0

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

def check_used_categories(currCategories, usedCategories):
    if currCategories == "None Found":
        currCategories = CATEGORIES
    if len(usedCategories) == 0:
        return currCategories

    if isinstance(currCategories, list):
        newCategories = [category for category in currCategories if category not in usedCategories]
        return newCategories
    elif isinstance(currCategories, str):
        if currCategories in usedCategories:
            return []
        else:
            return currCategories


def get_filters(userInput) :
    global TOTAL_TOKENS
    starttime = time.time()
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
    endtime = time.time()
    elapsed_time = endtime - starttime
    print(f"Execution Time for getting filters\n: {elapsed_time:.4f} seconds")
    print("Get Filter tokens:")
    print(response.usage.total_tokens)
    TOTAL_TOKENS += response.usage.total_tokens

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

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def invoke_articles(articlesText, prompt, chatHistory):
    global TOTAL_TOKENS
    query = HumanMessage(content=prompt+'\n\n'+articlesText)
    chatHistory.append(query)
    total_tokens = sum(count_tokens(msg.content) for msg in chatHistory)
    print()
    TOTAL_TOKENS += total_tokens
    startTime = time.time()
    result = model.invoke(chatHistory)
    endtime = time.time()
    elapsed_time = endtime - startTime
    print(f"Execution Time for invoking articles\n: {elapsed_time:.4f} seconds")
    print("Tokens used:")
    print(total_tokens)
    return result.content

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
    else:
        combinedArticles = make_request(news, categories, date, userInput) + "\n\n"
    return combinedArticles

def process_request(userInput, chatHistory, usedCategories):
    combinedArticles = ""
    response = get_filters(userInput)
    response = json.loads(response)
    categories = check_used_categories(response["category"], usedCategories)
    if len(categories) > 0:
        print("Searching the web...")
        startTime = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_news, news, categories, response["date"], userInput) for news in NEWS_APIS]
            for future in concurrent.futures.as_completed(futures):
                combinedArticles += future.result() + '\n\n'
        endtime = time.time()
        elapsed_time = endtime - startTime
        print(f"Execution Time for gathering articles\n: {elapsed_time:.4f} seconds")
        if isinstance(categories, list):
            usedCategories += categories
        else:
            usedCategories.append(categories)
    summary = invoke_articles(combinedArticles, userInput, chatHistory)
    return summary

def main():
    global TOTAL_TOKENS
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 250
    recognizer.pause_threshold = 2
    recognizer.dynamic_energy_adjustment_damping = 0.1
    chatHistory = []
    usedCategories = []
    systemMessage = SystemMessage(content="You are helpful assistant with news articles.")
    chatHistory.append(systemMessage)
    print("This is the News Summarizer")
    while True:
        input("Press enter when you are ready.")
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening for your input (say Quit to exit the program)...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        with open(".\\audio.wav", "wb") as f:
            f.write(audio.get_wav_data())
        with open(".\\audio.wav", "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        userInput = transcript.text
        whisperTokens = count_tokens(userInput, model="gpt-4")
        TOTAL_TOKENS += whisperTokens
        print("You:", userInput)
        if userInput.lower() == "quit.":
           print("Goodbye")
           break
        startTime = time.time()
        summary = process_request(userInput, chatHistory, usedCategories)
        endtime = time.time()
        elapsed_time = endtime - startTime
        print(f"Execution Time for full run\n: {elapsed_time:.4f} seconds")
        print("***TOTAL TOKENS***")
        print(TOTAL_TOKENS)
        print("AI Response: "+ summary)
        chatHistory.append(AIMessage(content=summary))

if __name__ == "__main__":
    main()