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
TOTAL_TIME = 0.0

CATEGORIES = ["business","entertainment","general","health","sports","science","technology"]

def check_used_categories(currCategories, usedCategories):
    if len(usedCategories) == 0:
        return currCategories
    else:
        newCategories = [category for category in currCategories if category["category"] not in usedCategories]
        return newCategories

def get_urls(userInput) :
    global TOTAL_TOKENS
    today = datetime.today()
    """Uses GPT to extract category and date from user input and return a structured response."""
    prompt = f"""
            You are an AI that generates NewsAPI and GNews API calls based on user input. Follow these rules:
            General Rules:
                1.	Generate API calls for both NewsAPI and GNews.
                2.	One API call per category and per news api.
                3.  Save into a list of dictionaries where the keys go [{{sourceName, category, url}}]
                4.	Always include apiKey or apikey.
                5.  If no date is present, use today.
                6.  If category from category options is not seen, formulate closest category by key word(s),
                    then search by category and query the key word.
            Category Options:
                •	business
                •	entertainment
                •	general
                •	health
                •	science
                •	sports
                •	technology
            NewsAPI Rules:
            Base URL: https://newsapi.org/v2/
                •	Endpoints:
                    o	everything: For all articles (requires q, does not support category).
                    o	top-headlines: For top news (requires country or category).
                •	Required Parameters:
                    o	q: Required for everything, optional for top-headlines. If a category is given for everything, place it in q.
                    o	category: Only for top-headlines, ignored for everything.
                    o	country: Optional.
                    o	from, to: (ISO 8601) Filters by date range.
                    o	sortBy: (relevancy, popularity, publishedAt).
                    o	pageSize: (max 100).
                •	Logic:
                    o	If the input suggests top headlines, use top-headlines with category if available.
                    o	Otherwise, use everything, placing category in q if provided.
                    o	Todays news is only stored in top headlines endpoint and all previous days are in everything endpoint.
                    o	If you formulated a category with keyword(s) headlines url with have: category={{formulated category}}&q{{keyword(s)}},
                        every url, since it does not have category: q={{category}} OR {{keyword(s)}}.
            GNews Rules:
            Base URL: https://gnews.io/api/v4/
                •	Endpoints:
                    o	search: For all articles (requires q).
                    o	top-headlines: For top news (supports category).
                •	Required Parameters:
                    o	q: Required for search, optional for top-headlines. If a category is given for search, place it in q.
                    o	category: Only for top-headlines, ignored for search.
                    o	lang, country: Optional filters.
                    o	max: (max 100).
                    o	from, to: (ISO 8601) Filters by date range (only for search).
                •	Logic:
                    o	If the input suggests top headlines, use top-headlines with category if available.
                    o	Otherwise, use search, placing category in q if provided.
                    o	If you formulated a category with keyword(s) headlines url with have: category={{formulated category}}&q{{keyword(s)}},
                        every url, since it does not have category: q={{category}} OR {{keyword(s)}}.
            Todays date: {today}
            Example Output:
            User: "Summarize top tech news in the US"
            [
                {{
                    "sourceName":"NewsAPI",
                    "category":"technology",
                    "url":"https://newsapi.org/v2/top-headlines?category=technology&country=us&apiKey=YOUR_API_KEY"
                }},
                {{
                    "sourceName":"GNews",
                    "category":"technology",
                    "url":"https://gnews.io/api/v4/top-headlines?category=technology&from={today.strftime("%Y-%m-%d")}&country=us&apikey=YOUR_API_KEY""
                }}
            ]
            User: "Summarize tech news in the US from 5 days ago."
            [
                {{
                    "sourceName":"NewsAPI",
                    "category":"technology",
                    "url":"https://newsapi.org/v2/everything?q=technology&from={(today - timedelta(days=5)).strftime("%Y-%m-%d")}&country=us&apiKey=YOUR_API_KEY"
                }},
                {{
                    "sourceName":"GNews",
                    "category":"technology",
                    "url":"https://gnews.io/api/v4/search?q=technology&from={(today - timedelta(days=5)).strftime("%Y-%m-%d")}&country=us&apikey=YOUR_API_KEY"
                }}
            ]
            User: "Summarize tech and sports headlines."
            [
                {{
                    "sourceName":"NewsAPI",
                    "category":"technology",
                    "url":"https://newsapi.org/v2/top-headlines?category=technology&apiKey=YOUR_API_KEY"
                }},
                {{
                    "sourceName":"NewsAPI",
                    "category":"sports",
                    "url":"https://newsapi.org/v2/top-headlines?category=sports&apiKey=YOUR_API_KEY"
                }},
                {{
                    "sourceName":"GNews",
                    "category":"technology",
                    "url":"https://gnews.io/api/v4/top-headlines?category=technology&from={today.strftime("%Y-%m-%d")}&apikey=YOUR_API_KEY"
                }},
                {{
                    "sourceName":"GNews",
                    "category":"sports",
                    "url":"https://gnews.io/api/v4/top-headlines?category=sports&from={today.strftime("%Y-%m-%d")}&apikey=YOUR_API_KEY"
                }}
            ]
            User: "Summarize video games and football news from 5 days ago."
            [
                {{
                    "sourceName":"NewsAPI",
                    "category":"technology",
                    "url":"https://newsapi.org/v2/everything?q=technology OR video games&from={(today - timedelta(days=5)).strftime("%Y-%m-%d")}&apiKey=YOUR_API_KEY""
                }},
                {{
                    "sourceName":"NewsAPI",
                    "category":"sports",
                    "url":"https://newsapi.org/v2/everything?q=sports OR football&from={(today - timedelta(days=5)).strftime("%Y-%m-%d")}&apiKey=YOUR_API_KEY"
                }},
                {{
                    "sourceName":"GNews",
                    "category":"technology",
                    "url":"https://gnews.io/api/v4/search?q=technology OR video games&from={(today - timedelta(days=5)).strftime("%Y-%m-%d")}&apikey=YOUR_API_KEY"
                }},
                {{
                    "sourceName":"GNews",
                    "category":"sports",
                    "url":"https://gnews.io/api/v4/top-headlines?q=sports OR football&from={(today - timedelta(days=5)).strftime("%Y-%m-%d")}&apikey=YOUR_API_KEY"
                }}
            ]


        Input: {userInput}
    """
    startTime = time.time()
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    endTime = time.time()
    elapsedTime = endTime - startTime
    print("Time Lapsed For Getting URLs:")
    print(f"{elapsedTime:.4f}")
    print("Tokens used:")
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
    global TOTAL_TIME
    query = HumanMessage(content=prompt+'\n\n'+articlesText)
    chatHistory.append(query)
    total_tokens = sum(count_tokens(msg.content) for msg in chatHistory)
    TOTAL_TOKENS += total_tokens
    startTime = time.time()
    result = model.invoke(chatHistory)
    endTime = time.time()
    elapsedTime = endTime - startTime
    print("Time Lapsed For Invoking Articles:")
    print(f"{elapsedTime:.4f}")
    print("Tokens Spent:")
    print(total_tokens)

    return result.content

def make_request(news):
    if news["sourceName"] ==  "NewsAPI":
        response = requests.get(news["url"].replace("YOUR_API_KEY", os.getenv("NEWSAPI_API_KEY")))
        articles = response.json().get("articles", [])
        text = '\n\n'.join([f'''Author: {article['author']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}\nContent: {article['content']}\nCategory: {news["category"]}\nApiSource: {news['sourceName']}\n''' for article in articles])
        return text
    elif news["sourceName"] == "GNews":
        response = requests.get(news["url"].replace("YOUR_API_KEY", os.getenv("G_NEWS_KEY")))
        articles = response.json().get("articles", [])
        text = '\n\n'.join([f'''Author: {article['source']['name']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}\nContent: {article['content']}\nCategory: {news["category"]}\nApiSource: {news['sourceName']}\n''' for article in articles])
        return text

def process_request(userInput, chatHistory, usedCategories):
    combinedArticles = ""
    newsApis = get_urls(userInput)
    newsApis = json.loads(newsApis)
    newsApis = check_used_categories(newsApis, usedCategories)
    if len(newsApis) > 0:
        startTime = time.time()
        print("Searching the web...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(make_request, news) for news in newsApis]
            for future in concurrent.futures.as_completed(futures):
                combinedArticles += future.result() + '\n\n'
        categories = [item["category"] for item in newsApis]
        usedCategories += categories
        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Time Lapsed For Performing API calls:")
        print(f"{elapsedTime:.4f}")
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
        print("Tokens used for Whisper:")
        print(whisperTokens)
        print("You:", userInput)
        if userInput.lower() == "quit.":
           print("Goodbye")
           break
        startTime = time.time()
        summary = process_request(userInput, chatHistory, usedCategories)
        print("AI Response: "+ summary)
        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Time Lapsed For Whole run:")
        print(f"{elapsedTime:.4f}")
        print("Totel Tokens for this Run:")
        print(TOTAL_TOKENS)
        TOTAL_TOKENS = 0
        chatHistory.append(AIMessage(content=summary))

if __name__ == "__main__":
    main()