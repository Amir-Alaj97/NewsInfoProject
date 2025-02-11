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
import time
import tiktoken

load_dotenv()

TOTAL_TOKENS = 0

model = ChatOpenAI(model="gpt-4o")

CATEGORIES = ["business","entertainment","general","health","sports","science","technology"]

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

NEWS_APIS = [
    {
        "name":"NewsApi",
        "headLineUrl":"https://newsapi.org/v2/top-headlines",
        "storiesUrl":"https://newsapi.org/v2/everything",
        "categoryParam":"category",
        "apiKeyParam":"apiKey",
        "fromDateParam":"from",
        "toDateParam":"to",
        "languageParam":"language",
        "countryParam":"country",
        "apiKey":os.getenv("NEWSAPI_API_KEY")
    },
    {
        "name":"GNews",
        "headLineUrl":"https://gnews.io/api/v4/top-headlines",
        "storiesUrl":"https://gnews.io/api/v4/search",
        "categoryParam":"category",
        "apiKeyParam":"apikey",
        "fromDateParam":"from",
        "toDateParam":"to",
        "languageParam":"lang",
        "countryParam":"country",
        "apiKey":os.getenv("G_NEWS_KEY")
    }    
]

def call_newsApi(news, category, date, language, country, userInput):
    params = {news["apiKeyParam"]:news["apiKey"]}
    #URL
    if date[0] == datetime.today().strftime("%Y-%m-%d"):
        if news["name"] == "NewsApi":
            url = news["headLineUrl"]
        elif news["name"] == "GNews" and ("headlines" in userInput or "top stories" in userInput):
            url = news["headLineUrl"]
        else:
            url = news["storiesUrl"]
    else:
        url = news["storiesUrl"]
    #Categories/Key Words
    if "top-headlines" in url:
        if isinstance(category, dict):
            params[news["categoryParam"]] = list(category.items())[0][0]
            params["q"] = ' OR '.join(f'"{word.strip()}"' for word in list(category.items())[0][1].split(","))
        else:
            params[news["categoryParam"]] = category
    else:
        if isinstance(category, dict):
            query = ""
            if list(category.items())[0][0] != "general":
                query = f'"{list(category.items())[0][0]}" OR '
            query += ' OR '.join(f'"{word.strip()}"' for word in list(category.items())[0][1].split(","))
            params["q"] = query
        else:
            params["q"] = category
    #dates
    params[news["fromDateParam"]] = date[0]
    if len(date) > 1:
        params[news["toDateParam"]] = date[1]
    if language != "":
        params[news["languageParam"]] = language
    if country != "":
        params[news["countryParam"]] = country
    print(url)
    print(params)
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

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
    today = datetime.today()

    prompt = f"""
            You are an AI that generates respective parameters based on user input:
            parameters are: categories(list), dates(list), laguage, and country
            Rules:
                1. categories optiones are these:
                    •	business
                    •	entertainment
                    •	general
                    •	health
                    •	science
                    •	sports
                    •	technology
                2. categories is stored in a list even if there is one.
                3. If category is not explicity stated, use keywords to define a category yourself.
                4. If you used keywords to define category store both like [{{categoryDefined}}:{{keywords used}}]
                5. Convert date format to YYYY-MM-DD, store in list even if there is one, and sort older to sooner.
                6. If date is not explicitly defined default to {today}
                7. use language code language default is ""
                8. use country code for country, default is ""
            Examples:
            User: Summarize headline technology articles from 5 days ago.
            {{
                "categories":["technology"],
                "date":["{(today - timedelta(days=5)).strftime("%Y-%m-%d")}"],
                "language":"",
                "country":""
            }}
            User: Were there any volleyball incidents in Russia?
            {{
                "categories":[{{"sports":"volleyball"}}],
                "date":["{today.strftime("%Y-%m-%d")}"],
                "language":"",
                "country":"ru"
            }}
            User: What's happening with video game, AI and football news? english
            {{
                "categories":[{{"technology":"video game, AI"}}, {{"sports":"football"}}],
                "date":["{today.strftime("%Y-%m-%d")}"],
                "language":"en",
                "country":""
            }}
            User: What is Elon Musk doing and how's busness news?
            {{
                "categories":[{{"general":"Elon Musk"}}, "business"],
                "date":["{today.strftime("%Y-%m-%d")}"],
                "language":"en",
                "country":""
            }}
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
    print(f"Get Filters Execution Time: {elapsedTime:.4f} seconds")
    print("Tokens Used:")
    print(response.usage.total_tokens)
    TOTAL_TOKENS += response.usage.total_tokens
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def invoke_articles(articlesText, prompt, chatHistory):
    global TOTAL_TOKENS
    query = HumanMessage(content=prompt+'\n\n'+articlesText)
    chatHistory.append(query)
    promptTokens = sum(count_tokens(msg.content) for msg in chatHistory)
    startTime = time.time()
    result = model.invoke(chatHistory)
    endTime = time.time()
    elapsedTime = endTime - startTime
    completionTokens = count_tokens(result.content)
    totalTokens = promptTokens + completionTokens
    TOTAL_TOKENS += totalTokens
    print(f"Invoking articles Execution Time: {elapsedTime:.4f} seconds")
    print("Tokens Used:")
    print(totalTokens)
    return result.content

def make_request(news, category, date, language, country, userInput):
    articlesList = call_newsApi(news, category, date, language, country, userInput)
    if news["name"] == "NewsApi":
        return '\n\n'.join([f'''Author: {article['author']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}\nContent: {article['content']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articlesList])
    elif news["name"] == "GNews":
       return '\n\n'.join([f'''Author: {article['source']['name']}\nTitle: {article['title']}\nPublished At: {article['publishedAt']}\nContent: {article['content']}\nCategory: {category}\nApiSource: {news['name']}\n''' for article in articlesList])

def fetch_news(news, response, userInput):
    combinedArticles = ""
    if isinstance(response, dict):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(make_request, news, category, response["date"], response["language"], response["country"], userInput) for category in response["categories"]]
            for future in concurrent.futures.as_completed(futures):
                combinedArticles += future.result() + '\n\n'
    return combinedArticles

def process_request(userInput, chatHistory, usedCategories):
    combinedArticles = ""
    response = get_filters(userInput)
    response = json.loads(response)
    #categories = check_used_categories(response["category"], usedCategories)
    if len(response) > 0:
        startTime = time.time()
        print("Searching the web...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_news, news, response, userInput) for news in NEWS_APIS]
            for future in concurrent.futures.as_completed(futures):
                combinedArticles += future.result() + '\n\n'
        endTime = time.time()
        elapsedTime = endTime - startTime
        print(f"Performing API calls Execution Time: {elapsedTime:.4f} seconds")
        '''
        if isinstance(categories, list):
            usedCategories += categories
        else:
            usedCategories.append(categories)
        '''
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
        '''
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
        print("You:", userInput)
        '''
        userInput = "Have there been any incidents in Russia?"
        if userInput.lower() == "quit.":
           print("Goodbye")
           break
        startTime = time.time()
        summary = process_request(userInput, chatHistory, usedCategories)
        print("AI Response: "+ summary)
        endTime = time.time()
        elapsedTime = endTime - startTime
        print("**********************************")
        print(f"Complete run Execution Time: {elapsedTime:.4f} seconds")
        print("Total Tokens used:")
        print(TOTAL_TOKENS)
        print("**********************************")

        chatHistory.append(AIMessage(content=summary))

if __name__ == "__main__":
    main()