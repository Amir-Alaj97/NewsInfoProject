import openai
import requests
from datetime import datetime, timedelta
#import dateparser
import os
import json
from dotenv import load_dotenv
import time

load_dotenv()



'''
That gets you your categories. But the same prompt can be used to get the summarization
'''

def get_llm_response(user_input):
    starttime = time.time()
    today = datetime.today()
    """Uses GPT to extract category and date from user input and return a structured response."""
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
                "date":[{today.strftime("%Y-%m-%d")}],
                "language":"",
                "country":"ru"
            }}
            User: What's happening with video game, AI and football news? english
            {{
                "categories":[{{"technology":"video game, AI"}}, {{"sports":"football"}}],
                "date":[{today.strftime("%Y-%m-%d")}],
                "language":"en",
                "country":""
            }}
            User: What is Elon Musk doing and how's busness news?
            {{
                "categories":[{{"general":"Elon Musk"}}, "business"],
                "date":[{today.strftime("%Y-%m-%d")}],
                "language":"en",
                "country":""
            }}
        Input: {user_input}
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    endtime = time.time()
    elapsed_time = endtime - starttime
    print(f"Execution Time: {elapsed_time:.4f} seconds")
    print(response.usage.total_tokens)
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def main():
    apis = get_llm_response("What is Donald Trump up to? and summarize tech articles")
    key = os.getenv("NEWSAPI_API_KEY")
    #print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
    