"""
Day 1: Summarize a website using OpenAI's Chat Completions API.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from scraper import fetch_website_contents

load_dotenv(override=True)
openai = OpenAI()

MODEL = "gpt-4.1-mini"

system_prompt = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.
"""


def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website},
    ]


def summarize(url):
    website_contents = fetch_website_contents(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages_for(website_contents)
    )
    return response.choices[0].message.content


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY found in environment - add one to your .env file")
        return

    url = input("Enter a URL to summarize: ") or "https://edwarddonner.com"
    print(f"\nSummarizing {url}...\n")
    print(summarize(url))


if __name__ == "__main__":
    main()