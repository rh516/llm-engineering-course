"""
Day 5: A full business solution - build a company brochure by:
1. Scraping a company's landing page and links
2. Asking an LLM which links are relevant (About, Careers, etc.)
3. Scraping those pages too
4. Asking an LLM to assemble a brochure from everything gathered
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from scraper import fetch_website_contents, fetch_website_links

load_dotenv(override=True)
openai = OpenAI()

LINKS_MODEL = "gpt-4.1-mini"
BROCHURE_MODEL = "gpt-4.1-mini"

link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""


def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company,
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):    
"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)

    return user_prompt


def select_relevant_links(url):
    print(f"Selecting relevant links for {url} by calling {LINKS_MODEL}")
    response = openai.chat.completions.create(
        model=LINKS_MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)},
        ],
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    print(f"Found {len(links['links'])} relevant links")

    return links


def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"

    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])

    return result


def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000]

    return user_prompt


def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model=BROCHURE_MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ]
    )
    return response.choices[0].message.content


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found in environment - add one to your .env file")
        return

    company_name = input("Company name: ") or "HuggingFace"
    url = input("Company URL: ") or "https://huggingface.co"
    print(create_brochure(company_name, url))


if __name__ == "__main__":
    main()