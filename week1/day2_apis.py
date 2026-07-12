
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

OLLAMA_BASE_URL = "http://localhost:11434/v1"


def call_with_raw_http(api_key, prompt="Tell me a fun fact"):
    """Call the Chat Completions endpoint directly, with no client library."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()["choices"][0]["message"]["content"]


def call_openai(prompt="Tell me a fun fact"):
    openai = OpenAI()
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_ollama(model="llama3.2", prompt="Tell me a fun fact"):
    """Ollama also exposes an OpenAI-compatible endpoint, running locally."""
    ollama = OpenAI(base_url=OLLAMA_BASE_URL)
    response = ollama.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY found in environment - add one to your .env file")
        return

    print("--- Raw HTTP call ---")
    print(call_with_raw_http(api_key))

    print("\n--- OpenAI client library ---")
    print(call_openai())

    print("\n--- Ollama (local, OpenAI-compatible endpoint) ---")
    try:
        print(call_ollama())
    except requests.exceptions.ConnectionError:
        print("Skipped: Ollama isn't running locally (run `ollama serve`)")


if __name__ == "__main__":
    main()