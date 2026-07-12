"""
Week 1 exercise: a tool that takes a technical question and explains it,
answered by both OpenAI and a local Ollama model for comparison.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL_GPT = "gpt-4.1-mini"
MODEL_LLAMA = "llama3.2"
OLLAMA_HOST = "http://localhost:11434/v1"

DEFAULT_QUESTION = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""


def ask_gpt(question):
    openai = OpenAI()
    response = openai.chat.completions.create(model=MODEL_GPT, messages=[{"role": "user", "content": question}])
    return response.choices[0].message.content


def ask_llama(question):
    ollama = OpenAI(base_url=OLLAMA_HOST, api_key="ollama")
    response = ollama.chat.completions.create(model=MODEL_LLAMA, messages=[{"role": "user", "content": question}])
    return response.choices[0].message.content


def main():
    question = input("Enter a technical question (or press enter for the default): ") or DEFAULT_QUESTION

    if os.getenv("OPENAI_API_KEY"):
        print("\n--- GPT answer ---")
        print(ask_gpt(question))
    else:
        print("\nSkipping GPT: no OPENAI_API_KEY found in environment")

    try:
        print("\n--- Llama (Ollama) answer ---")
        print(ask_llama(question))
    except Exception as exc:
        print(f"Skipped: {exc}")


if __name__ == "__main__":
    main()
