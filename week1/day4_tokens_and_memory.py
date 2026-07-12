"""
Day 4: Tokenization with tiktoken, and demonstrating that LLM calls are
stateless - "memory" is an illusion created by resending the conversation.
"""

import os
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"


def show_tokenization(text="Hi my name is Ray Huang and I like boba"):
    encoding = tiktoken.encoding_for_model(MODEL)
    tokens = encoding.encode(text)
    print(f"'{text}' -> {tokens}\n")
    for token_id in tokens:
        print(f"{token_id} = {encoding.decode([token_id])!r}")


def demonstrate_statelessness():
    openai = OpenAI()

    print("\n--- Call 1: introduce ourselves ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hi! I'm Ed!"},
    ]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    print(response.choices[0].message.content)

    print("\n--- Call 2: ask a follow-up in a NEW, unrelated call (no memory) ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What's my name?"},
    ]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    print(response.choices[0].message.content)

    print("\n--- Call 3: give the illusion of memory by resending the whole conversation ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hi! I'm Ed!"},
        {"role": "assistant", "content": "Hi Ed! How can I assist you today?"},
        {"role": "user", "content": "What's my name?"},
    ]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    print(response.choices[0].message.content)


def main():
    show_tokenization()

    if not os.getenv("OPENAI_API_KEY"):
        print("\nNo OPENAI_API_KEY found in environment - skipping memory demo")
        return

    demonstrate_statelessness()


if __name__ == "__main__":
    main()