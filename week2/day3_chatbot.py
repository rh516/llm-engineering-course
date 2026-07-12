"""
Day 3: Conversational AI - a Gradio ChatInterface-powered chatbot for a
clothes store, using the system prompt to steer tone and sales behavior.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
openai = OpenAI()
MODEL = "gpt-4.1-mini"

system_message = (
    "You are a helpful assistant in a clothes store. You should try to gently encourage "
    "the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. "
    "For example, if the customer says 'I'm looking to buy a hat', "
    "you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.' "
    "Encourage the customer to buy hats if they are unsure what to get.\n"
    "If the customer asks for shoes, you should respond that shoes are not on sale today, "
    "but remind the customer to look at hats!"
)


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    relevant_system_message = system_message

    if "belt" in message.lower():
        relevant_system_message += (
            " The store does not sell belts; if you are asked for belts, be sure to point out other items on sale."
        )

    messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found in environment - add one to your .env file")
        return

    gr.ChatInterface(fn=chat, type="messages").launch()


if __name__ == "__main__":
    main()
