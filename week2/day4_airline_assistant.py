"""
Day 4: Project - Airline AI Assistant.

A customer support chatbot for "FlightAI" that uses LLM tool-calling to look
up ticket prices from a small SQLite database.
"""

import os
import json
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
openai = OpenAI()

DB = os.path.join(os.path.dirname(__file__), "prices.db")

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

set_price_function = {
    "name": "set_ticket_price",
    "description": "Set (or update) the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The destination city"},
            "price": {"type": "number", "description": "The new ticket price in dollars"},
        },
        "required": ["city", "price"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": set_price_function},
]


def init_db():
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)")
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM prices")
        if cursor.fetchone()[0] == 0:
            defaults = {"london": 799, "paris": 899, "tokyo": 1420, "berlin": 499, "sydney": 2999}
            for city, price in defaults.items():
                cursor.execute(
                    "INSERT INTO prices (city, price) VALUES (?, ?) "
                    "ON CONFLICT(city) DO UPDATE SET price = excluded.price",
                    (city, price),
                )
            conn.commit()


def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM prices WHERE city = ?", (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"


def set_ticket_price(city, price):
    print(f"DATABASE TOOL CALLED: Setting price for {city} to ${price}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prices (city, price) VALUES (?, ?) "
            "ON CONFLICT(city) DO UPDATE SET price = ?",
            (city.lower(), price, price),
        )
        conn.commit()
    return f"Ticket price to {city} has been set to ${price}"


def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        if tool_call.function.name == "get_ticket_price":
            content = get_ticket_price(arguments.get("destination_city"))
        elif tool_call.function.name == "set_ticket_price":
            content = set_ticket_price(arguments.get("city"), arguments.get("price"))
        else:
            content = "Unknown tool"
        responses.append({ "role": "tool", "content": content, "tool_call_id": tool_call.id})
    return responses


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    while response.choices[0].finish_reason == "tool_calls":
        tool_message = response.choices[0].message
        responses = handle_tool_calls(tool_message)
        messages.append(tool_message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    return response.choices[0].message.content


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found in environment - add one to your .env file")
        return

    init_db()
    gr.ChatInterface(fn=chat, type="messages").launch()


if __name__ == "__main__":
    main()