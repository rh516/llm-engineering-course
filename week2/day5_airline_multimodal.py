"""
Day 5: Multimodal Airline AI Assistant.

Extends day4's tool-calling airline assistant with:
- Image generation of the destination city (gpt-image-1-mini)
- Text-to-speech playback of the assistant's reply (gpt-4o-mini-tts)
- A custom Gradio Blocks UI combining chat, image and audio panels
"""

import os
import json
import sqlite3
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
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

tools = [{"type": "function", "function": price_function}]


def init_db():
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)")
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM prices")
        if cursor.fetchone()[0] == 0:
            defaults = {"london": 799, "paris": 899, "tokyo": 1420, "berlin": 499, "sydney": 2999}
            for city, price in defaults.items():
                cursor.execute("INSERT INTO prices (city, price) VALUES (?, ?)", (city, price))
            conn.commit()


def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM prices WHERE city = ?", (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"


def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get("destination_city")
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({"role": "tool", "content": price_details, "tool_call_id": tool_call.id})
    return responses, cities


def artist(city):
    image_response = openai.images.generate(
        model="gpt-image-1-mini",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def talker(message):
    response = openai.audio.speech.create(model="gpt-4o-mini-tts", voice="onyx", input=message)
    return response.content


def chat(history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    voice = talker(reply)

    if cities:
        image = artist(cities[0])

    return history, voice, image


def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]


def build_ui():
    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(height=500, type="messages")
            image_output = gr.Image(height=500, interactive=False)
        with gr.Row():
            audio_output = gr.Audio(autoplay=True)
        with gr.Row():
            message = gr.Textbox(label="Chat with our AI Assistant:")

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output])

    return ui


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found in environment - add one to your .env file")
        return

    init_db()
    build_ui().launch(inbrowser=True, auth=("ed", "bananas"))


if __name__ == "__main__":
    main()
