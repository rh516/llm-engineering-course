"""
Day 2: Building user interfaces with Gradio - from a trivial "shout" function
up to a streaming, multi-model company brochure generator.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from scraper import fetch_website_contents

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI()

system_message = "You are a helpful assistant that responds in markdown without code blocks"


def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()


def message_gpt(prompt):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=messages)
    return response.choices[0].message.content


def stream_gpt(prompt):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    stream = openai.chat.completions.create(model="gpt-4.1-mini", messages=messages, stream=True)
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


def stream_brochure(company_name, url, model):
    yield ""
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += fetch_website_contents(url)
    if model == "GPT":
        result = stream_gpt(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


def build_shout_ui():
    return gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never")


def build_gpt_ui():
    message_input = gr.Textbox(label="Your message:", info="Enter a message for GPT-4.1-mini", lines=7)
    message_output = gr.Markdown(label="Response:")
    return gr.Interface(
        fn=stream_gpt,
        title="GPT",
        inputs=[message_input],
        outputs=[message_output],
        examples=[
            "Explain the Transformer architecture to a layperson",
            "Explain the Transformer architecture to an aspiring AI engineer",
        ],
        flagging_mode="never",
    )


def build_brochure_ui():
    name_input = gr.Textbox(label="Company name:")
    url_input = gr.Textbox(label="Landing page URL including http:// or https://")
    model_selector = gr.Dropdown(["GPT"], label="Select model", value="GPT")
    message_output = gr.Markdown(label="Response:")
    return gr.Interface(
        fn=stream_brochure,
        title="Brochure Generator",
        inputs=[name_input, url_input, model_selector],
        outputs=[message_output],
        examples=[
            ["Hugging Face", "https://huggingface.co", "GPT"],
            ["Edward Donner", "https://edwarddonner.com", "GPT"],
        ],
        flagging_mode="never",
    )


UIS = {
    "shout": build_shout_ui,
    "gpt": build_gpt_ui,
    "brochure": build_brochure_ui,
}


def main():
    if not openai_api_key:
        print("No OPENAI_API_KEY found in environment - add one to your .env file")
        return

    choice = input(f"Which UI to launch? ({'/'.join(UIS)}) [brochure]: ") or "brochure"
    builder = UIS.get(choice, build_brochure_ui)
    builder().launch()


if __name__ == "__main__":
    main()
    