"""
Day 1: Expert Knowledge Worker — brute-force RAG.

A low-cost Q&A assistant for employees of Insurellm (an Insurance Tech company),
using naive keyword-matching retrieval (no embeddings, no vector search) to
ground an LLM's answers in the company knowledge base.

Run directly to launch a Gradio chat UI in the browser.
"""

import glob
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
KNOWLEDGE_BASE = Path(__file__).parent / "knowledge-base"

SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""

openai = OpenAI()


def load_knowledge() -> dict[str, str]:
    """Load every employee and product document into a dict keyed by name."""
    knowledge: dict[str, str] = {}

    for filename in glob.glob(str(KNOWLEDGE_BASE / "employees" / "*")):
        # Employees are keyed by last name (e.g. "Avery Lancaster" -> "lancaster")
        name = Path(filename).stem.split(" ")[-1]
        knowledge[name.lower()] = Path(filename).read_text(encoding="utf-8")

    for filename in glob.glob(str(KNOWLEDGE_BASE / "products" / "*")):
        # Products are keyed by their full filename stem (e.g. "carllm")
        name = Path(filename).stem
        knowledge[name.lower()] = Path(filename).read_text(encoding="utf-8")

    return knowledge


knowledge = load_knowledge()


def get_relevant_context(message: str) -> list[str]:
    """Naive keyword lookup: strip punctuation, lowercase, and match words against knowledge keys."""
    text = "".join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]


def additional_context(message: str) -> str:
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        return "There is no additional context relevant to the user's question."
    return "The following additional context might be relevant in answering the user's question:\n\n" + "\n\n".join(
        relevant_context
    )


def chat(message: str, history: list[dict]) -> str:
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def main():
    gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


if __name__ == "__main__":
    main()
