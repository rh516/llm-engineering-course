# Week 1 LLM Engineering - Standalone Scripts

Standalone recreation of the Week 1 notebook exercises as plain Python scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts

- `scraper.py` - shared BeautifulSoup helpers for fetching page text/links
- `day1_summarizer.py` - summarize a website via the OpenAI Chat Completions API
- `day2_apis.py` - compares raw HTTP calls, the OpenAI client, and OpenAI-compatible endpoints (Gemini, Ollama)
- `day4_tokens_and_memory.py` - tokenization with tiktoken, and a demo of why LLM "memory" is an illusion
- `day5_brochure.py` - generates a company brochure by scraping a site, picking relevant links via LLM, then summarizing
- `exercise_tech_explainer.py` - week 1 homework: explain a technical question using both GPT and a local Ollama model

Run any script directly, e.g.:

```bash
python day1_summarizer.py
```

Ollama-based scripts require `ollama serve` running locally with the `llama3.2` model pulled (`ollama pull llama3.2`).
