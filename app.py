# app.py
"""Financial News Insight app: ML sentiment + local LLM (Ollama) summary, Q&A and insights."""
import os
import re

import requests
from flask import Flask, render_template, request

from model_training import load_sentiment_pipeline

app = Flask(__name__)

# 1. Trained sentiment pipeline (run model_training.py first)
predict_sentiment = load_sentiment_pipeline("sentiment_model.pkl", "vectorizer.pkl")

# 2. Local LLM served by Ollama (override with environment variables if needed)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

SYSTEM_ROLE = (
    "You are a careful financial news analyst. Use only the text provided. "
    "Be concise. You are not a financial advisor and must not give investment advice."
)
MAX_CHARS = 6000  # keep the prompt small enough for a 1B model


def ask_llm(task_prompt, text):
    """Send a prompt to the local Ollama model and return its text answer."""
    prompt = f"{SYSTEM_ROLE}\n\nArticle:\n{text[:MAX_CHARS]}\n\n{task_prompt}"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "").strip() or "[No response returned]"
        return f"[Error] Ollama responded with status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"[Error] Could not reach Ollama at {OLLAMA_URL}: {e}"


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def words_of(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# Fallbacks: simple extractive versions used only when the local LLM cannot respond,
# so the app still returns something useful. Each result is labeled as a fallback.
def fallback_summary(text):
    return "Basic summary (LLM unavailable): " + " ".join(split_sentences(text)[:2])


def fallback_insights(text):
    return "\n".join("- " + s for s in split_sentences(text)[:3])


def fallback_answer(text, question):
    sentences = split_sentences(text)
    if not sentences:
        return "[No text to answer from]"
    wanted = words_of(question)
    best = max(sentences, key=lambda s: len(wanted & words_of(s)))
    return "Closest sentence in the article (LLM unavailable): " + best


def get_summary(text):
    return ask_llm("Summarize this article in 2-3 sentences.", text)


def get_insights(text):
    return ask_llm("List 3 short bullet points with the key takeaways and risks mentioned.", text)


def get_answer(text, question):
    return ask_llm(f"Question: {question}\nAnswer briefly using only the article.", text)


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "user_text": "", "question": "", "sentiment": None, "confidence": None,
        "scores": [], "summary": None, "insights": None, "answer": None,
        "model_name": OLLAMA_MODEL,
    }

    if request.method == "POST":
        text = request.form.get("user_text", "").strip()
        question = request.form.get("question", "").strip()
        context["user_text"], context["question"] = text, question

        if text:
            # Step 1: traditional ML sentiment classification
            label, confidence, scores = predict_sentiment(text)
            context["sentiment"], context["confidence"] = label, round(confidence * 100)
            context["scores"] = [
                {"name": name, "pct": round(scores.get(name, 0) * 100)}
                for name in ("negative", "neutral", "positive")
            ]
            # Step 2: LLM generation
            summary = get_summary(text)
            if summary.startswith("[Error]"):
                # The local LLM is not responding: skip further LLM calls and use fallbacks
                print(f"[WARN] LLM unavailable, using fallback. Details: {summary}")
                context["summary"] = fallback_summary(text)
                context["insights"] = fallback_insights(text)
                if question:
                    context["answer"] = fallback_answer(text, question)
            else:
                context["summary"] = summary
                context["insights"] = get_insights(text)
                if question:
                    context["answer"] = get_answer(text, question)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
