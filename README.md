# Finance Insights App (Module 04: AI in Finance)

A small proof-of-concept web app that analyzes finance news text. A user pastes an article or headline and gets:

1. **Sentiment classification** (positive / negative / neutral) from a traditional ML model (TF-IDF + Naive Bayes, scikit-learn)
2. **A short summary** from a local LLM (`llama3.2:1b` served by Ollama)
3. **Key takeaways** (bullet points on main points and risks) from the LLM
4. **Q&A**: an optional question about the text, answered by the LLM

> For informational and educational purposes only. Not financial advice.

## Project structure

```
.
├── app.py                # Flask app: sentiment + Ollama LLM integration
├── model_training.py     # Text cleaning, training, and sentiment inference
├── sample_data.csv       # 76 labeled finance sentences (text,label)
├── templates/
│   └── index.html        # Minimal front-end
├── requirements.txt
└── README.md
```

## Setup and run

Requirements: Python 3.9+, and [Ollama](https://ollama.com) running locally with the `llama3.2:1b` model.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running and the model is available
ollama pull llama3.2:1b
ollama list                      # should show llama3.2:1b
# Ollama listens on http://localhost:11434 by default

# 3. Train the sentiment model (creates sentiment_model.pkl and vectorizer.pkl)
python model_training.py

# 4. Start the Flask app
python app.py
```

Open http://127.0.0.1:5000 in your browser.

To use a different model or port, set environment variables before running `app.py`:

```bash
OLLAMA_MODEL=llama3.2:3b OLLAMA_URL=http://localhost:11434/api/generate python app.py
```

## How to test

1. Click one of the **Try a sample** buttons (or paste your own finance text).
2. Optionally type a question, such as "What is the overall outlook for the company?".
3. Click **Analyze**. You should see the sentiment label with a confidence score, an LLM summary, key takeaways, and (if you asked one) an answer.
4. If Ollama is not running or crashes (for example, from low memory), the app prints a warning in the terminal and falls back to simple extractive versions: the first two sentences as a summary, the first three sentences as takeaways, and the sentence that best matches your question. These are labeled "LLM unavailable" so they are not mistaken for LLM output.

## How it works

1. **Data preprocessing** (`clean_text`): removes HTML tags and URLs, strips punctuation, lowercases, and normalizes whitespace.
2. **Traditional ML sentiment** (`model_training.py`): TF-IDF features (unigrams and bigrams) feed a Multinomial Naive Bayes classifier. The data is split 75/25 (stratified) and the script prints accuracy and a classification report.
3. **LLM integration** (`app.py`): the article text is truncated to a safe prompt length and sent to Ollama's `/api/generate` endpoint with a finance-analyst role prompt that forbids investment advice. Three prompts are used: summary, key takeaways, and Q&A.
4. **Front-end** (`templates/index.html`): a two-pane, responsive interface (HTML, CSS, and a little JavaScript) with the article on the left and a results panel on the right. It shows a large sentiment verdict with a probability bar for negative, neutral, and positive, then the summary, key takeaways, and an answer to the optional question. It supports sample articles, a live word count, a loading state, light and dark modes, and keyboard focus styles.

## Business use case

**Scenario:** A financial analyst or a news aggregator has to scan hundreds of headlines and articles each day.

**Value proposition:** The app tags each item's sentiment, produces a short summary, and answers targeted questions about the text. This helps the analyst triage which stories deserve a full read and gives a quick first view for investment research, risk monitoring, or newsletter generation. Running the LLM locally keeps sensitive text on the user's own machine.

**Limitations:**
- The sentiment model is trained on a very small dataset (76 sentences), so its accuracy is modest (roughly 60% in cross-validation) and it is best read as a demonstration of the pipeline, not a reliable signal. Confidence scores are low for the same reason.
- `llama3.2:1b` is a small model, so summaries and answers can be vague, miss details, or be wrong. Outputs should be checked against the source text.
- Both the data and the LLM can carry bias. The tool does not replace professional financial advice and should not be used to make trading decisions.

## Possible improvements

- Train on a larger public financial news dataset (for example, Financial PhraseBank headlines) to improve accuracy.
- Add topic or sector classification (Banking, Tech Stocks, Energy) with a second classifier.
- Compare against a rule-based baseline such as VADER.
- Add real-time headlines from a news API and a Streamlit or Plotly Dash dashboard.
