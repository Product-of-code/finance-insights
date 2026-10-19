# model_training.py
"""Data preprocessing + traditional ML sentiment classifier (TF-IDF + Naive Bayes)."""
import re
import string

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def clean_text(text):
    """Basic text cleaning: strip HTML tags and URLs, drop punctuation, lowercase."""
    text = re.sub(r"<[^>]+>", " ", text)                       # HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)        # URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_and_preprocess_data(csv_path):
    """Load a CSV with 'text' and 'label' columns; return cleaned text (X) and labels (y)."""
    df = pd.read_csv(csv_path)
    df["cleaned_text"] = df["text"].apply(clean_text)
    return df["cleaned_text"], df["label"]


def train_sentiment_model(csv_path="sample_data.csv",
                          model_out="sentiment_model.pkl",
                          vectorizer_out="vectorizer.pkl"):
    """Train a Naive Bayes classifier and save the model + TF-IDF vectorizer."""
    X, y = load_and_preprocess_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print("Accuracy on test set: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(clf, model_out)
    joblib.dump(vectorizer, vectorizer_out)
    print(f"[INFO] Model saved to {model_out}")
    print(f"[INFO] Vectorizer saved to {vectorizer_out}")


def load_sentiment_pipeline(model_path="sentiment_model.pkl",
                            vectorizer_path="vectorizer.pkl"):
    """Load the saved model and return predict_sentiment(text) -> (label, confidence, probabilities)."""
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    def predict_sentiment(text):
        features = vectorizer.transform([clean_text(text)])
        probabilities = clf.predict_proba(features)[0]
        best = probabilities.argmax()
        scores = {str(c): float(p) for c, p in zip(clf.classes_, probabilities)}
        return str(clf.classes_[best]), float(probabilities[best]), scores

    return predict_sentiment


if __name__ == "__main__":
    train_sentiment_model()
