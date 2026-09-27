"""
train.py

Loads the SMS spam dataset, cleans the text, trains a TF-IDF + Logistic
Regression model, evaluates it, and saves everything needed for
prediction into a single pickle file.

Run from the project root:
    python src/train.py
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import clean_text

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_classifier.pkl")
CONFUSION_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "confusion_matrix.png")

RANDOM_STATE = 42


def load_dataset(path):
    """
    Loads the SMS spam dataset. Handles a couple of common column-name
    variants since different sources (Kaggle vs UCI mirrors) name the
    columns differently.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at {path}.\n"
            "Download the SMS Spam Collection dataset (UCI/Kaggle) and place "
            "it at data/spam.csv with two columns: 'label' and 'message'."
        )

    # the original UCI file is latin-1 encoded, some Kaggle re-uploads are utf-8
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # some versions of this dataset use v1/v2 as column names, or have
    # extra unnamed columns - clean that up
    cols = {c.lower().strip(): c for c in df.columns}
    if "label" in cols and "message" in cols:
        df = df.rename(columns={cols["label"]: "label", cols["message"]: "message"})
    elif "v1" in cols and "v2" in cols:
        df = df.rename(columns={cols["v1"]: "label", cols["v2"]: "message"})
    else:
        raise ValueError(
            "Couldn't find label/message columns in the dataset. "
            f"Columns found: {list(df.columns)}"
        )

    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)

    return df


def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} messages ({df['label'].value_counts().to_dict()})")

    print("Cleaning text (this can take a few seconds)...")
    df["clean_message"] = df["message"].apply(clean_text)

    # drop rows that became empty after cleaning
    df = df[df["clean_message"].str.strip() != ""].reset_index(drop=True)

    X = df["clean_message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    # also train Naive Bayes just to compare, as the assignment suggested
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    def evaluate(name, clf):
        preds = clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(classification_report(y_test, preds, target_names=["ham", "spam"]))
        return preds, acc, prec, rec, f1

    lr_preds, acc, prec, rec, f1 = evaluate("Logistic Regression", model)
    evaluate("Multinomial Naive Bayes (for comparison)", nb_model)

    # confusion matrix for the model we're actually shipping (Logistic Regression)
    cm = confusion_matrix(y_test, lr_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["ham", "spam"], yticklabels=["ham", "spam"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"\nSaved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    # save vectorizer + model + metrics together so predict.py only needs one file
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            },
        },
        MODEL_PATH,
    )
    print(f"Saved trained model + vectorizer to {MODEL_PATH}")


if __name__ == "__main__":
    main()
