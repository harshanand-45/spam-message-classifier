"""
predict.py

Loads the saved model + vectorizer and gives a simple function to
classify a new message. Also runnable directly for a quick sanity check.

Run from the project root:
    python src/predict.py
"""

import os
import joblib

from preprocessing import clean_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_classifier.pkl")

_bundle = None


def _load_model():
    """Loads the model bundle once and caches it (so Streamlit doesn't reload it every click)."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run 'python src/train.py' first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_message(message):
    """
    Takes a raw message string and returns a dict with the prediction
    label ('spam' or 'ham') and the model's confidence for that label.
    """
    bundle = _load_model()
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    cleaned = clean_text(message)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]  # [prob_ham, prob_spam]

    label = "spam" if prediction == 1 else "ham"
    confidence = probabilities[1] if prediction == 1 else probabilities[0]

    return {
        "label": label,
        "confidence": float(confidence),
        "prob_ham": float(probabilities[0]),
        "prob_spam": float(probabilities[1]),
    }


if __name__ == "__main__":
    test_messages = [
        "Congratulations! You have won a free prize. Call now!",
        "Hey, are we meeting at 6 pm today?",
        "URGENT! You have been selected for a cash reward. Click this link now.",
        "Can you send me the notes from today's class?",
    ]

    for msg in test_messages:
        result = predict_message(msg)
        print(f"Message : {msg}")
        print(f"Predicted: {result['label']}  (confidence: {result['confidence']*100:.1f}%)")
        print("-" * 60)
