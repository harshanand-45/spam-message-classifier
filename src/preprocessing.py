"""
preprocessing.py

Cleans up raw SMS/email text before it goes into the TF-IDF vectorizer.
Steps: lowercase -> remove urls/html -> remove numbers & special chars ->
tokenize -> remove stopwords -> lemmatize.

Run this file directly once to make sure the NLTK downloads are in place:
    python src/preprocessing.py
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_data():
    """Downloads the NLTK resources we need, but only if they aren't already there."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


download_nltk_data()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Takes a raw message and returns a cleaned, lemmatized string
    ready to be fed into TfidfVectorizer.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove URLs (http/https/www links)
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove HTML tags, just in case (some "SMS" datasets sneak these in)
    text = re.sub(r"<.*?>", " ", text)

    # remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # remove numbers (phone numbers, prices etc. don't help much for a simple model)
    text = re.sub(r"\d+", " ", text)

    # keep only letters, drop punctuation/special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords and very short tokens, then lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned_tokens)


if __name__ == "__main__":
    # quick manual test
    sample = "Congratulations!! You have WON a FREE prize. Call 09061701461 now http://bit.ly/claim"
    print("before:", sample)
    print("after :", clean_text(sample))
