# Spam Message Classifier

## Overview

This is a small NLP project that classifies a text message (SMS/email/support
message) as **Spam** or **Not Spam (Ham)**. It uses classical NLP preprocessing
with NLTK, TF-IDF for feature extraction, and a Logistic Regression model.
There's also a Streamlit app so you can actually try it out on your own messages.

This was built as an intern project to practice a full, small NLP pipeline:
data -> cleaning -> features -> model -> evaluation -> simple web app.

## Features

- Cleans and normalizes raw text (lowercasing, removing URLs/HTML/numbers/punctuation)
- Tokenization, stopword removal and lemmatization with NLTK
- TF-IDF vectorization
- Logistic Regression classifier (with Multinomial Naive Bayes trained alongside for comparison)
- Evaluation with accuracy, precision, recall, F1-score and a confusion matrix
- Streamlit web app to test messages live, with a confidence score
- Optional CSV batch prediction from the app

## Technologies Used

- Python 3
- pandas, numpy
- NLTK (text preprocessing)
- scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, metrics)
- matplotlib, seaborn (confusion matrix plot)
- Streamlit (web app)
- joblib (saving the trained model)

## Dataset

This project uses the **SMS Spam Collection** dataset (5,572 labeled SMS
messages, "ham" or "spam"), originally from UCI and also available on Kaggle.

**Where to put it:** place the CSV at `data/spam.csv` with two columns:
- `label` - either `ham` or `spam`
- `message` - the raw text of the SMS

(If your copy of the dataset uses column names `v1`/`v2` instead, like some
Kaggle re-uploads do, `train.py` will detect and rename them automatically.)

The dataset used for this project is included at `data/spam.csv` and has the
following class distribution:

- ham: 4,825 messages
- spam: 747 messages

So the dataset is noticeably imbalanced towards "ham", which is why
`class_weight='balanced'` is used in the Logistic Regression model and why
precision/recall/F1 matter more than plain accuracy here.

## Project Structure

```
spam-message-classifier/
│
├── data/
│   └── spam.csv               # SMS Spam Collection dataset
│
├── models/
│   ├── spam_classifier.pkl    # trained model + vectorizer (created by train.py)
│   └── confusion_matrix.png   # confusion matrix plot (created by train.py)
│
├── notebooks/
│   └── spam_classifier.ipynb  # exploratory notebook version of the pipeline
│
├── src/
│   ├── preprocessing.py       # text cleaning function
│   ├── train.py                # loads data, trains model, evaluates, saves
│   └── predict.py              # loads saved model, predicts on new text
│
├── app.py                      # Streamlit app
├── requirements.txt
├── README.md
├── .gitignore
└── presentation.md
```

## NLP Preprocessing

The cleaning pipeline (see `src/preprocessing.py`) does the following, in order:

1. Lowercase the text
2. Remove URLs (`http...`, `www...`)
3. Remove HTML tags (just in case any slipped into the data)
4. Remove email addresses
5. Remove numbers
6. Remove special characters/punctuation, keeping only letters
7. Tokenize using NLTK's `word_tokenize`
8. Remove English stopwords (the, is, at, ...) and very short tokens
9. Lemmatize remaining words (e.g. "winning" -> "win")

Example:

```
Input : "Congratulations!! You have WON a FREE prize. Call 09061701461 now http://bit.ly/claim"
Output: "congratulation free prize call"
```

## Model

- **Features:** TF-IDF with unigrams + bigrams, max 3000 features
- **Model:** Logistic Regression (`class_weight='balanced'` to handle the
  ham/spam imbalance, `max_iter=1000` so it actually converges)
- Multinomial Naive Bayes was also trained on the same features for comparison
  (see below). Logistic Regression was picked as the final model because it's
  simpler to explain, and had noticeably better recall on spam, which matters
  more than raw accuracy for a spam filter (you'd rather catch spam than let
  it slip through).

## Training

To retrain the model from scratch:

```bash
python src/train.py
```

This will:
1. Load `data/spam.csv`
2. Clean all the messages
3. Split into train/test (80/20, stratified by label, `random_state=42`)
4. Fit the TF-IDF vectorizer and Logistic Regression model
5. Print evaluation metrics for both models
6. Save a confusion matrix plot to `models/confusion_matrix.png`
7. Save the trained model + vectorizer to `models/spam_classifier.pkl`

## Evaluation

These are the actual results from running `train.py` on the included dataset
(test set of 1,112 messages, 20% split):

**Logistic Regression (final model used in the app)**

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97.48% |
| Precision | 90.07% |
| Recall    | 91.28% |
| F1-score  | 90.67% |

**Multinomial Naive Bayes (for comparison only, not used in the app)**

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97.66% |
| Precision | 100.0% |
| Recall    | 82.55% |
| F1-score  | 90.44% |

Naive Bayes actually gets slightly higher accuracy and perfect precision, but
it misses a lot more real spam messages (lower recall). For a spam filter,
catching more actual spam is usually more useful than avoiding false alarms,
so Logistic Regression was kept as the final model. The confusion matrix for
the Logistic Regression model is saved at `models/confusion_matrix.png` after
running the training script.

## How to Run

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

The app will open in your browser (usually at `http://localhost:8501`).

## Sample Predictions

These are real outputs from the trained model (`python src/predict.py`):

| Message | Prediction | Confidence |
|---|---|---|
| "Congratulations! You have won a free prize. Call now!" | 🚨 Spam | 91.6% |
| "Hey, are we meeting at 6 pm today?" | ✅ Not Spam | 87.4% |
| "URGENT! You have been selected for a cash reward. Click this link now." | 🚨 Spam | 85.5% |
| "Can you send me the notes from today's class?" | ✅ Not Spam | 83.9% |

## Future Improvements

- Try a bigger/more diverse dataset (e.g. combine with the Enron email dataset)
  so the model generalizes better to email-style spam, not just SMS
- Try a fine-tuned transformer model (e.g. DistilBERT) to see if it beats
  TF-IDF + Logistic Regression, at the cost of more complexity
- Add basic handling for emojis and non-English text
- Show which words most influenced a prediction (model coefficients) in the app
- Add simple unit tests for the preprocessing function
