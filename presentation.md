# Spam Message Classifier - 5 Minute Presentation

## Slide 1 - Project Title
**Spam Message Classifier**
An NLP project that classifies SMS/email messages as Spam or Not Spam.

## Slide 2 - Problem Statement
- Spam messages are annoying and sometimes dangerous (phishing, scams).
- Goal: build a model that automatically flags a message as spam or not,
  so it can be filtered before reaching a human.
- Binary classification problem: Spam vs Ham.

## Slide 3 - Dataset
- SMS Spam Collection dataset (UCI/Kaggle)
- 5,572 labeled SMS messages
- Class distribution: 4,825 ham / 747 spam (imbalanced towards ham)

## Slide 4 - NLP Preprocessing
- Lowercase text
- Remove URLs, HTML tags, email addresses, numbers, punctuation
- Tokenize with NLTK
- Remove stopwords
- Lemmatize words to their root form
- Example: "Congratulations!! You WON a FREE prize" -> "congratulation free prize"

## Slide 5 - TF-IDF + Logistic Regression
- TF-IDF converts cleaned text into numeric feature vectors (unigrams + bigrams, top 3000 features)
- Logistic Regression trained on these features
- Used `class_weight='balanced'` since the dataset is imbalanced
- Also trained Multinomial Naive Bayes as a comparison model

## Slide 6 - Evaluation Results
Logistic Regression (final model):
- Accuracy: 97.48%
- Precision: 90.07%
- Recall: 91.28%
- F1-score: 90.67%

Naive Bayes had slightly higher accuracy (97.66%) and perfect precision, but
lower recall (82.55%) - it missed more actual spam. Logistic Regression was
chosen as the final model because catching more spam matters more here.

## Slide 7 - Streamlit Application
- Simple web app: paste a message, click Predict
- Shows Spam/Not Spam result with a confidence score
- Color-coded output (red for spam, green for not spam)
- Optional CSV upload for batch predictions

## Slide 8 - Sample Predictions
- "Congratulations! You have won a free prize. Call now!" -> Spam (91.6%)
- "Hey, are we meeting at 6 pm today?" -> Not Spam (87.4%)
- "URGENT! You have been selected for a cash reward. Click this link now." -> Spam (85.5%)
- "Can you send me the notes from today's class?" -> Not Spam (83.9%)

## Slide 9 - Challenges/Learnings
- Handling class imbalance (much more ham than spam)
- Deciding how much to clean the text without losing useful signal
  (e.g. keeping bigrams helped catch phrases like "free prize")
- Comparing two models and picking one based on which metric actually
  matters for the use case, not just accuracy

## Slide 10 - Future Improvements
- Train on a larger/more varied dataset (e.g. add email spam data)
- Try a transformer-based model (DistilBERT) for comparison
- Show which words drove a prediction, for more transparency
- Add automated tests for the preprocessing pipeline
