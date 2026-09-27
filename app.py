"""
app.py

Streamlit app for the spam message classifier.
Run with: streamlit run app.py
"""

import sys
import os
import pandas as pd
import streamlit as st

# so we can import from src/ regardless of where streamlit is launched from
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_message

st.set_page_config(page_title="Spam Message Classifier", page_icon="📩")

st.title("📩 Spam Message Classifier")
st.write("Check whether a message is Spam or Not Spam")

st.divider()

message = st.text_area("Enter a message:", height=150, placeholder="Paste an SMS, email, or support message here...")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        try:
            result = predict_message(message)
            confidence_pct = result["confidence"] * 100

            if result["label"] == "spam":
                st.error(f"🚨 Spam  (confidence: {confidence_pct:.1f}%)")
            else:
                st.success(f"✅ Not Spam  (confidence: {confidence_pct:.1f}%)")

            st.progress(result["prob_spam"])
            st.caption(f"Probability of spam: {result['prob_spam']*100:.1f}% | Probability of not spam: {result['prob_ham']*100:.1f}%")

        except FileNotFoundError as e:
            st.error(str(e))

st.caption("Note: this is a machine learning prediction based on a Logistic Regression model, "
           "not a certainty. It can occasionally be wrong.")

st.divider()

# --- optional batch prediction from CSV ---
st.subheader("Batch Prediction (CSV)")
st.write("Upload a CSV file with a column named `message` to classify multiple messages at once.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        if "message" not in batch_df.columns:
            st.error("CSV must contain a column named 'message'.")
        else:
            labels = []
            confidences = []
            for msg in batch_df["message"]:
                result = predict_message(str(msg))
                labels.append(result["label"])
                confidences.append(round(result["confidence"] * 100, 1))

            batch_df["prediction"] = labels
            batch_df["confidence (%)"] = confidences

            st.write(batch_df)

            csv_output = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv_output,
                file_name="spam_predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Could not process file: {e}")
