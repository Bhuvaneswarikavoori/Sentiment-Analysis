from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st

# Custom styling
st.markdown(
    """
<style>
body {
    color: #fff;
    background-color: #00ffff; 
    background-size: cover;
}
.custom-font {
    font-family: 'Courier New', Courier, monospace;
}
</style>
""",
    unsafe_allow_html=True,
)

# Define a function to predict sentiment
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).tolist()[0]

    sentiment = "positive" if probs[1] > probs[0] else "negative"
    confidence = max(probs)

    return sentiment, confidence

# Streamlit web app
st.title("SentimentScope: Sentiment Analysis for Reviews")
st.write("Enter a review or a batch of reviews (separated by line breaks) and get the sentiment summary.", className="custom-font")

# Input
input_text = st.text_area("Enter the reviews:", className="custom-font")

if input_text:
    reviews = input_text.split("\n")
    positive_count = 0
    negative_count = 0
    total_confidence = 0

    for review in reviews:
        if review.strip():
            sentiment, confidence = predict_sentiment(review)
            if sentiment == "positive":
                positive_count += 1
            else:
                negative_count += 1
            total_confidence += confidence

    # Output
    total_reviews = positive_count + negative_count
    st.write(f"Total Reviews: {total_reviews}", className="custom-font")
    st.write(f"Positive Reviews: {positive_count}", className="custom-font")
    st.write(f"Negative Reviews: {negative_count}", className="custom-font")
    st.write(f"Average Confidence: {total_confidence/total_reviews:.2f}", className="custom-font")

    # Sentiment summary
    if positive_count > negative_count:
        summary = "Overall, the sentiment is positive."
    elif negative_count > positive_count:
        summary = "Overall, the sentiment is negative."
    else:
        summary = "Overall, the sentiment is neutral."

    st.write(summary, className="custom-font")
