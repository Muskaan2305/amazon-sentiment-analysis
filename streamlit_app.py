import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# UI
st.title("🛍️ Amazon Review Sentiment Analyzer")
st.write("Type a product review below and find out if it's **Positive** or **Negative**!")

review = st.text_area("Enter your review here:", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.success(f"✅ Positive Review! (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.error(f"❌ Negative Review! (Confidence: {probability[0]*100:.1f}%)")