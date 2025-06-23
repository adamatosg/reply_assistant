import streamlit as st
import pandas as pd
import openai
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🤖 AI Reply Assistant")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_whatsapp_pairs.csv")

# --- Load embeddings ---
@st.cache_resource
def load_embeddings():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)

# --- Compute embedding for input ---
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

# --- Similarity search ---
def find_similar_replies(user_input, embeddings, data, top_n=3):
    user_embedding = get_embedding(user_input)
    if user_embedding is None:
        return []

    similarity_scores = cosine_similarity([user_embedding], embeddings)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    return [(data.iloc[i]["response"], similarity_scores[i]) for i in top_indices]

# --- Load data ---
data = load_data()
embeddings = load_embeddings()

# --- UI ---
st.title("🤖 AI Reply Assistant")
user_input = st.text_area("💬 Enter a customer message:")

if st.button("🔍 Find Similar Replies"):
    if not user_input.strip():
        st.warning("Please enter a customer message.")
    else:
        st.subheader("🧠 Matched historical replies:")
        matches = find_similar_replies(user_input, embeddings, data)
        if not matches:
            st.info("No relevant replies found.")
        else:
            for i, (reply, score) in enumerate(matches, 1):
                st.markdown(f"**#{i} — Similarity Score:** `{score:.2f}`")
                st.code(reply, language="text")

if st.button("✍️ Generate AI Reply"):
    try:
        system_prompt = "You are a helpful assistant replying to customer WhatsApp inquiries. Respond politely and informatively."
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        reply = completion["choices"][0]["message"]["content"]
        st.subheader("💡 Suggested AI Reply:")
        st.success(reply)
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
