
import streamlit as st
import pandas as pd
import openai
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="AI Reply Assistant", layout="wide")
st.title("🤖 AI Reply Assistant")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_whatsapp_pairs.csv")

@st.cache_resource
def load_embeddings():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)

data = load_data()
embeddings = load_embeddings()

openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

user_input = st.text_area("💬 Enter a customer message:")

def generate_reply(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful sales assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

if st.button("🔍 Find Similar Replies"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        response = openai.Embedding.create(input=[user_input], model="text-embedding-3-small")
        input_vec = np.array(response.data[0].embedding).reshape(1, -1)
        scores = cosine_similarity(input_vec, embeddings)[0]
        top_indices = scores.argsort()[-3:][::-1]
        st.subheader("🧠 Similar Past Replies")
        for i, idx in enumerate(top_indices):
            row = data.iloc[idx]
            st.markdown(f"**#{i+1}** — Similarity Score: `{scores[idx]:.2f}`")
            st.info(row['sales_reply'])

        if st.button("✍️ Generate AI Reply"):
            reply = generate_reply(user_input)
            st.success(reply)
