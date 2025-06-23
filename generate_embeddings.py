
import openai
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

openai.api_key = os.getenv("OPENAI_API_KEY") or "sk-xxxxx"
df = pd.read_csv("cleaned_whatsapp_pairs.csv")
texts = df["customer_message"].astype(str).tolist()
embeddings = []
batch_size = 20

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    try:
        response = openai.Embedding.create(input=batch, model="text-embedding-3-small")
        embeddings.extend([r["embedding"] for r in response["data"]])
    except Exception as e:
        print(f"Error in batch {i}: {e}")
        embeddings.extend([[0.0]*1536]*len(batch))

with open("embeddings.pkl", "wb") as f:
    pickle.dump(np.array(embeddings), f)
