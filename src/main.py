import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch, faiss, open_clip
import os

st.set_page_config(page_title="Photo Search")
st.title("Photo Search with MobileCLIP")

@st.cache_resource(show_spinner=True)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        'MobileCLIP-S1', 
        pretrained="datacompdr"
    )
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
    model = model.to(device).eval()
    return model, tokenizer, preprocess, device

model, tokenizer, preprocess, device = load_model()

def encode_query(text: str) -> np.ndarray:
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)  # Use the stored device
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype("float32")

@st.cache_resource(show_spinner=True)
def load_embeddings():
    index = faiss.read_index("../data/embeddings/photo_embeddings.index")
    photo_ids = pd.read_parquet("../data/embeddings/photo_ids.parquet")["photo_id"].tolist()
    return index, photo_ids

index, photo_ids = load_embeddings()

query = st.text_input("Enter your search query:")
k = st.slider("Results to show", 1, min(10, len(photo_ids)), 6, 1)
submit_button = st.button("Search")

if submit_button and query:
    query_vec = encode_query(query)
    D, I = index.search(query_vec, k)
    
    st.write(f"Top {k} results for query: **{query}**")
    
    if len(I[0]) == 0:
        st.write("No results found.")
    else:
        for i in range(k):
            photo_id = photo_ids[I[0][i]]
            img_path = f"../data/images/{photo_id}.jpg"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=f"Photo ID: {photo_id}, Score: {D[0][i]:.4f}")
            else:
                st.write(f"Image for Photo ID {photo_id} not found.")