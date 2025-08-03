import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch, faiss, open_clip
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Photo Search")
st.title("Photo Search with MobileCLIP")

# HF dataset config
DATASET_NAME = "devanshu125/unsplash-lite-small"

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
def load_embeddings_and_ids():
    index_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename="data/faiss_data/photo_embeddings.index",
        repo_type="dataset"
    )
    index = faiss.read_index(index_path)

    ids_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename="data/faiss_data/photo_ids.parquet",
        repo_type="dataset"
    )
    photo_ids = pd.read_parquet(ids_path)["photo_id"].tolist()

    return index, photo_ids

def get_image_from_hf(photo_id):
    try:
        img_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=f"data/images/{photo_id}.jpg",
            repo_type="dataset"
        )
        return Image.open(img_path)
    except Exception as e:
        st.error(f"Error loading image {photo_id}: {e}")
        return None

index, photo_ids = load_embeddings_and_ids()

query = st.text_input("Enter your search query:")
k = st.slider("Results to show", 1, min(10, len(photo_ids)), 3, 1)
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
            img = get_image_from_hf(photo_id)
            if img:
                st.image(img, caption=f"Photo ID: {photo_id}, Score: {D[0][i]:.4f}")
            else:
                st.write(f"Photo ID: {photo_id}, Score: {D[0][i]:.4f} - Image not available")

if st.sidebar.button("Show Dataset Info"):
    st.sidebar.write(f"Dataset: {DATASET_NAME}")
    st.sidebar.write(f"Total photos: {len(photo_ids)}")
    st.sidebar.write(f"FAISS index dimension: {index.d}")
    st.sidebar.write(f"Index type: {type(index).__name__}")