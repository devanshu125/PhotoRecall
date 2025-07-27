import pandas as pd
import numpy as np
import torch
import faiss
import open_clip
from torchvision import transforms
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'MobileCLIP-S1', 
    pretrained="datacompdr"
)
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
model = model.to(device).eval()

keywords_df = pd.read_parquet("../data/metadata/keywords.parquet")
photos_df = pd.read_parquet("../data/metadata/photos.parquet")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4815, 0.4578, 0.4082),
        std=(0.2686, 0.2613, 0.2758)),
])

def encode_text(texts):
    with torch.no_grad():
        text_tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype("float32")

def encode_images(req_path):
    image = Image.open(req_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype("float32")

# for faster processing, use only images that exist
existing_images = set(os.listdir("../data/images/"))
existing_images = {img.split('.')[0] for img in existing_images if img.endswith('.jpg')}
photos_df = photos_df[photos_df["photo_id"].isin(existing_images)]
print(f"Total photos to process: {len(photos_df)}")

# pre-compute keyword vectors only for existing photos
keywords_df = keywords_df[keywords_df["photo_id"].isin(photos_df["photo_id"].unique())]
print(f"Total keywords to process: {len(keywords_df)}")
print("Embedding keywords...")
unq_keywords = keywords_df["keyword"].unique().tolist()
print(f"Unique keywords: {len(unq_keywords)}")
kw_vecs = encode_text(unq_keywords)
print("Keyword embeddings shape:", kw_vecs.shape)

# aggregate per photo
weights = {"img": 0.5, "kw": 0.5}

photo_vecs = []
photo_ids = []
kw_lookup = keywords_df.groupby("photo_id")

photos_df = photos_df.reset_index(drop=True)
for idx, row in photos_df.iterrows():

    if idx % 100 == 0:
        print(f"Processing photo {idx + 1}/{len(photos_df)}: {row['photo_id']}")

    photo_id = row["photo_id"]

    photo_ids.append(photo_id)
    
    # encode image
    img_vec = encode_images(f"../data/images/{photo_id}.jpg")

    # encode keywords
    if photo_id in kw_lookup.groups:
        kw_list = kw_lookup.get_group(photo_id)["keyword"].tolist()
        if kw_list:
            kw_vec = encode_text(kw_list)
            kw_vec = np.mean(kw_vec, axis=0, keepdims=True)
        else:
            embedding_dim = img_vec.shape[-1]
            kw_vec = np.zeros((1, embedding_dim), dtype="float32")
    else:
        embedding_dim = img_vec.shape[-1]
        kw_vec = np.zeros((1, embedding_dim), dtype="float32")

    # blend vectors
    blended_vec = (weights["img"] * img_vec + weights["kw"] * kw_vec).flatten()
    photo_vecs.append(blended_vec)

photo_vecs = np.vstack(photo_vecs).astype("float32")
print("Blended embeddings shape:", photo_vecs.shape)

# sample photo vectors
print(photo_vecs[:5])

# write in faiss index
index = faiss.IndexFlatIP(photo_vecs.shape[1])
index.add(photo_vecs)
faiss.write_index(index, "../data/embeddings/photo_embeddings.index")

# save photo ids
photo_ids_df = pd.DataFrame({"photo_id": photo_ids})
photo_ids_df.to_parquet("../data/embeddings/photo_ids.parquet", index=False)