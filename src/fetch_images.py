from PIL import Image
import pandas as pd
import os
import requests
from io import BytesIO

def save_one_image(row):
    req_img_save_path = f"../data/images/{row['photo_id']}.jpg"
    if os.path.exists(req_img_save_path):
        return True # already exists
    
    try:
        r = requests.get(row['photo_image_url'], timeout=10)
        r.raise_for_status()  # Raise an error for bad responses
        img = Image.open(BytesIO(r.content))
        img.save(req_img_save_path, format='JPEG')
        return True
    except Exception as e:
        print(f"Error saving image {row['photo_id']}: {e}")
        return False
    
def fetch_images_from_file(file_path, debug=False, req_sample_photos=None):
    df = pd.read_parquet(file_path)
    all_rows = df[['photo_id', 'photo_image_url']].drop_duplicates()
    if debug:
        print(f"IN DEBUG MODE. TAKING ONLY {req_sample_photos} SAMPLES")
        sample_rows = all_rows.sample(n=req_sample_photos, random_state=42)
        all_rows = sample_rows
    
    print(f"Total images to save: {len(all_rows)}")
    all_rows = all_rows.reset_index(drop=True)
    for index, row in all_rows.iterrows():
        if index % 100 == 0:
            print(f"Processing row {index + 1}/{len(all_rows)}: {row['photo_id']}")

        if not save_one_image(row):
            print(f"Failed to save image for row {index}")

    print("Image fetching completed.")

if __name__ == "__main__":

    req_file_path = "../data/metadata/photos.parquet"
    fetch_images_from_file(req_file_path, debug=True, req_sample_photos=1000)