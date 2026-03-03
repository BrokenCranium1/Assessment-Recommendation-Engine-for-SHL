# precompute_embeddings.py
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

print("="*60)
print("PRE-COMPUTING EMBEDDINGS - Run this ONCE locally")
print("="*60)

# Check if file already exists
if os.path.exists('data/embeddings.pkl'):
    response = input("⚠️ embeddings.pkl already exists. Overwrite? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        exit()

print("\n🔄 Loading model...")
model = SentenceTransformer('paraphrase-albert-small-v2')

print("\n🔄 Loading catalog...")
df = pd.read_csv('data/shl_catalog_final.csv')
df['description'] = df['description'].fillna('')
df['combined_text'] = df['name'] + " " + df['description'] + " " + df['test_type']
print(f"✅ Loaded {len(df)} assessments")

print("\n🔄 Generating embeddings (this takes 2-3 minutes)...")
embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

print("\n🔄 Saving embeddings to data/embeddings.pkl...")
with open('data/embeddings.pkl', 'wb') as f:
    pickle.dump({
        'embeddings': embeddings,
        'texts': df['combined_text'].tolist(),
        'model_name': 'paraphrase-albert-small-v2'
    }, f)

# Check file size
file_size = os.path.getsize('data/embeddings.pkl') / (1024*1024)
print(f"✅ Saved! File size: {file_size:.2f} MB")

print("\n🎉 DONE! Now commit this file to GitHub and redeploy.")