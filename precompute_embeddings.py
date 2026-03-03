import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

print("="*60)
print("PRE-COMPUTING EMBEDDINGS - Run this ONCE locally")
print("="*60)

# Load model
print("\n🔄 Loading model...")
model = SentenceTransformer('paraphrase-albert-small-v2')

# Load catalog
print("🔄 Loading catalog...")
df = pd.read_csv('data/shl_catalog_final.csv')
df['description'] = df['description'].fillna('')
df['combined_text'] = df['name'] + " " + df['description'] + " " + df['test_type']
print(f"✅ Loaded {len(df)} assessments")

# Generate embeddings
print("\n🔄 Generating embeddings (this takes 2-3 minutes)...")
embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

# Save embeddings
print("🔄 Saving embeddings...")
with open('data/embeddings.pkl', 'wb') as f:
    pickle.dump({
        'embeddings': embeddings,
        'texts': df['combined_text'].tolist()
    }, f)

# Check file size
size_mb = os.path.getsize('data/embeddings.pkl') / (1024*1024)
print(f"✅ Saved! File size: {size_mb:.2f} MB")
print("\n🎉 DONE! Now commit this file to GitHub.")