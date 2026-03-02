import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Any
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json

# Force CPU mode to prevent GPU memory usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self.df = pd.read_csv(catalog_path)
        self.df['description'] = self.df['description'].fillna('')
        self.df['combined_text'] = self.df['name'] + " " + self.df['description'] + " " + self.df['test_type']
        
        # Initialize BM25 for keyword search
        tokenized_corpus = [doc.lower().split() for doc in self.df['combined_text'].tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize Embeddings - USING SMALLER MODEL FOR RENDER FREE TIER
        # Changed from 'all-MiniLM-L6-v2' (80MB) to 'paraphrase-albert-small-v2' (43MB)
        logger.info("Loading smaller model to fit in 512MB memory limit...")
        self.embed_model = SentenceTransformer('paraphrase-albert-small-v2')
        
        # Process in smaller batches to reduce memory spike
        batch_size = 32
        all_embeddings = []
        texts = self.df['combined_text'].tolist()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embed_model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        self.embeddings = np.vstack(all_embeddings)
        
        # Initialize FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        
        logger.info(f"Engine initialized with {len(self.df)} records using {self.embeddings.nbytes / 1024 / 1024:.2f}MB for embeddings.")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        # 1. Semantic Search (Dense)
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # 2. Keyword Search (Sparse)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 3. Hybrid Reranking (Simple Weighted Average)
        combined_scores = {}
        
        # Normalize FAISS distances (lower is better, convert to score)
        max_dist = np.max(distances) if np.max(distances) > 0 else 1
        for i, idx in enumerate(indices[0]):
            score = 1 - (distances[0][i] / max_dist)
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.5
            
        # Normalize BM25 scores
        max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
        for idx in bm25_indices:
            score = bm25_scores[idx] / max_bm25
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.5
            
        # Sort and get top results
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in sorted_indices[:top_k]:
            item = self.df.iloc[idx].to_dict()
            item['score'] = float(score)
            results.append(item)
            
        return results

    def get_balanced_recommendations(self, query: str, top_k: int = 5, api_key: str = None) -> List[Dict[str, Any]]:
        """
        Uses LLM to rerank and balance recommendations (Hard vs Soft skills).
        """
        search_k = max(top_k * 3, 15) # Search more to allow for reranking
        raw_results = self.search(query, top_k=search_k)
        
        if not api_key:
             # Fallback to simple top_k if no LLM key provided
             return raw_results[:top_k]

        # Use Gemini to rerank for balance
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        context = "\n".join([
            f"- Name: {r['name']}, URL: {r['url']}, Type: {r['test_type']}, Desc: {r['description'][:200]}..."
            for r in raw_results
        ])
        
        prompt = f"""
        You are an expert recruiter recommending SHL assessments.
        Query: {query}
        
        Candidates:
        {context}
        
        Select the top {top_k} most relevant assessments. 
        CRITICAL: If the query spans multiple domains (e.g., technical and behavioral), ensure a balanced mix.
        Test types: K (Knowledge/Skills), P (Personality/Behavior), A (Ability/Aptitude), B (Biodata), C (Competencies), E (Exercises), S (Simulations).
        
        Return ONLY a JSON list of objects with the 'url' field from the original list, in order of relevance.
        Example: ["url1", "url2", ...]
        """
        
        try:
            response = model.generate_content(prompt)
            # Clean response text and parse JSON
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            recommended_urls = json.loads(text)
            
            # Map back to full records
            final_recs = []
            for url in recommended_urls:
                match = self.df[self.df['url'] == url]
                if not match.empty:
                    final_recs.append(match.iloc[0].to_dict())
            
            return final_recs[:top_k]
        except Exception as e:
            logger.error(f"LLM Reranking failed: {e}")
            return raw_results[:top_k]

if __name__ == "__main__":
    engine = RecommendationEngine("data/shl_catalog_final.csv")
    recs = engine.get_balanced_recommendations("Java developer with good communication skills")
    for r in recs:
        print(f"{r['name']} ({r['test_type']}) - {r['url']}")