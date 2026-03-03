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
import re

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
        
        # Initialize BM25 for keyword search (lightweight)
        tokenized_corpus = [doc.lower().split() for doc in self.df['combined_text'].tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Keywords for intent detection
        self.technical_keywords = [
            'java', 'python', 'sql', 'javascript', 'coding', 'developer', 
            'programming', 'engineer', 'technical', 'backend', 'frontend',
            'full stack', 'database', 'api', 'algorithm', 'data structure',
            'aws', 'azure', 'cloud', 'devops', 'software', 'code'
        ]
        
        self.behavioral_keywords = [
            'collaborat', 'team', 'communicat', 'lead', 'interpersonal',
            'soft', 'behavior', 'personality', 'attitude', 'cultural',
            'teamwork', 'leadership', 'management', 'emotional', 'empathy',
            'conflict', 'negotiat', 'present', 'verbal', 'written'
        ]
        
        # CRITICAL FIX: Don't load model at startup - lazy load on first request
        self.embed_model = None
        self.embeddings = None
        self.index = None
        self._model_loaded = False
        
        logger.info(f"Engine initialized with {len(self.df)} records. Model will load on first request.")

    def _lazy_load_model(self):
        """Load model and create embeddings only when first needed"""
        if self._model_loaded:
            return
            
        logger.info("🔄 Lazy-loading model on first request (this may take 5-10 seconds)...")
        
        # Load the small model
        self.embed_model = SentenceTransformer('paraphrase-albert-small-v2')
        
        # Process in very small batches to manage memory
        batch_size = 16  # Smaller batches = less memory spike
        all_embeddings = []
        texts = self.df['combined_text'].tolist()
        total_batches = (len(texts) - 1) // batch_size + 1
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embed_model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            logger.info(f"  Processed batch {i//batch_size + 1}/{total_batches}")
        
        self.embeddings = np.vstack(all_embeddings)
        
        # Initialize FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        
        self._model_loaded = True
        memory_used = self.embeddings.nbytes / 1024 / 1024
        logger.info(f"✅ Model loading complete. Embeddings use {memory_used:.2f}MB.")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        # Ensure model is loaded before searching
        self._lazy_load_model()
        
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
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.6  # Semantic weight 0.6
            
        # Normalize BM25 scores
        max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
        for idx in bm25_indices:
            score = bm25_scores[idx] / max_bm25
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.4  # Keyword weight 0.4
            
        # Sort and get top results
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in sorted_indices[:top_k]:
            item = self.df.iloc[idx].to_dict()
            item['score'] = float(score)
            results.append(item)
            
        return results

    def _detect_query_intent(self, query: str) -> Dict[str, bool]:
        """Detect if query needs technical, behavioral, or both"""
        query_lower = query.lower()
        
        # Check for technical keywords
        needs_technical = any(k in query_lower for k in self.technical_keywords)
        
        # Check for behavioral keywords
        needs_behavioral = any(k in query_lower for k in self.behavioral_keywords)
        
        return {
            'technical': needs_technical,
            'behavioral': needs_behavioral,
            'mixed': needs_technical and needs_behavioral
        }

    def _balance_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Balance results between K-type and P-type assessments"""
        
        # Separate results by type
        k_results = []
        p_results = []
        other_results = []
        
        for r in results:
            test_type = str(r.get('test_type', ''))
            if 'K' in test_type:
                k_results.append(r)
            elif 'P' in test_type:
                p_results.append(r)
            else:
                other_results.append(r)
        
        # Calculate target counts (at least 40% of each, minimum 1)
        target_per_type = max(1, int(top_k * 0.4))
        
        # Build balanced list
        balanced = []
        
        # Add technical (K-type)
        balanced.extend(k_results[:target_per_type])
        
        # Add behavioral (P-type)
        balanced.extend(p_results[:target_per_type])
        
        # If we have less than top_k, fill with best remaining
        if len(balanced) < top_k:
            remaining = top_k - len(balanced)
            # Combine remaining results (prioritize K and P leftovers)
            leftovers = k_results[target_per_type:] + p_results[target_per_type:] + other_results
            balanced.extend(leftovers[:remaining])
        
        # If we somehow have more than top_k, trim
        return balanced[:top_k]

    def get_balanced_recommendations(self, query: str, top_k: int = 5, api_key: str = None) -> List[Dict[str, Any]]:
        """
        Returns balanced recommendations mixing technical and behavioral assessments
        """
        # Get more results than needed to allow for balancing
        search_k = max(top_k * 3, 15)
        raw_results = self.search(query, top_k=search_k)
        
        # Detect query intent
        intent = self._detect_query_intent(query)
        logger.info(f"Query intent - Technical: {intent['technical']}, Behavioral: {intent['behavioral']}, Mixed: {intent['mixed']}")
        
        # For mixed queries, always balance
        if intent['mixed']:
            logger.info("Mixed query detected - applying balance logic")
            balanced_results = self._balance_results(raw_results, top_k)
            
            # Log the balance for debugging
            k_count = sum(1 for r in balanced_results if 'K' in str(r.get('test_type', '')))
            p_count = sum(1 for r in balanced_results if 'P' in str(r.get('test_type', '')))
            logger.info(f"Balanced results - K: {k_count}, P: {p_count}, Total: {len(balanced_results)}")
            
            return balanced_results
        
        # For non-mixed queries, just return top results
        logger.info("Single intent query - returning top results")
        return raw_results[:top_k]

if __name__ == "__main__":
    engine = RecommendationEngine("data/shl_catalog_final.csv")
    
    # Test with mixed query
    print("\n" + "="*50)
    print("TESTING MIXED QUERY: Java developer with good communication skills")
    print("="*50)
    recs = engine.get_balanced_recommendations("Java developer with good communication skills", top_k=5)
    for r in recs:
        print(f"✅ {r['name']} ({r['test_type']})")
    
    # Test with pure technical query
    print("\n" + "="*50)
    print("TESTING TECHNICAL QUERY: Python developer")
    print("="*50)
    recs = engine.get_balanced_recommendations("Python developer", top_k=5)
    for r in recs:
        print(f"✅ {r['name']} ({r['test_type']})")
    
    # Test with pure behavioral query
    print("\n" + "="*50)
    print("TESTING BEHAVIORAL QUERY: Team collaboration skills")
    print("="*50)
    recs = engine.get_balanced_recommendations("Team collaboration skills", top_k=5)
    for r in recs:
        print(f"✅ {r['name']} ({r['test_type']})") 