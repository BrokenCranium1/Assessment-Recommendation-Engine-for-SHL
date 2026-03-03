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
import pickle
import traceback  # Added for better error logging

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
        
        # Keywords for intent detection (fallback only)
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
        """Load pre-computed embeddings instead of computing them from scratch"""
        if self._model_loaded:
            return
            
        logger.info("🔄 Loading pre-computed embeddings...")
        
        # 🔍 DEBUGGING - SHOW ALL FILES AND PATHS
        import os
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in root directory: {os.listdir('.')}")
        
        # Check data directory
        if os.path.exists('data'):
            logger.info(f"Files in ./data directory: {os.listdir('data')}")
        else:
            logger.info("❌ ./data directory does NOT exist!")
            
        # Check app directory (Docker often uses /app)
        if os.path.exists('/app'):
            logger.info(f"Files in /app directory: {os.listdir('/app')}")
            if os.path.exists('/app/data'):
                logger.info(f"Files in /app/data directory: {os.listdir('/app/data')}")
        
        try:
            # List of possible paths to try
            possible_paths = [
                'data/embeddings.pkl',
                './data/embeddings.pkl',
                'embeddings.pkl',
                './embeddings.pkl',
                '/app/data/embeddings.pkl',
                '/app/embeddings.pkl',
                os.path.join(os.path.dirname(__file__), 'data', 'embeddings.pkl'),
                os.path.join(os.path.dirname(__file__), 'embeddings.pkl'),
            ]
            
            embeddings_path = None
            for path in possible_paths:
                logger.info(f"Checking path: {path}")
                if os.path.exists(path):
                    embeddings_path = path
                    logger.info(f"✅ Found embeddings at: {path}")
                    break
            
            if embeddings_path is None:
                logger.error("❌ No embeddings file found in any of the checked paths!")
                logger.error("Falling back to slow embedding computation...")
                return self._compute_embeddings_fallback()
            
            # Load pre-computed embeddings
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                
            # Handle different pickle formats
            if isinstance(data, dict) and 'embeddings' in data:
                self.embeddings = data['embeddings']
            elif isinstance(data, np.ndarray):
                self.embeddings = data
            else:
                logger.error(f"❌ Unexpected pickle format: {type(data)}")
                return self._compute_embeddings_fallback()
                
            logger.info(f"✅ Loaded {len(self.embeddings)} pre-computed embeddings")
            
            # Still need the model for encoding queries
            logger.info("🔄 Loading ML model for query encoding...")
            self.embed_model = SentenceTransformer('paraphrase-albert-small-v2')
            
            # Initialize FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(self.embeddings).astype('float32'))
            
            self._model_loaded = True
            memory_used = self.embeddings.nbytes / 1024 / 1024
            logger.info(f"✅ Ready! Embeddings use {memory_used:.2f}MB in memory")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            logger.info("Falling back to slow embedding computation...")
            self._compute_embeddings_fallback()

    def _compute_embeddings_fallback(self):
        """Fallback method that computes embeddings (slow) - only used if pre-computed file missing"""
        logger.info("⚠️ Computing embeddings from scratch (this will be slow)...")
        
        # Load the small model
        self.embed_model = SentenceTransformer('paraphrase-albert-small-v2')
        
        # Process in very small batches to manage memory
        batch_size = 16
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
        Returns intelligent recommendations using Gemini for ALL queries
        """
        # Get more results than needed
        search_k = max(top_k * 3, 15)
        raw_results = self.search(query, top_k=search_k)
        
        # Log what we found before Gemini
        python_count = sum(1 for r in raw_results[:20] if 'python' in r['name'].lower())
        java_count = sum(1 for r in raw_results[:20] if 'java' in r['name'].lower())
        logger.info(f"Before Gemini - Python: {python_count}, Java: {java_count} in top 20")
        
        # If no API key, fall back to keyword-based balancing
        if not api_key:
            logger.warning("⚠️ No Gemini API key - using keyword fallback")
            intent = self._detect_query_intent(query)
            if intent['mixed']:
                return self._balance_results(raw_results, top_k)
            return raw_results[:top_k]
        
        # USE GEMINI FOR INTELLIGENT RERANKING!
        try:
            logger.info("🤖 Using Gemini for intelligent reranking")
            genai.configure(api_key=api_key)
            
            # Using Gemini 2.5 Flash-Lite
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            # Prepare RICH context with FULL details including descriptions
            context_items = raw_results[:20]
            context_parts = []
            for i, r in enumerate(context_items, 1):
                # Truncate description but keep enough for language detection
                desc = r['description'][:300] if r['description'] else "No description available"
                context_parts.append(
                    f"Assessment {i}:\n"
                    f"Name: {r['name']}\n"
                    f"Type: {r['test_type']}\n"
                    f"Description: {desc}\n"
                    f"URL: {r['url']}"
                )
            context = "\n\n".join(context_parts)
            
            # Log sample for debugging
            logger.info(f"Sending {len(context_items)} assessments to Gemini")
            if context_items:
                logger.info(f"Sample: {context_items[0]['name']} - {context_items[0]['description'][:100]}...")
            
            prompt = f"""You are an expert SHL assessment consultant. Your task is to recommend the most relevant assessments for a given query.

QUERY: "{query}"

🔴 CRITICAL INSTRUCTION - READ CAREFULLY:
The query mentions a specific programming language. You MUST prioritize assessments that test that language.

For example:
- If query contains "Python", assessments with "Python" in the name or description are HIGHEST priority
- If query contains "Java", assessments with "Java" in the name or description are HIGHEST priority
- If query contains "JavaScript", prioritize JavaScript assessments
- If query contains "SQL", prioritize SQL/database assessments

Here are the available assessments with their FULL details (name, type, description, URL):

{context}

INSTRUCTIONS:
1. First, identify the primary skill/language in the query: "{query}"
2. Select the {top_k} most relevant assessments
3. Return ONLY a JSON array of the assessment URLs in order of relevance
4. If the query mentions both technical and soft skills, ensure a balanced mix:
   - Technical assessments (Type K)
   - Behavioral assessments (Type P)
5. CRITICAL: Assessments containing the exact language name in their title or description MUST be prioritized

Example output format: ["url1", "url2", "url3", "url4", "url5"]

Return only the JSON array, no other text."""
            
            response = model.generate_content(prompt)
            
            # Parse JSON from response
            text = response.text.strip()
            # Clean markdown if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            recommended_urls = json.loads(text)
            
            # Map URLs back to full records
            url_to_record = {r['url']: r for r in raw_results}
            gemini_results = []
            for url in recommended_urls[:top_k]:
                if url in url_to_record:
                    gemini_results.append(url_to_record[url])
            
            if gemini_results:
                # Log what Gemini returned
                python_final = sum(1 for r in gemini_results if 'python' in r['name'].lower())
                java_final = sum(1 for r in gemini_results if 'java' in r['name'].lower())
                logger.info(f"✅ Gemini returned {len(gemini_results)} recommendations - Python: {python_final}, Java: {java_final}")
                return gemini_results
            else:
                logger.warning("⚠️ Gemini returned no valid URLs, using fallback")
                
        except Exception as e:
            logger.error(f"❌ Gemini reranking failed: {e}")
            traceback.print_exc()
        
        # Fallback to keyword balancing if Gemini fails
        logger.warning("⚠️ Falling back to keyword balancing")
        intent = self._detect_query_intent(query)
        if intent['mixed']:
            return self._balance_results(raw_results, top_k)
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