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
        # Fill NaN and create a rich combined_text for better keyword matching
        self.df['description'] = self.df['description'].fillna('')
        
        # ENHANCED: Add implicit keywords if they are missing from name/description
        def enrich_text(row):
            text = f"{row['name']} {row['description']} {row['test_type']}"
            name_lower = str(row['name']).lower()
            # If it's a technical test (K) and doesn't have common keywords, add them
            if 'K' in str(row['test_type']):
                if 'developer' not in name_lower and 'coding' not in name_lower:
                    text += " developer coding technical programming"
            return text
            
        self.df['combined_text'] = self.df.apply(enrich_text, axis=1)
        
        # Initialize BM25 for keyword search
        tokenized_corpus = [doc.lower().split() for doc in self.df['combined_text'].tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Expanded keywords for better intent detection
        self.technical_keywords = [
            'java', 'python', 'sql', 'javascript', 'coding', 'developer', 
            'programming', 'engineer', 'technical', 'backend', 'frontend',
            'full stack', 'database', 'api', 'algorithm', 'data structure',
            'aws', 'azure', 'cloud', 'devops', 'software', 'code', 'react',
            'angular', 'node', 'typescript', 'c#', 'c++', 'ruby', 'php'
        ]
        
        self.behavioral_keywords = [
            'collaborat', 'team', 'communicat', 'lead', 'interpersonal',
            'soft', 'behavior', 'personality', 'attitude', 'cultural',
            'teamwork', 'leadership', 'management', 'emotional', 'empathy',
            'conflict', 'negotiat', 'present', 'verbal', 'written', 'working information'
        ]
        
        # Model lazy loading setup...
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
        
        try:
            # Check for embeddings.pkl in multiple common locations
            possible_paths = [
                'data/embeddings.pkl',
                './data/embeddings.pkl',
                'embeddings.pkl',
                '/app/data/embeddings.pkl',
                os.path.join(os.path.dirname(__file__), 'data', 'embeddings.pkl'),
            ]
            
            embeddings_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
            if not embeddings_path:
                logger.error("❌ No embeddings file found! Falling back to slow computation...")
                return self._compute_embeddings_fallback()
            
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, dict) and 'embeddings' in data:
                self.embeddings = data['embeddings']
            else:
                self.embeddings = data
                
            self.embed_model = SentenceTransformer('paraphrase-albert-small-v2')
            
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(self.embeddings).astype('float32'))
            
            self._model_loaded = True
            logger.info(f"✅ Loaded {len(self.embeddings)} embeddings.")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            self._compute_embeddings_fallback()

    def _compute_embeddings_fallback(self):
        """Fallback method that computes embeddings (slow)"""
        logger.info("⚠️ Computing embeddings from scratch...")
        self.embed_model = SentenceTransformer('paraphrase-albert-small-v2')
        texts = self.df['combined_text'].tolist()
        self.embeddings = self.embed_model.encode(texts, show_progress_bar=True)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        self._model_loaded = True

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        self._lazy_load_model()
        query_lower = query.lower()
        
        # 1. Semantic Search (Dense)
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # Log top 3 Semantic matches
        logger.info(f"🔍 Top 3 Semantic matches for '{query}':")
        for i in range(min(3, len(indices[0]))):
            idx = indices[0][i]
            logger.info(f"  {i+1}. Dist: {distances[0][i]:.4f} | {self.df.iloc[idx]['name']}")
        
        # 2. Keyword Search (Sparse)
        tokenized_query = query_lower.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Log top 3 BM25 matches
        logger.info(f"🔍 Top 3 BM25 matches for '{query}':")
        for i in range(min(3, len(bm25_indices))):
            idx = bm25_indices[i]
            if bm25_scores[idx] > 0:
                logger.info(f"  {i+1}. Score: {bm25_scores[idx]:.4f} | {self.df.iloc[idx]['name']}")
        
        # 3. Hybrid Reranking with Keyword Boost
        combined_scores = {}
        
        # Robust Normalization:
        # Use min-max scaling over the returned distances to avoid tiny spreads
        # causing large relative differences that can over-favor certain items (e.g., Java).
        max_dist = float(np.max(distances)) if np.max(distances) > 0 else 1.0
        min_dist = float(np.min(distances))
        dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
        
        for i, idx in enumerate(indices[0]):
            # Convert distances into similarity in a stable [0,1] range
            normalized = (distances[0][i] - min_dist) / dist_range
            score = 1.0 - normalized
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.3
            
        max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
        for idx in bm25_indices:
            score = bm25_scores[idx] / max_bm25
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.7
            
        # --- NEW: APPLY KEYWORD & LANGUAGE BOOST TO ALL CANDIDATES ---
        # Iterate over all indices added to combined_scores so far (top semantic + top BM25)
        # Detect explicit language keywords from the query so we can strongly prioritize them.
        language_keywords = [
            "python",
            "java",
            "javascript",
            "typescript",
            "sql",
            "c++",
            "c#",
            "ruby",
            "php",
        ]
        query_languages = {w for w in tokenized_query if w in language_keywords}
        has_explicit_language = len(query_languages) > 0
        
        for idx in list(combined_scores.keys()):
            name_lower = str(self.df.iloc[idx]['name']).lower()
            combined_text_lower = str(self.df.iloc[idx]['combined_text']).lower()
            
            # General keyword / name boost
            for word in tokenized_query:
                if len(word) < 3: continue
                
                # Big boost for direct name match
                if word in name_lower:
                    combined_scores[idx] += 1.0
                    logger.info(f"🚀 NAME MATCH BOOST: '{word}' in '{self.df.iloc[idx]['name']}'")
                    break
                # Smaller boost for presence in enriched text (case where semantic missed it)
                elif word in combined_text_lower:
                    combined_scores[idx] += 0.3
                    logger.info(f"✨ TEXT MATCH BOOST: '{word}' in '{self.df.iloc[idx]['name']}'")
                    break
            
            # Additional explicit language prioritization:
            # If the query specifies a language (e.g., 'python'), give a large boost
            # to items whose name contains that language, and slightly down-weight
            # items that are strongly about other languages when they are not requested.
            if has_explicit_language:
                for lang in language_keywords:
                    in_query = lang in query_languages
                    in_name = lang in name_lower
                    in_text = lang in combined_text_lower
                    
                    if in_query and (in_name or in_text):
                        # Strong positive boost for exact language matches mentioned in the query
                        combined_scores[idx] += 3.0
                        logger.info(f"🔥 LANGUAGE PRIORITY BOOST: '{lang}' in '{self.df.iloc[idx]['name']}' (query specified this language)")
                    elif (not in_query) and in_name:
                        # Mild penalty for other languages not requested, to prevent Java from
                        # dominating when the user explicitly asked for Python, SQL, etc.
                        combined_scores[idx] -= 0.5
                        logger.info(f"⚖️ LANGUAGE DE-EMPHASIS: '{lang}' in '{self.df.iloc[idx]['name']}' (not requested in query)")
            
        # Sort and get top results
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"📊 Final Top Hybrid Scores (Query: '{query}'):")
        results = []
        for i, (idx, score) in enumerate(sorted_indices[:10]):
            item = self.df.iloc[idx].to_dict()
            item['score'] = float(score)
            results.append(item)
            if i < 10:
                logger.info(f"  {i+1}. Total: {score:.4f} | {item['name']}")
            
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