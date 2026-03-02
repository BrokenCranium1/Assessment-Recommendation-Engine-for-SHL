import pandas as pd
import numpy as np
from engine import RecommendationEngine

def get_slug(url: str) -> str:
    if not url or not isinstance(url, str): return ""
    return url.strip("/").split("/")[-1].lower()

def evaluate(engine, weights=[0.5, 0.5]):
    df_train = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name=0)
    recalls = []
    
    # Temporarily override search weights
    original_search = engine.search
    def custom_search(query, top_k=10):
        # 1. Semantic Search
        query_embedding = engine.embed_model.encode([query])
        distances, indices = engine.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # 2. Keyword Search
        tokenized_query = query.lower().split()
        bm25_scores = engine.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        combined_scores = {}
        max_dist = np.max(distances) if np.max(distances) > 0 else 1
        for i, idx in enumerate(indices[0]):
            score = 1 - (distances[0][i] / max_dist)
            combined_scores[idx] = combined_scores.get(idx, 0) + score * weights[0]
            
        max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
        for idx in bm25_indices:
            score = bm25_scores[idx] / max_bm25
            combined_scores[idx] = combined_scores.get(idx, 0) + score * weights[1]
            
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in sorted_indices[:top_k]:
            item = engine.df.iloc[idx].to_dict()
            results.append(item)
        return results

    engine.search = custom_search
    
    for _, row in df_train.iterrows():
        query = row.get('Query') or row.get('query')
        gt_raw = row.get('Assessment_url') or row.get('assessment_url') or ""
        gt_urls = [u.strip() for u in str(gt_raw).split(',')] if gt_raw else []
        
        preds = engine.search(query, top_k=10)
        # Use slug matching
        gt_slugs = set([get_slug(u) for u in gt_urls if u])
        pred_slugs = [get_slug(p['url']) for p in preds]
        
        relevant_retrieved = len(gt_slugs & set(pred_slugs))
        recall = relevant_retrieved / len(gt_slugs) if gt_slugs else 0
        recalls.append(recall)
        
    return np.mean(recalls)

engine = RecommendationEngine("data/shl_catalog_final.csv")

results = []
# 1. Semantic Only
results.append(("Semantic Only", evaluate(engine, [1.0, 0.0])))
# 2. Hybrid 50/50
results.append(("Hybrid (0.5/0.5)", evaluate(engine, [0.5, 0.5])))
# 3. Hybrid 30/70
results.append(("Hybrid (0.3/0.7)", evaluate(engine, [0.3, 0.7])))
# 4. Hybrid 70/30
results.append(("Hybrid (0.7/0.3)", evaluate(engine, [0.7, 0.3])))

print("\n" + "="*40)
print("FINAL OPTIMIZATION SUMMARY")
print("="*40)
print(f"{'Configuration':<25} | {'Mean Recall@10'}")
print("-" * 40)
for config, recall in results:
    print(f"{config:<25} | {recall:.4f}")
print("="*40)
