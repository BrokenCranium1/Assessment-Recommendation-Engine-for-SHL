import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import RecommendationEngine
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate():
    # Load engine
    catalog_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'shl_catalog_final.csv'))
    engine = RecommendationEngine(catalog_path=catalog_path)
    
    # Load training data from Train-Set sheet
    train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Gen_AI Dataset.xlsx'))
    train_df = pd.read_excel(train_path, sheet_name='Train-Set')
    
    logger.info(f"Loaded {len(train_df)} training examples")
    
    recall_scores = []
    
    for idx, row in train_df.iterrows():
        query = row['Query']
        true_urls = [row['Assessment_url']]  # Single URL per row
        
        # Use the correct method name: get_balanced_recommendations
        recommendations = engine.get_balanced_recommendations(query, top_k=10)
        recommended_urls = [r['url'] for r in recommendations]
        
        # Calculate recall@10 with flexible matching
        found = 0
        for true_url in true_urls:
            true_slug = str(true_url).rstrip('/').split('/')[-1]
            for rec_url in recommended_urls:
                if true_slug in rec_url:
                    found = 1
                    break
        
        recall = found / len(true_urls) if true_urls else 0
        recall_scores.append(recall)
        
        logger.info(f"Query {idx+1}: Recall@10 = {recall:.4f}")
    
    mean_recall = sum(recall_scores) / len(recall_scores)
    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL MEAN RECALL@10: {mean_recall:.4f}")
    logger.info(f"{'='*50}")
    
    return mean_recall

def generate_test_predictions():
    """Generate predictions for test set"""
    catalog_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'shl_catalog_final.csv'))
    engine = RecommendationEngine(catalog_path=catalog_path)
    
    test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Gen_AI Dataset.xlsx'))
    test_df = pd.read_excel(test_path, sheet_name='Test-Set')
    
    predictions = []
    
    for idx, row in test_df.iterrows():
        query = row['Query']
        recommendations = engine.get_balanced_recommendations(query, top_k=10)
        
        for rec in recommendations:
            predictions.append({
                'query': query,
                'Assessment_url': rec['url']
            })
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_predictions.csv'))
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(pred_df)} predictions to {output_path}")

if __name__ == '__main__':
    mean_recall = evaluate()
    generate_test_predictions()