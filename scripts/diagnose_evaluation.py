
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import RecommendationEngine
import pandas as pd
import json

print("="*60)
print("EVALUATION DIAGNOSTIC")
print("="*60)

# Initialize engine
catalog_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'shl_catalog_final.csv'))
print(f"\n📁 Catalog path: {catalog_path}")

try:
    engine = RecommendationEngine(catalog_path=catalog_path)
    print(f"\n✅ Engine loaded with {len(engine.df)} records")
except Exception as e:
    print(f"\n❌ Error loading engine: {e}")
    sys.exit(1)

# Load training data from Train-Set sheet
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Gen_AI Dataset.xlsx'))
print(f"\n📁 Training data path: {train_path}")

try:
    train_df = pd.read_excel(train_path, sheet_name='Train-Set')
    print(f"\n✅ Training data loaded: {len(train_df)} rows")
    print(f"Columns: {list(train_df.columns)}")
except Exception as e:
    print(f"\n❌ Error loading Train-Set sheet: {e}")
    sys.exit(1)

# Check first few training examples
print("\n📋 FIRST 3 TRAINING EXAMPLES:")
for i in range(min(3, len(train_df))):
    # Assuming columns are 'Query' and 'Assessment_url' based on PDF
    query = train_df.iloc[i]['Query'] if 'Query' in train_df.columns else train_df.iloc[i][0]
    url = train_df.iloc[i]['Assessment_url'] if 'Assessment_url' in train_df.columns else train_df.iloc[i][1]
    
    print(f"\nQuery {i+1}: {query}")
    print(f"URL: {url}")

# Check URL matches
print("\n🔍 CHECKING URL MATCHES:")
url_col = 'Assessment_url' if 'Assessment_url' in train_df.columns else train_df.columns[1]
urls_to_check = train_df[url_col].dropna().unique()[:5]

for url in urls_to_check:
    # Handle potential NaN or non-string values
    if pd.isna(url):
        continue
    
    url = str(url)
    url_slug = url.rstrip('/').split('/')[-1]
    
    # Try different matching strategies
    exact_matches = engine.df[engine.df['url'] == url]
    slug_matches = engine.df[engine.df['url'].str.contains(url_slug, na=False)]
    name_matches = engine.df[engine.df['name'].str.contains(url_slug.replace('-', ' '), case=False, na=False)]
    
    print(f"\nURL: {url}")
    print(f"Slug: {url_slug}")
    print(f"Exact matches: {len(exact_matches)}")
    print(f"Slug matches: {len(slug_matches)}")
    print(f"Name matches: {len(name_matches)}")
    
    if len(slug_matches) > 0:
        print(f"Sample match: {slug_matches.iloc[0]['name']}")

# Test first query manually
print("\n🧪 TESTING FIRST QUERY MANUALLY:")
test_query = train_df.iloc[0]['Query'] if 'Query' in train_df.columns else train_df.iloc[0][0]
print(f"Query: {test_query}")

try:
    results = engine.recommend(test_query, top_k=10)
    print(f"\nTop 10 results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}")
        
        # Show test type for balance check
        if 'test_type' in r:
            print(f"   Test type: {r['test_type']}")
    
    # Check if ground truth appears
    truth_url = str(train_df.iloc[0]['Assessment_url'] if 'Assessment_url' in train_df.columns else train_df.iloc[0][1])
    truth_slug = truth_url.rstrip('/').split('/')[-1]
    
    found = False
    for r in results:
        if truth_slug in r['url']:
            found = True
            print(f"\n✅ Found ground truth: {r['name']}")
            break
    
    if not found:
        print(f"\n❌ Ground truth not found in top 10")
        print(f"   Looking for slug: {truth_slug}")
        
except Exception as e:
    print(f"Error during recommendation: {e}")

print("\n" + "="*60)
