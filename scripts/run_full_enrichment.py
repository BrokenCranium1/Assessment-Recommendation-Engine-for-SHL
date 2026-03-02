import pandas as pd
import time
from bs4 import BeautifulSoup
import re
import requests
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enrichment_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

session = requests.Session()
session.headers.update(HEADERS)

def get_page(url):
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def extract_description(soup):
    # Try headers at multiple levels
    for header in soup.find_all(['h2', 'h3', 'h4'], 
                                string=re.compile('Description|Overview|About|Details', re.I)):
        desc_parts = []
        next_elem = header.find_next_sibling()
        while next_elem and next_elem.name not in ['h2', 'h3', 'h4']:
            if next_elem.name == 'p':
                text = next_elem.get_text(strip=True)
                if text and len(text) > 20:  # Ignore short fragments
                    desc_parts.append(text)
            next_elem = next_elem.find_next_sibling()
        if desc_parts:
            return ' '.join(desc_parts)
    
    # Fallback to class-based selectors
    for class_name in ['product-description', 'description', 'product-view__description', 'product-detail__description']:
        desc_div = soup.find('div', class_=class_name)
        if desc_div:
            return desc_div.get_text(strip=True)
    
    return 'Unknown'

def enrich():
    logger.info("Starting Batch Enrichment Fix")
    df = pd.read_csv('data/shl_catalog_final.csv')
    
    total = len(df)
    for i, row in df.iterrows():
        url = row['url']
        logger.info(f"[{i+1}/{total}] Processing: {url}")
        
        soup = get_page(url)
        if soup:
            # Update Description
            df.at[i, 'description'] = extract_description(soup)
            
            full_text = soup.get_text()
            
            # Update Duration (using broad patterns)
            duration_patterns = [
                r'(?:Completion Time|Time|Duration)[:=]\s*(\d+)\s*(?:min|minutes?)',
                r'Approximate Completion Time in minutes\s*=\s*(\d+)',
                r'(\d+)\s*min'
            ]
            for p in duration_patterns:
                duration_match = re.search(p, full_text, re.I)
                if duration_match:
                    df.at[i, 'duration'] = duration_match.group(1)
                    break 
            
            # Update Remote/Adaptive if detail page has better info
            remote_match = re.search(r'Remote Testing:\s*(Yes|No)', full_text, re.I)
            if remote_match:
                df.at[i, 'remote_support'] = remote_match.group(1).capitalize()
                
            adaptive_match = re.search(r'(?:Adaptive|IRT):\s*(Yes|No)', full_text, re.I)
            if adaptive_match:
                df.at[i, 'adaptive_support'] = adaptive_match.group(1).capitalize()
        
        # Reduced sleep for faster fix, but still being respectful
        time.sleep(1)
        
        # Intermediate Save every 50 records
        if (i+1) % 50 == 0:
            df.to_csv('data/shl_catalog_final.csv', index=False)
            logger.info("Intermediate save completed.")

    # Final Save
    df.to_csv('data/shl_catalog_final.csv', index=False)
    
    # Generate updated validation report
    with open('data/validation_report.txt', 'w') as f:
        f.write(f"SHL Enrichment Fix Validation Report\n")
        f.write(f"=============================\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Descriptions found: {df[df['description'] != 'Unknown'].shape[0]}\n")
        f.write(f"Durations found: {df['duration'].notna().sum()}\n")
        f.write(f"Field completion percentages:\n{df.replace('Unknown', None).notna().mean() * 100}\n")
        
        # Test type distribution
        all_types = []
        for val in df['test_type']:
            if val != 'Unknown':
                all_types.extend(val.split(','))
        type_counts = pd.Series(all_types).value_counts()
        f.write(f"\nTest type distribution:\n{type_counts}\n")
        
        f.write(f"\nRandom Sample of 5:\n")
        f.write(df.sample(min(5, len(df))).to_string())

    logger.info("Enrichment fix complete!")

if __name__ == "__main__":
    enrich()
