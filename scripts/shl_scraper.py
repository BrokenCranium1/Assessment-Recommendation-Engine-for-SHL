import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import logging
import random
import os

# Setup Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://www.shl.com"
CATALOG_URL_TEMPLATE = "https://www.shl.com/solutions/products/product-catalog/?start={}"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

session = requests.Session()
session.headers.update(HEADERS)

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

def get_page(url):
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

def analyze_structure(soup):
    logger.info("Phase 1: Structure Analysis")
    tables = soup.find_all('div', class_='custom__table-responsive')
    valid_tables = []
    
    for table in tables:
        # Check all th entries for the target text
        headers = table.find_all('th')
        header_texts = [h.get_text(strip=True) for h in headers]
        logger.info(f"Found table with headers: {header_texts}")
        
        is_individual = any("Individual Test Solutions" in h for h in header_texts)
        is_job = any("Pre-packaged Job Solutions" in h for h in header_texts)
        
        if is_individual:
            valid_tables.append(table)
            logger.info("-> Identified as 'Individual Test Solutions' table")
        elif is_job:
            logger.info("-> Identified as 'Pre-packaged Job Solutions' table (to be skipped)")
            
    return valid_tables

def extract_listing_data(valid_tables, seen_urls):
    page_results = []
    for table in valid_tables:
        rows = table.find_all('tr')
        logger.info(f"Processing table with {len(rows)} rows")
        for i, row in enumerate(rows):
            if row.find('th'): continue
            
            
            # Column 1: Name and URL
            name_cell = row.select_one('.custom__table-heading__title')
            if not name_cell:
                cols = row.find_all('td')
                if cols: name_cell = cols[0]
                else: continue
                
            link = name_cell.find('a')
            if not link: continue
            
            url = BASE_URL + link['href'] if not link['href'].startswith('http') else link['href']
            url = url.split('?')[0]
            
            if url in seen_urls: continue
            seen_urls.add(url)
            
            name = name_cell.get_text(strip=True)
            
            # Column 2: Remote
            remote = "No"
            # Remote is 2nd column (index 1)
            cols = row.find_all('td')
            if len(cols) > 1:
                remote_cell = cols[1]
                if remote_cell.select_one('.catalogue__circle.-yes') or "-yes" in str(remote_cell):
                    remote = "Yes"
                
            # Column 3: Adaptive
            adaptive = "No"
            if len(cols) > 2:
                adaptive_cell = cols[2]
                if adaptive_cell.select_one('.catalogue__circle.-yes') or "-yes" in str(adaptive_cell):
                    adaptive = "Yes"
                
            # Column 4: Test Types
            test_types = []
            if len(cols) > 3:
                badge_cell = cols[3]
                badges = badge_cell.select('.product-catalogue__key')
                for b in badges:
                    val = b.get_text(strip=True).upper()
                    if val in ['K','P','S','A','E'] and val not in test_types:
                        test_types.append(val)
            
            test_type_str = ','.join(test_types) if test_types else 'Unknown'
            
            logger.info(f"Extracted: {name} | Remote: {remote} | Adaptive: {adaptive} | Types: {test_type_str}")
            
            page_results.append({
                'name': name,
                'url': url,
                'remote_listing': remote,
                'adaptive_listing': adaptive,
                'test_type': test_type_str
            })
    return page_results

def quality_gate(results):
    logger.info("Phase 3: Quality Gate Check")
    multi_type = any(',' in r['test_type'] for r in results)
    remote_count = sum(1 for r in results if r['remote_listing'] == 'Yes')
    adaptive_count = sum(1 for r in results if r['adaptive_listing'] == 'Yes')
    bad_urls = any(re.search(r'job-solutions|pre-packaged', r['url'], re.I) for r in results)
    
    logger.info(f"Quality Stats: Multi-type: {multi_type}, Remote Yes: {remote_count}, Adaptive Yes: {adaptive_count}, Bad URLs: {bad_urls}")
    
    if not multi_type:
        logger.error("Quality Gate Failed: No multi-type assessments found.")
        return False
    if remote_count < 10:
        logger.error("Quality Gate Failed: Not enough remote assessments.")
        return False
    if adaptive_count < 2:
        logger.error("Quality Gate Failed: Not enough adaptive assessments.")
        return False
    if bad_urls:
        logger.error("Quality Gate Failed: Prohibited URLs found in results.")
        return False
        
    logger.info("Quality Gate Passed!")
    return True

def enhance_details(records):
    logger.info("Phase 4: Detail Enhancement")
    enhanced_records = []
    for i, record in enumerate(records):
        logger.info(f"[{i+1}/{len(records)}] Enhancing: {record['url']}")
        soup = get_page(record['url'])
        
        detail_remote = None
        detail_adaptive = None
        description = ""
        duration = ""
        
        if soup:
            # Description
            description = extract_description(soup)
            
            # Regex searches in whole text
            full_text = soup.get_text()
            
            remote_match = re.search(r'Remote Testing:\s*(Yes|No)', full_text, re.I)
            if remote_match:
                detail_remote = remote_match.group(1).capitalize()
                
            adaptive_match = re.search(r'(?:Adaptive|IRT):\s*(Yes|No)', full_text, re.I)
            if adaptive_match:
                detail_adaptive = adaptive_match.group(1).capitalize()
                
            # Expanded Duration Patterns
            duration_patterns = [
                r'(?:Completion Time|Time|Duration)[:=]\s*(\d+)\s*(?:min|minutes?)',
                r'Approximate Completion Time in minutes\s*=\s*(\d+)',
                r'(\d+)\s*min'
            ]
            for p in duration_patterns:
                duration_match = re.search(p, full_text, re.I)
                if duration_match:
                    duration = duration_match.group(1)
                    break 
        
        # Progressive Enhancement
        record['description'] = description
        record['duration'] = duration
        record['remote_support'] = detail_remote if detail_remote else record['remote_listing']
        record['adaptive_support'] = detail_adaptive if detail_adaptive else record['adaptive_listing']
        
        # Cleanup temp fields
        del record['remote_listing']
        del record['adaptive_listing']
        
        enhanced_records.append(record)
        time.sleep(3)
        
    return enhanced_records

def main():
    seen_urls = set()
    results = []
    page_num = 0
    consecutive_empty = 0
    
    # Phase 1 & 2
    logger.info("Starting Scraper Execution")
    
    while len(results) < 377:
        url = CATALOG_URL_TEMPLATE.format(page_num * 12)
        logger.info(f"Fetching catalog page {page_num + 1}: {url}")
        
        soup = get_page(url)
        if not soup:
            break
            
        valid_tables = analyze_structure(soup)
        page_data = extract_listing_data(valid_tables, seen_urls)
        
        if not page_data:
            consecutive_empty += 1
            logger.warning(f"No new items found on page {page_num + 1}")
        else:
            consecutive_empty = 0
            results.extend(page_data)
            logger.info(f"Added {len(page_data)} items. Total: {len(results)}")
            
        # Phase 3: Quality Gate
        if page_num == 2: # After 3 pages (0, 1, 2)
            if not quality_gate(results):
                logger.error("Exiting due to Quality Gate failure.")
                return

        if consecutive_empty >= 3:
            logger.info("Stopping: 3 consecutive pages with no new items.")
            break
            
        # Check for next page link as extra safeguard
        if not soup.find('a', string=re.compile(r'Next', re.I)):
             # In some cases 'Next' might be an icon, but our start parameter logic is primary
             pass

        page_num += 1
        time.sleep(random.uniform(2, 4))
    
    # Phase 4: Enhancement
    results = enhance_details(results)
    
    # Phase 5: Export & Validation
    df = pd.DataFrame(results)
    df.to_csv('data/shl_catalog_final.csv', index=False)
    
    with open('data/validation_report.txt', 'w') as f:
        f.write(f"SHL Scraper Validation Report\n")
        f.write(f"=============================\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Field completion percentages:\n{df.notna().mean() * 100}\n")
        
        # Test type distribution
        all_types = []
        for row in df['test_type']:
            if row != 'Unknown':
                all_types.extend(row.split(','))
        type_counts = pd.Series(all_types).value_counts()
        f.write(f"\nTest type distribution:\n{type_counts}\n")
        
        multi_count = df['test_type'].str.contains(',').sum()
        f.write(f"\nMulti-type assessments: {multi_count}\n")
        
        f.write(f"\nRandom Sample of 10:\n")
        f.write(df.sample(min(10, len(df))).to_string())

    logger.info(f"Scraping complete. Saved {len(df)} records to data/shl_catalog_final.csv")
    
    # Final Criteria Check
    try:
        assert len(df) >= 377
        assert df['remote_support'].isin(['Yes','No']).all()
        assert df['adaptive_support'].isin(['Yes','No']).all()
        assert df['test_type'].str.contains(',').sum() > 0
        assert not df['url'].str.contains('job-solutions|pre-packaged', case=False).any()
        logger.info("✓ All quality checks passed")
    except AssertionError as e:
        logger.error(f"Final Quality Check Failed: {e}")

if __name__ == "__main__":
    main()
