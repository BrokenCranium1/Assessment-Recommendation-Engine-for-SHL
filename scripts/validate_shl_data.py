import pandas as pd
import os

def validate_results(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    report = []
    
    report.append("SHL Data Validation Report")
    report.append("==========================")
    report.append(f"Total Records: {len(df)}")
    
    # Check for Target Count
    if len(df) == 377:
        report.append("[PASS] Target count (377) met.")
    else:
        report.append(f"[FAIL] Record count mismatch: {len(df)} (Target: 377)")
        
    # Check for Duplicates
    duplicates = df['url'].duplicated().sum()
    if duplicates == 0:
        report.append("[PASS] No duplicate URLs found.")
    else:
        report.append(f"[FAIL] {duplicates} duplicate URLs found.")
        
    # Check for Missing Descriptions
    missing_desc = df['description'].isna().sum() + (df['description'] == 'Unknown').sum()
    if missing_desc == 0:
        report.append("[PASS] All records have descriptions.")
    else:
        report.append(f"[WARNING] {missing_desc} records missing descriptions.")
        
    # Check for Multi-type badges
    multi_type = df['test_type'].str.contains(',').sum()
    report.append(f"Multi-type Assessments: {multi_type}")
    
    # Check for Remote/Adaptive indicators
    remote_yes = (df['remote_listing'] == 'Yes').sum()
    adaptive_yes = (df['adaptive_listing'] == 'Yes').sum()
    report.append(f"Remote Supported: {remote_yes}")
    report.append(f"Adaptive Supported: {adaptive_yes}")
    
    # Save report
    with open("validation_report.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print("Validation finished. Report saved to validation_report.txt")
    print("\n".join(report))

if __name__ == "__main__":
    validate_results("shl_catalog_final.csv")
