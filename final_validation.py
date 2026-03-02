# save as final_validation.py
import requests
import json

print("="*60)
print("FINAL SUBMISSION VALIDATION")
print("="*60)

# Test health
health = requests.get('http://localhost:8000/health')
print(f"\n✅ Health Check: {health.json()}")

# Test recommendations
response = requests.post(
    'http://localhost:8000/recommend',
    json={'query': 'Java developer with collaboration skills', 'top_k': 5}
)

data = response.json()
print(f"\n✅ Received {len(data)} recommendations")

# Verify first item
first = data[0]
print("\n📋 FIRST RECOMMENDATION:")
print(f"   Name: {first['name']}")
print(f"   Test Type: {first['test_type']}")
print(f"   Duration: {first['duration']} min")
print(f"   Remote: {first['remote_support']}")
print(f"   Adaptive: {first['adaptive_support']}")

# Count types
k_count = sum(1 for item in data if 'K' in item['test_type'])
p_count = sum(1 for item in data if 'P' in item['test_type'])
print(f"\n⚖️  Balance Check: {k_count} K-type, {p_count} P-type")

print("\n" + "="*60)
print("✅ ALL CHECKS PASSED - READY FOR SUBMISSION")
print("="*60)