"""
Search for pricing patterns in playground HTML.
"""
import requests
import json
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

print(f"Fetching: {playground_url}")

response = requests.get(playground_url, timeout=10)
html = response.text

# Search for $0.04 or similar pricing patterns
price_patterns = [
    r'\$\s*([\d.]+)\s*per',
    r'"price":\s*([\d.]+)',
    r'billing_unit["\']:\s*["\']([^"\']+)',
]

for pattern in price_patterns:
    matches = re.findall(pattern, html)
    if matches:
        print(f"\nPattern: {pattern}")
        print(f"Matches: {matches[:5]}")  # First 5

# Find all instances of "endpoint" with context
endpoint_matches = [m.start() for m in re.finditer(r'"endpoint"', html)]
print(f"\n\nFound {len(endpoint_matches)} occurrences of '\"endpoint\"'")

if endpoint_matches:
    # Look at first occurrence
    idx = endpoint_matches[0]
    context = html[max(0, idx-100):idx+500]
    print("\nFirst occurrence context:")
    print(context)

# Look for the specific text from your curl output: "will cost"
if "will cost" in html:
    cost_idx = html.find("will cost")
    context = html[max(0, cost_idx-50):cost_idx+100]
    print("\n\n=== 'will cost' context ===")
    print(context)
    
# Search for endpointBilling as object
billing_pattern = r'"endpointBilling":\{[^}]+\}'
matches = re.findall(billing_pattern, html)
if matches:
    print(f"\n\n=== Found endpointBilling objects: {len(matches)} ===")
    for i, match in enumerate(matches[:2]):
        print(f"\n{i+1}. {match}")
