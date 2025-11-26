"""
Test pricing_text extraction pattern.
"""
import requests
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

response = requests.get(playground_url, timeout=10)
html = response.text

print(f"HTML length: {len(html)}\n")

# Test the pattern from the code
cost_pattern = r'will cost[^<]*<[^>]*>\$[^<]*<[^>]*>([0-9.]+)[^<]*<[^>]*>[^<]*per[^<]*<[^>]*>([^<]+)<'
cost_matches = re.findall(cost_pattern, html)

if cost_matches:
    print("Pattern 1 matched:")
    for price_val, unit_val in cost_matches:
        pricing_text = f"will cost ${price_val} per {unit_val.strip()}"
        print(f"  {pricing_text}")
else:
    print("Pattern 1 did not match\n")
    
# Alternative: simpler pattern
simple_pattern = r'will cost<!-- --> <span[^>]*>\$<!-- -->([0-9.]+)</span> <!-- -->per <!-- -->([^<]+)<'
simple_matches = re.findall(simple_pattern, html)

if simple_matches:
    print("\nPattern 2 (simpler) matched:")
    for price_val, unit_val in simple_matches:
        pricing_text = f"will cost ${price_val} per {unit_val.strip()}"
        print(f"  {pricing_text}")
else:
    print("\nPattern 2 did not match")

# Show actual text around "will cost"
will_cost_idx = html.find("will cost")
if will_cost_idx != -1:
    print(f"\nActual text around 'will cost':")
    print(html[will_cost_idx:will_cost_idx + 200])
