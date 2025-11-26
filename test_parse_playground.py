"""
Parse pricing from playground HTML.
"""
import requests
import json
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

print(f"Fetching: {playground_url}")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

response = requests.get(playground_url, headers=headers, timeout=10)
html = response.text

print(f"HTML length: {len(html)} chars")

# Look for endpointBilling or publicEndpointBilling in the HTML
# The data is embedded in Next.js hydration scripts

# Try to find endpointBilling JSON
patterns = [
    r'"endpointBilling":\s*(\{[^}]+\})',
    r'"publicEndpointBilling":\s*(\{[^}]+\})',
    r'endpointBilling["\']:\s*(\{[^}]+\})',
]

for pattern in patterns:
    matches = re.findall(pattern, html)
    if matches:
        print(f"\nFound matches with pattern: {pattern}")
        for i, match in enumerate(matches[:3]):  # First 3 matches
            print(f"\nMatch {i+1}:")
            print(match[:300])
            try:
                # Try to parse as JSON
                data = json.loads(match)
                print("\nParsed as JSON:")
                print(json.dumps(data, indent=2))
            except:
                print("Could not parse as JSON")

# Also try simpler search
if '"price":' in html:
    print("\n\n=== Found 'price' in HTML ===")
    # Find context around price
    price_index = html.find('"price":')
    context = html[max(0, price_index-200):price_index+300]
    print(context)
