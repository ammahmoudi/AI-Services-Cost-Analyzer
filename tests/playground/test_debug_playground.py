"""
Test playground fetch with error visibility.
"""
import requests
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

print(f"Fetching: {playground_url}")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

response = requests.get(
    playground_url,
    headers=headers,
    timeout=10
)

print(f"Status: {response.status_code}")

if response.status_code != 200:
    print("Failed: non-200 status")
else:
    html = response.text
    print(f"HTML length: {len(html)}")
    
    # Extract fields
    endpoints = re.findall(r'\\"endpoint\\":\\"([^"]*' + re.escape(model_id) + r'[^"]*)\\"', html)
    units = re.findall(r'\\"billing_unit\\":\\"([^"]*)\\"', html)
    prices = re.findall(r'\\"price\\":([0-9.]+)', html)
    
    print(f"\nEndpoints found: {endpoints}")
    print(f"Units found: {units}")
    print(f"Prices found: {prices}")
    
    if endpoints and units and prices:
        billing_data = {
            'endpoint': endpoints[0],
            'billing_unit': units[0],
            'price': float(prices[0])
        }
        print(f"\n✅ Success:")
        print(f"  {billing_data}")
    else:
        print("\n❌ Failed: missing fields")
