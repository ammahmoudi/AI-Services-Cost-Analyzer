"""
Extract billing info using regex on escaped JSON.
"""
import requests
import re
import json

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

print(f"Fetching: {playground_url}")

response = requests.get(playground_url, timeout=10)
html = response.text

print(f"Status: {response.status_code}")
print(f"HTML Length: {len(html)} chars")

# The data is in JavaScript like: {\"endpointBilling\":{\"endpoint\":\"...\",\"billing_unit\":\"...\",\"price\":0.04,...}}
# Let's extract it with proper handling of escaped quotes

# Pattern to find endpointBilling with all fields
pattern = r'\\"endpointBilling\\":\{[^}]*\\"endpoint\\":\\"([^"]*)\\"[^}]*\\"billing_unit\\":\\"([^"]*)\\"[^}]*\\"price\\":([0-9.]+)[^}]*\}'

matches = re.findall(pattern, html)

if matches:
    print(f"\nFound {len(matches)} billing entries:")
    for endpoint, unit, price in matches:
        print(f"\n  Endpoint: {endpoint}")
        print(f"  Unit: {unit}")
        print(f"  Price: ${price}")
else:
    print("\nNo matches with combined pattern")

# Try separate patterns
print("\n--- Separate field search ---")
endpoints = re.findall(r'\\"endpoint\\":\\"([^"]*bria/reimagine[^"]*)\\"', html)
units = re.findall(r'\\"billing_unit\\":\\"([^"]*)\\"', html)
prices = re.findall(r'\\"price\\":([0-9.]+)', html)

print(f"Endpoints found: {set(endpoints)}")
print(f"Units found: {set(units)}")
print(f"Prices found: {set(prices)}")

# If we have at least one of each, extract the billing info
if endpoints and units and prices:
    print(f"\nExtracted billing info:")
    print(f"  Endpoint: {endpoints[0]}")
    print(f"  Unit: {units[0]}")
    print(f"  Price: ${prices[0]}")
