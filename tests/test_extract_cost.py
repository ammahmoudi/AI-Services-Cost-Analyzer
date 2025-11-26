"""
Extract pricing from "will cost" HTML pattern.
"""
import requests
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

print(f"Fetching: {playground_url}")

response = requests.get(playground_url, timeout=10)
html = response.text

# Pattern: "will cost $X.XX per Y"
cost_pattern = r'will cost.*?\$\s*([\d.]+)\s*.*?per\s*.*?<!-- -->([^<]+)<'
matches = re.findall(cost_pattern, html, re.DOTALL)

if matches:
    print("\nFound pricing:")
    for price, unit in matches:
        print(f"  Price: ${price}")
        print(f"  Unit: {unit.strip()}")
else:
    print("\nNo matches found")

# Alternative: just find the price after "will cost"
simple_pattern = r'will cost[^$]*\$\s*([\d.]+)'
simple_matches = re.findall(simple_pattern, html)
if simple_matches:
    print(f"\nSimple extraction: ${simple_matches[0]}")

# Also try to find "per image" or similar units
unit_pattern = r'per\s*<!-- -->.*?<!-- -->([^<]+)<'
units = re.findall(unit_pattern, html)
if units:
    print(f"Units found: {units[:3]}")
