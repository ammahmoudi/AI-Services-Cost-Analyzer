"""
Simple approach: Extract billing info from playground page HTML.
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

# Look for embedded JSON data in script tags
script_pattern = r'<script[^>]*>([^<]+endpointBilling[^<]+)</script>'
scripts = re.findall(script_pattern, html, re.DOTALL)

if scripts:
    print(f"\nFound {len(scripts)} scripts with endpointBilling")
    for i, script in enumerate(scripts[:2]):
        print(f"\n--- Script {i+1} (first 500 chars) ---")
        print(script[:500])

# Also look for "will cost" pattern
cost_matches = re.findall(r'will cost[^$]*\$\s*([\d.]+)', html)
if cost_matches:
    print(f"\n'will cost' prices: {cost_matches}")

# Look for per X pattern
per_matches = re.findall(r'\$[\d.]+\s*per\s*(\w+)', html)
if per_matches:
    print(f"'per' units: {per_matches}")

# Direct search for billing_unit
billing_unit_matches = re.findall(r'"billing_unit"\s*:\s*"([^"]+)"', html)
if billing_unit_matches:
    print(f"\nbilling_unit values: {set(billing_unit_matches)}")

# Direct search for price
price_json_matches = re.findall(r'"price"\s*:\s*([\d.]+)', html)
if price_json_matches:
    print(f"price values: {set(price_json_matches)}")

# Look for the full endpointBilling JSON object (with proper nesting)
# The pattern needs to handle nested objects
def extract_billing_info(text):
    """Extract endpointBilling JSON object from text."""
    # Find the start of endpointBilling
    start_pattern = r'"endpointBilling"\s*:\s*\{'
    matches = list(re.finditer(start_pattern, text))
    
    results = []
    for match in matches:
        start_idx = match.start()
        # Find the matching closing brace
        brace_count = 0
        in_string = False
        escape = False
        
        # Start from the opening brace
        json_start = text.find('{', start_idx)
        if json_start == -1:
            continue
            
        for i in range(json_start, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
                
            if char == '\\':
                escape = True
                continue
                
            if char == '"' and not escape:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the matching close brace
                        json_str = text[json_start:i+1]
                        try:
                            parsed = json.loads(json_str)
                            results.append(parsed)
                        except json.JSONDecodeError as e:
                            pass
                        break
    return results

billing_objects = extract_billing_info(html)
if billing_objects:
    print(f"\nExtracted {len(billing_objects)} endpointBilling objects:")
    for obj in billing_objects[:2]:
        print(f"\n{json.dumps(obj, indent=2)}")
else:
    print("\nNo endpointBilling objects extracted")
