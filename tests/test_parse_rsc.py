"""
Parse React Server Component (RSC) response to extract billing info.
"""
import requests
import json
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}"

print(f"POSTing to: {playground_url}")

headers = {
    "Accept": "text/x-component",
    "Content-Type": "text/plain;charset=UTF-8",
    "Next-Action": "4eed2a03c59a967e3e2d13aa234e5bca44df682b",
}

response = requests.post(
    playground_url,
    headers=headers,
    data='[{"ensureSignedIn":false}]',
    timeout=10
)

print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")

rsc_data = response.text
print(f"\nRSC Length: {len(rsc_data)} chars")
print(f"Response preview: {rsc_data[:500]}")

# Look for endpointBilling in the RSC payload
billing_pattern = r'"endpointBilling":\{([^}]+)\}'
matches = re.findall(billing_pattern, rsc_data)

if matches:
    print("\nFound endpointBilling:")
    for match in matches[:3]:
        print(f"  {match}")
        
# More robust: extract the entire endpointBilling object with nested braces
billing_obj_pattern = r'"endpointBilling":\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
billing_objs = re.findall(billing_obj_pattern, rsc_data)

if billing_objs:
    print("\n\nFull endpointBilling objects:")
    for obj in billing_objs[:2]:
        print(f"\n{obj}")
        # Try to parse as JSON
        try:
            json_str = "{" + obj
            billing_data = json.loads(json_str)
            print(f"\nParsed billing:")
            print(f"  Price: ${billing_data['endpointBilling'].get('price')}")
            print(f"  Unit: {billing_data['endpointBilling'].get('billing_unit')}")
            print(f"  Endpoint: {billing_data['endpointBilling'].get('endpoint')}")
        except Exception as e:
            print(f"  Parse error: {e}")

# Alternative: look for specific billing fields
print("\n\nDirect field search:")
price_matches = re.findall(r'"price":([\d.]+)', rsc_data)
unit_matches = re.findall(r'"billing_unit":"([^"]+)"', rsc_data)

if price_matches and unit_matches:
    print(f"  Prices found: {set(price_matches)}")
    print(f"  Units found: {set(unit_matches)}")
